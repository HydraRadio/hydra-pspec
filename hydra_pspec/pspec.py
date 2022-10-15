
import numpy as np
import scipy as sp
from scipy.stats import mode
from scipy.signal.windows import blackmanharris as BH
from scipy.stats import invgamma
from scipy.optimize import minimize, Bounds

from . import utils
import os, time


def GCR(dat, w, S, N, realisations=1, inpaint_mode='inpaint', dat2=None, bla=None, poolmap=False):
    """
    Returns a number of constrained realizations for a flagged data vector with signal prior S and noise prior N,
    following the Gaussian constrained realization equation. 
    
    Important note: any given realization will not match the data in the unflagged region. If the desire is to
    in-paint flagged regions, you will need to select only this region of the output vector.
    
    Parameters:
	    d (array_like):
	    	Complex data vector. 
	    w (array_like):
	    	Flagging/mask vector (1 for unflagged data, 0 for flagged data).
	    S (array_like):
	    	Signal prior covariance matrix. Has the same dimension as the data 
	    	vector. May only be real-valued.   
	    N (array_like):
	    	Noise prior covariance matrix. Has the same dimension as the data 
	    	vector. May only be real-valued.
	    realisations (int):
	    	Number of realisations to return.
	    inpaint_mode (str):
	    	How to handle flagged regions. If set to `'inpaint'`, replace 
	    	flagged regions with the in-painted solution (but leave the 
	    	unflagged regions as the input data). If `'subtract'`, subtract 
	    	the GCR solution from the data. Otherwise, return the GCR solution.
	
	Returns:
		gcr_soln (array_like):
			GCR solution, in-painted data, or filtered data, depending on the 
			`inpaint_mode` setting.
    """
    s = len(dat)
    d = dat.reshape((1,max(s,len(dat.T))))
    nbaselines = 1
    if bla is not None:
        nbaselines = bla.shape[0] # n_rows (each redundant baseline is one row)
        d = np.sum(bla, axis=0).reshape((1,max(len(dat),len(dat.T))))
        
    if dat2 is not None:
        d2 = dat2.reshape((1,max(len(dat2),len(dat2.T))))
        d = (d+d2)/2

    if np.iscomplexobj(d) or np.iscomplexobj(S) or np.iscomplexobj(N):
        complex_data=True
    else:
    	complex_data=False
            
    Sh = sp.linalg.sqrtm( S )
    Nh = sp.linalg.sqrtm( N )
    Si = np.linalg.inv( S )
    Ni = w.T*np.linalg.inv( N )*w
    Sih = sp.linalg.sqrtm( Si )
    Nih = sp.linalg.sqrtm( Ni )
           
    A = nbaselines*Sh @ Ni @ Sh  + np.eye(s)
    Ai = np.linalg.inv(A)
    b = Sh @ Ni @ (w*d).T
        
    wiener, _ = sp.sparse.linalg.cg(A, b, maxiter=1e5, M=Ai) # Wiener / max-likelihood solution
    Wnr = Sh@wiener
    
    # Create empty solution array
    if complex_data:
    	solns = np.zeros((realisations,s), dtype=complex) # array for solutions to GCR equation
    else:
    	solns = np.zeros((realisations,s))
    
    if complex_data:
    	# Complex-valued version of the calculation
    	for i in range(realisations):
            omi, omj = np.random.randn(s,1),np.random.randn(s,1)
            
            cri = (omi+1j*omj)/2**0.5 + Sh @ Nih @ (   np.sum(np.random.randn(s,nbaselines),axis=1)+                                                    1j*np.sum(np.random.randn(s,nbaselines),axis=1)   ).reshape((s,1))/2**0.5
            
            bcri = b + cri
            xboth, info2 = sp.sparse.linalg.cg(A, bcri, maxiter=1e5, M=Ai)
            solns[i] = Sh@xboth
    else:
    	# Real-valued version of the calculation
        for i in range(realisations):
            omi = np.random.randn(s,1)
            cri = omi + Sh @ Nih @ np.sum(np.random.randn(s,nbaselines),axis=0)
            bcri = b + cri
            xboth, info2 = sp.sparse.linalg.cg(A, bcri, maxiter=1e5, M=Ai)
            solns[i] = Sh@xboth
    
    # In-paint into flagged regions if requested
    unflagged_indices = np.where(w==1)
    if inpaint_mode == 'inpaint':
        Wnr[unflagged_indices] = dat[unflagged_indices]
        for i,sol in enumerate(solns):
            solns[i][unflagged_indices] = dat[unflagged_indices]
    
    # Subtract the solution from the unflagged data
    if inpaint_mode == 'subtract':
        Z = np.zeros(s, dtype=complex)
        Z[unflagged_indices] = dat[unflagged_indices] - Wnr[unflagged_indices]
        Wnr = Z
        for i,sol in enumerate(solns):
            Z = np.zeros(s, dtype=complex)
            Z[unflagged_indices] = dat[unflagged_indices] - sol[unflagged_indices]
            solns[i] = Z
        
    if poolmap==True:
    	return solns
    else:
    	return Wnr, solns
    

def GCR_OQEarray(V, w, S, N, inpaint='inpaint'):
    
    VW = np.zeros(V.shape, dtype=complex)
    VC = np.zeros(V.shape, dtype=complex)
    
    for i,rzn in enumerate(V):
        if not i%2: 
            id2=i+1 
            wnr, cr = GCR(rzn, w, S, N, realisations=1, inpaint=inpaint, dat2=V[id2])
            VW[i] = wnr
            VC[i] = cr
            if i==0: print('complex: data',np.iscomplexobj(rzn),'C_s',np.iscomplexobj(S),'C_n',np.iscomplexobj(N))

            fi = np.where(w==0)
            VW[i+1] = V[i+1]
            VC[i+1] = V[i+1]
            VW[i+1][fi] = wnr[fi]
            VC[i+1][fi] = cr[:,fi]

        if not i%100: print(i, end=' ')
    return VW, VC


def GCR_array(V, w, S, N, inpaint='inpaint', bla=None, ncpu=2):
    
    """
    bla set to nbaselines (not None) - take noiseless sims and generate 
    nbaselines \times noisy sims to hand to the GCR solver.
    """
    VW = np.zeros(V.shape, dtype=complex)
    VC = np.zeros(V.shape, dtype=complex)
       
    Vidxs = np.arange(V.shape[0])
    
    st = time.time()
    if bla is None:
        with Pool(ncpu) as pool:
            VC = pool.map(lambda idx: GCR(V[idx], w, S, N, 
            							  realisations=1, 
            							  inpaint=inpaint, 
            							  poolmap=True), 
                          Vidxs)
                
    else:
        nbaselines = bla
        for i,rzn in enumerate(V): # assuming now that these V are noiseless, we're going to create redundant baseline data here
            
            noises = 1.0 * np.random.multivariate_normal(mv, C_noise, nbaselines) \
                   + 1.j * np.random.multivariate_normal(mv, C_noise, nbaselines)
            redundant_bls = rzn + noises # broadcasting single V to nbaselines * noise
            
            wnr, cr = GCR(rzn, w, S, N, realisations=1, inpaint=inpaint, bla=redundant_bls)
            VW[i] = wnr
            VC[i] = cr
            if i==0:
            	print('complex: data', np.iscomplexobj(rzn),
            		  'C_s', np.iscomplexobj(S),
            		  'C_n', np.iscomplexobj(N))

            if not i%100:
            	print(i, end=' ')

    print('%.1fs'%(time.time()-st), end=' ')
    return VW, np.array(VC).reshape(V.shape)



def GCR_eigarray(V, w, S, F_evecs, N, ncpu=2):
    
    VC = np.zeros(V.shape, dtype=complex)

    Vidxs = np.arange(V.shape[0])
    
    st=time.time()
    
    with Pool(ncpu) as pool:
        VC = pool.map(lambda idx: GCR_eig(V[idx], w, S, F_evecs, N), Vidxs)

    print('%.1fs'%(time.time()-st), end=' ')
    return np.array(VC).reshape(V.shape)



def wfcorrection(S,N):
    # does this need an fftshift? adding one in the results script
    T = np.zeros((s,s), dtype=complex)

    for i in range(s):
        T[i] = m(i,s)
        
    return np.diag(T.conj().T @ ( S @ np.linalg.inv(S+N) @ N    ) @ T)



def sample_S(s=None, sk=None, prior=None):
    """
    Draw samples of the bandpowers of S, p(S|s). This assumes that the conditional 
    distribution for the bandpowers are uncorrelated with one another, i.e. the Fourier-
    space covariance S has no off-diagonals.
    
    Parameters:
        s (array_like):
            A set of real-space samples of the field, of shape (Ntimes, Nfreq).
    """
    if s is None and sk is None:
        raise ValueError("Must pass in s (real space) or sk (Fourier space) vector.")

    if sk is None:
        sk = np.fft.fft(s, axis=-1) 
    Nobs, Nfreqs = sk.shape
    
    beta = np.sum(sk * sk.conj(), axis=0).real 
    alpha = Nobs/2. - 1.
    
    x = np.zeros(Nfreqs)
    for i in range(Nfreqs):
        x[i] = invgamma.rvs(a=alpha) * beta[i] # y = x / beta
    
    
    if prior is not None:
        for i in range(Nfreqs):
            if prior[0,i] ==0: continue
            else: 
                if x[i] > prior[0,i]:  x[i] = prior[0,i]
                if x[i] < prior[1,i]:  x[i] = prior[1,i]
                    
    return x
 

def sprior(signals, bins, factor):
    
    # prior on cov samples 
    
    # bins - number of bins past zero delay to take, either side. e.g. bins=2 takes delays [-2,-1,0,1,2] from centre
    # factor is maximum factor to multiply / divide the truth by
    Nobs, Nfreq = signals.shape

    sk_ = np.fft.fft(signals, axis=-1)
    ds = np.sum(sk_ * sk_.conj(), axis=0).real
    prior = np.zeros((2,Nfreq))
    
    prior[0] = ds*factor
    prior[1] = ds/factor
    
    prior[0,bins+1:-bins] = 0
    prior[1,bins+1:-bins] = 0
    
    return prior/(Nobs/2 -1)


def GCR_eig(dat, w, matlib, F_evecs, f0=None, amps=None):
    """
    GCR w/ fitted eigenmodes of C_foreground
    
    F_evecs --> matrix of foreground eigenvectors
    f0 --> mean of foregrounds
    matlib --> contains a load of matrices the GCR needs
    
    """
    s = F_evecs.shape[0]
    fvs = F_evecs.shape[1]
        
    d = dat.reshape((1,max(s,len(dat.T))))

    Sh = matlib[0][0]
    Si = matlib[0][1]
    Ni = matlib[0][2]
    Sih = matlib[0][3]
    Nih = matlib[0][4]
    A = matlib[1][0]
    Ai = matlib[1][1]
            
    omi, omj = np.random.randn(s,1),np.random.randn(s,1)
    omk, oml = np.random.randn(s,1),np.random.randn(s,1)
    oma, omb = (omi+1j*omj)/2**0.5 , (omk+1j*oml)/2**0.5

    b = np.zeros((s+fvs,1), dtype=complex)

    b[:s] = Ni @ (w*d).T + Sih@oma + Nih @ omb
    b[s:] = F.T.conj() @ (Ni @ (w*d).T + Nih @ omb) 
    
    if f0 is not None:
        xboth, info2 = sp.sparse.linalg.cg(A, b, maxiter=1e5, x0=np.concatenate((np.zeros(s,dtype=complex),f0)), M=Ai)
    else: 
        xboth, info2 = sp.sparse.linalg.cg(A, b, maxiter=1e5, M=Ai)
        
    sig_sol = xboth[:s]
    fg_sol = xboth[s:] @ F.T            

    sol = sig_sol + fg_sol
       

    if amps==True: return xboth
    else: return sig_sol


def makeS(delayspec):
    
    # transforms the sampled delay spectrum back into freq-freq for the next iter's C_signal
    
    N_freq = delayspec.size
    
    C_sigfft = np.zeros((N_freq, N_freq), dtype=complex)
    
    for i in range(N_freq):
        C_sigfft[i,i] = delayspec[i]
        
    C_sig = (FO.T.conj() @ C_sigfft @ FO).real
    
    return C_sig
    

def GCR_eigarray(V, w, matlib, F_evecs, f0=None, ncpu=2, amps=None):
    
    # performing the GCR step on all LSTs, uses parallelization 
    
    if amps: VC = np.zeros((V.shape[0],V.shape[1]+F_evecs.shape[1]), dtype=complex)
    else: VC = np.zeros(V.shape, dtype=complex)

    Vidxs = np.arange(V.shape[0])
    
    st=time.time()
    with Pool(ncpu) as pool:
        VC = pool.map(lambda idx: GCR_eig(V[idx], w, matlib, F_evecs, f0=f0, amps=amps), Vidxs)

    print('%.1fs'%(time.time()-st), end=' ')
    if amps: return np.array(VC).reshape((V.shape[0],-1))
    else: return np.array(VC).reshape(V.shape)


# the single iteration / step function

def Gibbs_eig_step(vis, S, F_evecs,  N, flags, f0=None, amps=None, prior=None, ncpu=2):
    
    Nvis,Nfreq = vis.shape
    
    fvs = F_evecs.shape[1]
    A_l = int(s + fvs) 
    
    matlib = [0,0]
    matlib[0] = np.zeros((5,Nfreq,Nfreq),dtype=complex)
    matlib[1] = np.zeros((2,A_l,A_l),dtype=complex)

    matlib[0][0] = sp.linalg.sqrtm( S ) # Sh
    matlib[0][1] = np.linalg.inv( S )   # Si
    matlib[0][2] = flags.T*np.linalg.inv( N )*flags # Ni
    matlib[0][3] = sp.linalg.sqrtm( matlib[0][1] )  # Sih
    matlib[0][4] = sp.linalg.sqrtm( matlib[0][2] )  # Nih
    
    A = np.zeros((A_l, A_l), dtype=complex)
    A[:s,:s] =  matlib[0][1] + matlib[0][2]  # Si + Ni
    A[:s,s:] = matlib[0][2] @ F_evecs     
    A[s:,:s] =  F_evecs.T.conj() @ matlib[0][2]  
    A[s:,s:] = F_evecs.T.conj() @ matlib[0][2] @ F_evecs

    matlib[1][0] = A
    matlib[1][1] = np.linalg.inv(A)
    
    cr = GCR_eigarray(vis, flags, matlib, F_evecs, f0=f0, ncpu=ncpu, amps=amps)
    if amps: 
        amplitudes = cr[:,-F_evecs.shape[1]:]
        cr = cr[:,:-F_evecs.shape[1]]
    
    ds_sample = sample_S(s=cr, prior=prior)
    
    Snew = makeS(ds_sample/(2*Nfreq**2)) # divide by factor of 2 * N_freq^2
    
    if amps: return cr, Snew, ds_sample, amplitudes
    else: return cr, Snew
    
