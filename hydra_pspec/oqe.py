
import numpy as np

#from multiprocess import Pool
from multiprocessing import Pool

def m(tau, s):
    y = np.zeros(s)
    y[tau] = 1
    return np.fft.fft(y)

    
def Q(tau, s):
    filename = 'Qs/Q' + str(s) + '_' + str(tau)+ '.npy'
    if os.path.isfile(filename):
        Q = np.load(filename)
    else:
        Q = np.outer( m(tau,s).conj(), m(tau,s) )
        np.save(filename, Q)
    return Q


def bias(tau, s, R, C_noise_total):
    return 0.5 * np.trace( C_noise_total @ R.conj() @ Q(tau,s) @ R )


def qhat(x, tau, s, R, bias):
	# redundant in HERA setup, but useful for comparison
    E = R.conj() @ Q(tau, s) @ R
    return 0.5 *  ( x.conj().T @ E @ x ) - bias


def qhat_h(x1, x2, tau, s, R):
    # HERA-like cross corr    
    # exact Pspec code:
	# Rx1, Rx2 = np.dot(R, x1) , np.dot(R, x2)
	# QRx2 = np.dot(Q(tau,s), Rx2)
	# return 0.5 * np.einsum('i...,i...->...', Rx1.conj(), QRx2) # can test these vs each other....
    Rx1, Rx2 = R @ x1, R @ x2
    return 0.5 * Rx1.conj().T @ Q(tau,s) @ Rx2  


def F(s, R):
    
    t = np.arange(s)
    F = np.zeros((s,s), dtype=complex)
    for a in range(s):
        for b in range(s):
            F[a,b] = 0.5 * np.trace(  R.conj() @ Q(t[a], s) @ R @ Q(t[b], s)  )
    return F


def Ft(s, R):
    t = np.arange(s)
    F = np.zeros((s,s), dtype=complex)
    
    iR1Q1, iR2Q2 = {}, {}
    
    for i in range(s):
        iR1Q1[i] = np.dot(np.conj(R).T, Q(i,s)) # R_1 Q_alt
        iR2Q2[i] = np.dot(R, Q(i,s)) # R_2 Q
    
    for a in range(s):
        for b in range(s):
            F[a,b] = 0.5*np.einsum('ab,ba', iR1Q1[a], iR2Q2[b])    
    return F


def M_Fhalf(F):
    return np.linalg.inv(sp.linalg.sqrtm(F))


def M_Finv(F):
    return np.linalg.inv(F)


def M_opt(F):
    
    M = np.diag(np.divide(1, np.diag(F)))
    W = M @ F
    for row in range(0, np.shape(M)[0]):
        M[row] = np.divide(M[row], np.sum(W[row])) # Wll normalisation - does it make sense? perhaps not
    
    return M 



def q(V, s, R, bias):
    """
    Calculates qhat across the tau range
    Need to calculate bias beforehand - needs to be an array (see return statement below)
    """
    
    N = len(V)
    t_ = np.arange(s)
    qs = np.zeros((N,s))
    
    for i in range(N):
        qs[i] = np.array([qhat(V[i], tau, s, R, bias[tau]) for tau in t_])
    
    return qs


def q_h(V, s, R, taper=None):
    
    N = len(V) // 2 # Should be even if creating pairs of visibilities
    
    t_ = np.arange(s)
    qs = np.zeros((N,s), dtype=complex)

    for i in range(N):
        qs[i] = np.array([qhat_h(V[2*i], V[2*i+1], t, s, R) for t in t_])
    
    return qs


def p(q, M):
    return M @ q 


def matc(M):
    evs = np.linalg.eigvals(M).real
    Minv = np.linalg.inv(M)
    print(np.all(evs > 0),' - positive definite')
    print(np.format_float_scientific( max(evs)/min(evs)  ),' - eigval ratio')
    print('%f'%(np.linalg.norm(M)*np.linalg.norm(Minv)),' - condition (norm C x norm Cinv)')
    print('')


def getqs(Vis, R):
    """
    Creates required matrices and runs the skeleton OQE over the given set of 
    data using weighting R, returning unnormalized qs to be normalized by the 
    pstats function.
    """
    st = time.time()
    s = len(Vis[0])
    matc(R)
    Fm = F(s, R) # Fisher matrix
    MB = M_opt(Fm)
    MA = M_Finv(Fm)
    qs = q_h(Vis, s, R)
    print('%.3fs'%(time.time()-st))
    return qs, Fm, MB, MA


def q_hp(V, s, R, ncpu):
    st=time.time()
    N = len(V)//2 
    t_ = np.arange(s)
    if np.iscomplexobj(V):  qs = np.zeros((N,s), dtype=complex)
    else: qs = np.zeros((N,s))
        
    Vidxs = np.arange(N)
    with Pool(ncpu) as pool:
        qs = pool.map(lambda idx: np.array([qhat_h(V[2*idx], V[2*idx+1], t, s, R) for t in t_]), Vidxs)
    print('%.3fs'%(time.time()-st))
    return qs


def Sig_QEN(R, C_noise, norm):
    
    # In jianrong's paper, E is normalized. So, need to divide by the sum of 
    # the (row) of the window function
    s = len(R)
    Sig = np.zeros(s, dtype=complex)
    
    for i in range(s):
        E = R @ Q(i, s) @ R * norm

        Sig[i] = 0.5 * np.trace( E @ C_noise @ E @ C_noise )
    
    return Sig



def Sig_QESN(R, C_noise, C_S, norm):
    
    s = len(R)
    Sig = np.zeros(s, dtype=complex)
    
    for i in range(s):
        E = R @ Q(i, s) @ R * norm
        Sig[i] = 0.5 * np.trace( (E @ C_noise @ E @ C_noise) + (E @ C_S @ E @ C_noise) + (E @ C_noise @ E @ C_S))
        
    return Sig