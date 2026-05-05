import matplotlib.pyplot as plt
import numpy as np
# from config import *
import os
import scipy.stats as sci_st
from pyuvdata import UVData
from astropy import units
from pathlib import Path
from matplotlib.colors import LogNorm, Normalize
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from uvtools.plot import waterfall

# if hera_sim.__version__.startswith('0'):
#     from hera_sim.rfi import _listify
# else:
#     from hera_sim.utils import _listify
op_dir='paper_plots/'
    
def plot_inputs(eor_vis, fg_vis, vis, vis_sys, bsys_test, y_test):
    
    '''Check if input plot directory exists'''
    input_plot_dir = output_dir_path+'input_plots/'
    if os.path.isdir(input_plot_dir) == False:
        os.makedirs(input_plot_dir)

    '''Visibility plots'''
    fig,ax=plt.subplots(2,4,figsize=(56,14))

    im=ax[0,0].imshow(eor_vis.real,aspect='auto')
    ax[0,0].set_title("EoR visibilities")
    ax[0,0].set_ylabel("Real Part")
    plt.colorbar(im)

    im=ax[0,1].imshow(fg_vis.real,aspect='auto')
    ax[0,1].set_title("FG Visibilities")
    plt.colorbar(im)

    im=ax[0,2].imshow(vis.real,aspect='auto')
    ax[0,2].set_title("Clean sky Visibilities")
    plt.colorbar(im)  
    
    im=ax[0,3].imshow(vis_sys.real,aspect='auto')
    ax[0,3].set_title("Visibilities with systematics")
    plt.colorbar(im)
    
    im=ax[1,0].imshow(eor_vis.imag,aspect='auto')
    ax[1,0].set_ylabel("Imaginary Part")
    plt.colorbar(im)

    im=ax[1,1].imshow(fg_vis.imag,aspect='auto')
    plt.colorbar(im)

    im=ax[1,2].imshow(vis.imag,aspect='auto')
    plt.colorbar(im)  
    
    im=ax[1,3].imshow(vis_sys.imag,aspect='auto')
    plt.colorbar(im)
    
    fig.suptitle("Input visibilities",ha='center',va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    plt.savefig(input_plot_dir+'Input_visibility_plots.png',bbox_inches='tight',dpi=300)
    
    '''Systematics plots'''
    fig,ax=plt.subplots(1,4,figsize=(21,7))
    
    ax[0].plot(bsys_test.real,marker='.')
    ax[0].set_title("Real part")

    ax[1].plot(bsys_test.imag,marker='.')
    ax[1].set_title("imaginary part")

    im=ax[2].matshow(y_test.real, aspect='auto')
    ax[2].set_title("Real part")
    plt.colorbar(im)
    
    im=ax[3].matshow(y_test.imag, aspect='auto')
    ax[3].set_title("Imaginary part")
    plt.colorbar(im)

    fig.suptitle("The systematics vector",ha='center',va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    plt.savefig(input_plot_dir+'Input_systematics_plots.png',bbox_inches='tight',dpi=300)


def plot_matrices(h_j, nm_list, signal_S, fgmodes, N_cov, Ninv):
    '''Check if input plot directory exists'''
    input_plot_dir = output_dir_path+'input_plots/'
    if os.path.isdir(input_plot_dir) == False:
        os.makedirs(input_plot_dir)

    '''H operator'''
    fig,ax=plt.subplots(1,2,figsize=(20,10))
    nm_ticks = [str(nm) for nm in nm_list]
    tick_locs = np.linspace(0,len(nm_list)-1,len(nm_list))
    
    im0=ax[0].matshow(h_j.real,origin='lower',aspect='auto')
    ax[0].set_title('Real part')
    ax[0].set_xlabel('Modes')
    ax[0].set_xticks(tick_locs, labels=nm_ticks)
    plt.colorbar(im0)

    im1=ax[1].matshow(h_j.imag,origin='lower',aspect='auto')
    ax[1].set_title('Imaginary part')
    ax[1].set_xlabel('Modes')
    ax[1].set_xticks(tick_locs, labels=nm_ticks)
    plt.colorbar(im1)

    fig.suptitle("H operator",ha='center',va='bottom',fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    plt.savefig(input_plot_dir+'H_operator.png',bbox_inches='tight',dpi=300)

    '''Matrices from file'''
    fig, ax= plt.subplots(2, 4, figsize=(45,18))

    im0=ax[0,0].imshow(signal_S.real, origin='lower',cmap='PuRd')
    ax[0,0].set_title("EoR cov",fontsize=20)
    ax[0,0].set_ylabel("Real part",fontsize=20)
    plt.colorbar(im0)

    im1=ax[0,1].imshow(fgmodes.real, origin='lower',cmap='PuRd')
    ax[0,1].set_title("Foreground covariance",fontsize=20)
    plt.colorbar(im1)

    im2=ax[0,2].imshow(N_cov.real, origin='lower',cmap='PuRd')
    ax[0,2].set_title("Noise covariance",fontsize=20)
    plt.colorbar(im2)

    im3=ax[0,3].imshow(Ninv.real,origin='lower',cmap='PuRd')
    ax[0,3].set_title("Noise cov inverse",fontsize=20)
    plt.colorbar(im3)

    im0=ax[1,0].imshow(signal_S.imag, origin='lower',cmap='PuRd')
    ax[1,0].set_ylabel("Imaginary part",fontsize=20)
    plt.colorbar(im0)

    im1=ax[1,1].imshow(fgmodes.imag, origin='lower',cmap='PuRd')
    plt.colorbar(im1)

    im2=ax[1,2].imshow(N_cov.imag, origin='lower',cmap='PuRd')
    plt.colorbar(im2)

    im3=ax[1,3].imshow(Ninv.imag,origin='lower',cmap='PuRd')
    plt.colorbar(im3)

    fig.suptitle("Plotting all the matrices from file",ha='center',va='bottom',fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    
    plt.savefig(input_plot_dir+'matrices_from_file.png',bbox_inches='tight',dpi=300)

def plot_results(vis, vis_sys, model, y_test, sys_model, eor_vis, signal_cr, fg_sol, fg_test, bsys_test, b_sys):
    '''Check if results plot directory exists'''
    result_plot_dir = output_dir_path+'result_plots/'
    if os.path.isdir(result_plot_dir) == False:
        os.makedirs(result_plot_dir)
        
    '''Comparison of visibilities'''
    fig, ax= plt.subplots(2,3,figsize=(27,15))

    im0=ax[0,0].imshow((vis_sys).real,origin='lower',cmap='PuRd')
    ax[0,0].set_title('Visibilities (simulated)')
    ax[0,0].set_ylabel('Real Part')
    plt.colorbar(im0)

    im1=ax[0,1].imshow(model.real,origin='lower',cmap='PuRd')
    ax[0,1].set_title('Visibilities (solved)')
    plt.colorbar(im1)

    im2=ax[0,2].imshow((model - (vis_sys)).real,origin='lower',cmap='PuRd')
    ax[0,2].set_title('Residuals')
    plt.colorbar(im2)

    im0=ax[1,0].imshow((vis_sys).imag,origin='lower',cmap='PuRd')
    ax[1,0].set_ylabel('Imaginary Part')
    plt.colorbar(im0)

    im1=ax[1,1].imshow(model.imag,origin='lower',cmap='PuRd')
    plt.colorbar(im1)

    im2=ax[1,2].imshow((model - (vis_sys)).imag,origin='lower',cmap='PuRd')
    plt.colorbar(im2)

    fig.suptitle("Comparison of test and solved visibilities",ha='center',va='bottom',fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    plt.savefig(result_plot_dir+'visibilities_test_vs_solved.png',bbox_inches='tight',dpi=300)

    '''Inspecting systematics solutions'''
    fig, ax= plt.subplots(2,3,figsize=(27,15))

    im3=ax[0,0].imshow(y_test.real,origin='lower',cmap='PuRd')
    ax[0,0].set_title('Systematics (simulated)')
    ax[0,0].set_ylabel('Real Part')
    plt.colorbar(im3)

    im4=ax[0,1].imshow(sys_model.real,origin='lower',cmap='PuRd')
    ax[0,1].set_title('Systematics (solved)')
    plt.colorbar(im4)

    im5=ax[0,2].imshow((sys_model-y_test).real,origin='lower',cmap='PuRd')
    ax[0,2].set_title('Residuals')
    plt.colorbar(im5)
    
    im3=ax[1,0].imshow(y_test.imag,origin='lower',cmap='PuRd')
    ax[1,0].set_ylabel('Imaginary Part')
    plt.colorbar(im3)

    im4=ax[1,1].imshow(sys_model.imag,origin='lower',cmap='PuRd')
    plt.colorbar(im4)

    im5=ax[1,2].imshow((sys_model-y_test).imag,origin='lower',cmap='PuRd')
    plt.colorbar(im5)
    
    fig.suptitle("Comparison of systematics data and solution",ha='center',va='bottom',fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)

    plt.savefig(result_plot_dir+'systematics_test_vs_solved.png',bbox_inches='tight',dpi=300)

    '''Comparing EoR test vs solutions'''
    fig,ax=plt.subplots(2,3,figsize=(28,14))

    im0=ax[0,0].imshow(eor_vis.real)
    plt.colorbar(im0)
    ax[0,0].set_ylabel("Real part")
    ax[0,0].set_title("EoR test")

    im1=ax[0,1].imshow(signal_cr.real)
    plt.colorbar(im1)
    ax[0,1].set_title("EoR solved")

    im1=ax[0,2].imshow(signal_cr.real-eor_vis.real)
    plt.colorbar(im1)
    ax[0,2].set_title("Residuals")

    im2=ax[1,0].imshow(eor_vis.imag)
    plt.colorbar(im2)
    ax[1,0].set_ylabel("Imaginary part")

    im3=ax[1,1].imshow(signal_cr.imag)
    plt.colorbar(im3)

    im3=ax[1,2].imshow((signal_cr.imag)-eor_vis.imag)
    plt.colorbar(im3)
    
    fig.suptitle("Comparison of EoR test and solution",ha='center',va='bottom',fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    plt.savefig(result_plot_dir+'eor_test_vs_solved.png',bbox_inches='tight',dpi=300)

    '''Foreground test vs solution'''
    fig,ax=plt.subplots(2,3,figsize=(28,14))

    im0=ax[0,0].matshow(fg_test.real)
    plt.colorbar(im0)
    ax[0,0].set_ylabel("Real part")
    ax[0,0].set_title("FG test")

    im1=ax[0,1].matshow(fg_sol.real)
    plt.colorbar(im1)
    ax[0,1].set_title("FG solved")

    im1=ax[0,2].matshow(fg_test.real-fg_sol.real)
    plt.colorbar(im1)
    ax[0,2].set_title("Residuals")

    im2=ax[1,0].matshow(fg_test.imag)
    plt.colorbar(im2)
    ax[1,0].set_ylabel("Imaginary part")

    im3=ax[1,1].matshow(fg_sol.imag)
    plt.colorbar(im3)

    im3=ax[1,2].matshow(fg_test.imag-fg_sol.imag)
    plt.colorbar(im3)

    fig.suptitle("Comparison of FG test and solution",ha='center',va='bottom',fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    plt.savefig(result_plot_dir+'FG_test_vs_solved.png',bbox_inches='tight',dpi=300)

    '''Comparison of sky models'''
    
    sky_sol = (signal_cr + fg_sol)

    fig,ax=plt.subplots(2,3,figsize=(28,14))

    im0=ax[0,0].imshow(vis.real)
    plt.colorbar(im0)
    ax[0,0].set_ylabel("Real part")
    ax[0,0].set_title("Sky model test")

    im1=ax[0,1].imshow(sky_sol.real)
    plt.colorbar(im1)
    ax[0,1].set_title("Solved sky model")

    im1=ax[0,2].imshow(vis.real-sky_sol.real)
    plt.colorbar(im1)
    ax[0,2].set_title("Residuals")

    im2=ax[1,0].imshow(vis.imag)
    plt.colorbar(im2)
    ax[1,0].set_ylabel("Imaginary part")

    im3=ax[1,1].imshow(sky_sol.imag)
    plt.colorbar(im3)

    im3=ax[1,2].imshow(vis.imag-sky_sol.imag)
    plt.colorbar(im3)
    
    fig.suptitle("Comparison of sky model test and solution",ha='center',va='bottom',fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    plt.savefig(result_plot_dir+'sky_model_test_vs_solved.png',bbox_inches='tight',dpi=300)

    '''systematics vector test vs solution'''
    fig,ax=plt.subplots(1,2,figsize=(14,7))

    x=np.arange(0, len(bsys_test),1)

    ax[0].plot(x, bsys_test.real,'rx',label='x_true')
    ax[0].plot(x, b_sys.real,'b.',label='x_solution')
    ax[0].legend()
    ax[0].set_title("Real part")
    ax[0].set_xlabel("Indices")
    ax[0].set_ylabel("Values")

    ax[1].plot(x, bsys_test.imag,'rx',label='x_true')
    ax[1].plot(x, b_sys.imag,'b.',label='x_solution')
    ax[1].legend()
    ax[1].set_title("Imaginary part")
    ax[1].set_xlabel("Indices")
    ax[1].set_ylabel("Values")

    fig.suptitle("Comparison of systematics vector: solved and true",ha='center',va='bottom')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    
    plt.savefig(result_plot_dir+'sys_vector_test_vs_solved.png',bbox_inches='tight',dpi=300)

def master_plotter(
    data_sets,
    col_labels=None,
    fig_title='Data comparison',
    plot_type='imshow',
    norm='linear',
    save_flag=True,
    cmap='seismic',
    dir=op_dir,
    imag_flag=True,
    vmin=None,
    vmax=None,
    show=False
):
    """
    Plot a list of 2D complex data sets in a grid of subplots, displaying real, imaginary,
    and absolute components (if `imag_flag` is True).

    Parameters
    ----------
    data_sets : list of 2D np.ndarray
        List of 2D complex-valued arrays to plot. Each entry will be shown in a column of subplots.

    col_labels : list of str, optional
        Labels for each data set to be shown above the corresponding column. If not provided, generic labels will be used.

    fig_title : str, optional
        Title for the entire figure. Also used as the filename if saving.

    plot_type : {'imshow', 'matshow'}, optional
        Type of matplotlib plot to use for each subplot. Default is 'imshow'.

    norm : {'linear', 'log'} or matplotlib.colors.Normalize, optional
        Color normalization to apply. Can be 'linear', 'log', or a custom Normalize object.

    save_flag : bool, optional
        Whether to save the resulting figure to file.

    cmap : str or matplotlib colormap, optional
        Colormap to use for the plots.

    dir : str, optional
        Directory to save the figure in, if `save_flag` is True. Default is `op_dir` specified in config_plots.

    imag_flag : bool, optional
        If True, plots all three of real, imaginary, and absolute parts. If False, only plots the real part.

    vmin : float or list of floats, optional
        Minimum value(s) for colormap normalization. Can be a scalar or list matching the number of data sets.

    vmax : float or list of floats, optional
        Maximum value(s) for colormap normalization. Can be a scalar or list matching the number of data sets.

    show : bool, optional
        If True, displays the figure using `plt.show()`. If False, closes the figure after saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plotted subplots.

    Raises
    ------
    ValueError
        If the number of column labels does not match the number of data sets, or if an unsupported `plot_type` or `norm` is provided.

    Notes
    -----
    - This function is useful for visually comparing multiple 2D complex-valued matrices in terms of their real, imaginary, and magnitude components.
    """

    num_sets = len(data_sets)
    # data_sets = np.array(data_sets)
    col_labels = col_labels or [f"Data {i}" for i in range(num_sets)]

    if len(col_labels) != num_sets:
        raise ValueError("Number of column labels must match number of data sets.")

    if isinstance(norm, str):
        if norm == 'linear':
            norm_fn = None
        elif norm == 'log':
            norm_fn = LogNorm()
        else:
            raise ValueError(f"Unknown norm '{norm}'")
    else:
        norm_fn = norm

    nrows = 3 if imag_flag else 1
    fig, ax = plt.subplots(nrows, num_sets, figsize=(num_sets * 5, nrows * 6), squeeze=False)
    ylabels = ['Real', 'Imaginary', 'Absolute']

    for i in range(num_sets):
        data = data_sets[i]
        vmin_i = vmin[i] if isinstance(vmin, (list, tuple, np.ndarray)) else vmin
        vmax_i = vmax[i] if isinstance(vmax, (list, tuple, np.ndarray)) else vmax
        for j, part in enumerate([np.real(data), np.imag(data), np.abs(data)] if imag_flag else [np.real(data)]):
            plot_ax = ax[j, i]
            if plot_type == 'imshow':
                im = plot_ax.imshow(part, origin='lower', cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i)
            elif plot_type == 'matshow':
                im = plot_ax.matshow(part, cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i, aspect='auto')
            else:
                raise ValueError("plot_type must be 'imshow' or 'matshow'")
            if i == 0:
                plot_ax.set_ylabel(ylabels[j], fontsize=16)
            plot_ax.set_title(col_labels[i], fontsize=16)
            cbar = plt.colorbar(im, ax=plot_ax,  fraction=0.046, pad=0.04)
            cbar.set_label(label='Amplitudes',fontsize=16)
            cbar.ax.tick_params(which='both')

    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_flag:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir+fig_title+'.png',bbox_inches='tight',dpi=300, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_dps(vis_eor_path, res_dir, dir=op_dir, Nburn=0, conf_interval=95, ):
    # Load in EoR visibilities
    uvd = UVData()
    uvd.read(vis_eor_path)
    uvd.conjugate_bls()
    # uvd = form_pseudo_stokes_vis(uvd)
    # The test data only contains a single baseline (0, 1) and the pseudo-Stokes I
    # visibilities after `form_pseudo_stokes_vis` are stored in the XX polarization
    vis_eor = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)

    # Get freuqency metadata
    freqs = uvd.freq_array * units.Hz
    if uvd.use_future_array_shapes:
        freqs = freqs[0]
    df = freqs[1] - freqs[0]
    Nfreqs = freqs.size

    # Compute the delay power spectrum of the input EoR signal
    axes = (1,)
    ds_eor_true = np.fft.ifftshift(vis_eor, axes=axes)
    ds_eor_true = np.fft.fftn(ds_eor_true, axes=axes)
    ds_eor_true = np.fft.fftshift(ds_eor_true, axes=axes)
    dps_eor_true = (np.abs(ds_eor_true)**2).mean(axis=0)
    delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

    # Load in results from hydra_pspec
    dps_eor_hp = np.load(Path(res_dir) / "dps-eor.npy")
    ln_post = np.load(Path(res_dir) / "ln-post.npy")
    if Nburn > 0:
        dps_eor_hp = dps_eor_hp[Nburn:]
        ln_post = ln_post[Nburn:]
    # Posterior-weighted mean delay power spectrum
    dps_eor_hp_pwm = np.average(dps_eor_hp, weights=ln_post, axis=0)
    # Confidence interval of delay power spectrum posteriors
    percentile = conf_interval/2 + 50
    dps_eor_hp_ubound = np.percentile(dps_eor_hp, percentile, axis=0)
    dps_eor_hp_lbound = np.percentile(dps_eor_hp, 100-percentile, axis=0)
    dps_eor_hp_err = np.vstack((
        dps_eor_hp_pwm - dps_eor_hp_lbound,
        dps_eor_hp_ubound - dps_eor_hp_pwm
    ))

    # Plot the true and recovered delay power spectra
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(delays, dps_eor_true, "k:", label="True")
    ax.errorbar(
        delays,
        dps_eor_hp_pwm,
        yerr=np.abs(dps_eor_hp_err),
        color="k",
        # ls="",
        marker="o",
        capsize=3,
        label=f"Recovered ({conf_interval}% Confidence)"
    )
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"$P(\tau)$ [arb. units]")
    ax.set_title("EoR Delay Power Spectrum Comparison (systematics)")
    ax.set_yscale("log")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS_comparison.png',bbox_inches='tight',dpi=300)

    res=dps_eor_hp_pwm - dps_eor_true
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays,res, yerr=0.68*np.abs(dps_eor_hp_err),marker="o",
        capsize=3,)
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Data - true dps")
    ax.set_title("Residuals vs delays")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS_res_vs_delays.png',bbox_inches='tight',dpi=300)


    z_sc=sci_st.zscore(res)
    sig=np.std(dps_eor_hp_err)
    yerr_z=dps_eor_hp_err/sig

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays,z_sc, yerr=np.abs(yerr_z),marker="o",markerfacecolor='blue',
        capsize=3,ecolor='blue')
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Z score")
    ax.set_title("Z score vs delays")
    ax.set_ylim(-5,5)
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir+'EoR_DPS__Score_vs_delays.png',bbox_inches='tight',dpi=300)

    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(delays,dps_eor_hp_err[0,:],marker="o",label='Lower limit',ls='dotted')
    ax.plot(delays,dps_eor_hp_err[1,:],marker="o",label='Upper limit',ls='dotted')
    ax.plot(delays,np.mean(dps_eor_hp_err, axis=0),marker="o",label='Mean',c='k')
    ax.set_title("Error bar means and upper-lower limits")
    ax.grid()
    ax.legend()
    plt.savefig(dir+'EoR_DPS_Error_bar_mins_limits.png',bbox_inches='tight',dpi=300,transparent=True)


def plot_waterfalls(data, freqs, times, windows=None, mode='log', fig=None,ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,
                    freq_window_kwargs=None, time_window_kwargs=None):
    """
    Make a 2x2 grid of waterfall plots.
    
    This function takes a 2D array of visibility data (in units of Jy), as well 
    as the corresponding frequency and time arrays (in units of Hz and JD, respectively), 
    and makes a 2x2 grid of plots where each plot shows each one of the possible choices 
    for Fourier transforming along an axis. The upper-left plot is in the frequency-time 
    domain; the upper-right plot is in the frequency-fringe-rate domain; the lower-left 
    plot is in the delay-time domain; and the lower-right plot is in the delay-fringe-rate 
    domain.
    
    Parameters
    ----------
    data : ndarray, shape=(NTIMES,NFREQS)
        Array containing the visibility to be plotted. Assumed to be in units of Jy. 
        
    freqs : ndarray, shape=(NFREQS,)
        Array containing the observed frequencies. Assumed to be in units of Hz.
        
    times : ndarray, shape=(NTIMES,)
        Array containing the observed times. Assumed to be in units of JD.
        
    windows : tuple of str or str, optional
        Choice of taper to use for the fringe-rate and delay transforms. Must be 
        either tuple, list, or string. If a tuple or list, then it must be either 
        length 1 or length 2; if it is length 2, then the zeroth entry is the taper 
        to be applied along the time axis for the fringe-rate transform, with the 
        other entry specifying the taper to be applied along the frequency axis 
        for the delay transform. Each entry is passed to uvtools.dspec.gen_window. 
        If ``windows`` is a length 1 tuple/list or a string, then it is assumed 
        that the same taper is to be used for both axes. Default is to use no 
        taper (or, equivalently, a boxcar).
        
    mode : str, optional
        Plotting mode to use; passed directly to uvtools.plot.waterfall. Default is 
        'log', which plots the base-10 logarithm of the absolute value of the data. 
        
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for visualizing the data. Default is to use the inferno 
        colormap.
        
    dynamic_range : float, optional
        Number of orders of magnitude to use for limiting the dynamic range of the 
        colormap. This parameter is only used if ``mode`` is set to 'log' and the 
        ``limit_drng`` parameter is not None. If the conditions to use this 
        parameter are met, then the vmin parameter is set to be dynamic_range orders 
        of magnitude less than vmax. That is, if vmax = np.log10(np.abs(data)).max(), 
        then vmin = vmax - dynamic_range. Default is to not limit the dynamic range.
        
    limit_drng : str or array-like of str, optional
        Choice of which plots for which to limit the dynamic range. Possible choices 
        are 'freq', 'time', 'delay', and 'fringe_rate'. If any of these are chosen, 
        then the plots that have one of the axes match the specified choices will 
        have their dynamic range limited. For example, passing 'delay' to this 
        parameter will limit the dynamic range for the delay-time and delay-fringe-rate 
        plots. Default is to limit the dynamic range for all plots. 
        
    baseline : float or array-like of float, optional
        Baseline length or baseline position in units of meters. If this parameter is 
        specified, then the geometric horizon is plotted as a vertical line in the 
        delay-space plots. Default is to not plot the geometric horizon.
        
    horizon_color : str, 3-tuple, or 4-tuple, optional
        Color to use for the vertical lines indicating the geometric horizon. This 
        may either be a string, 3-tuple specifying RGB values, or 4-tuple specifying 
        RGBA values. Default is to use magenta.
        
    plot_limits : dict, optional
        Dictionary whose keys may be any of ('freq', 'time', 'delay', 'fringe-rate') 
        and whose values are length 2 array-like objects specifying the bounds for 
        the corresponding axis. For horizontal axes, these should be ordered from low 
        to high; for vertical axes, these should be ordered from high to low. For 
        example, passing {'delay' : (-500, 500)} will limit the delay axis to values 
        between -500 and +500 nanoseconds. Frequency units should be in Hz; time 
        units should be in JD; delay units should be in ns; fringe rate units should 
        be in mHz. Default is to use the full extent of each axis.
        
    freq_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        frequency taper. Default is to pass no keyword arguments.
        
    time_window_kwargs : dict, optional
        Keyword arguments to pass to uvtools.dspec.gen_window for generating the 
        time taper. Default is to pass no keyword arguments.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib.figure.Figure object containing the plots.
    """
    # do some data prep
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}
    time_window = gen_window(windows, times.size, **time_window_kwargs)
    freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)
        
    time_window = time_window[:, None]
    freq_window = freq_window[None, :]
    data_fr = FFT(data * time_window, axis=0)
    data_dly = FFT(data * freq_window, axis=1)
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)
    
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3 # mHz
    dlys = fourier_freqs(freqs) * 1e9 # ns
    plot_freqs = freqs / 1e6
    jd = int(np.floor(times[0]))
    plot_times = times - jd
    
    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9
    
    if ax==None:
        fig = plt.figure(figsize=(10,10),facecolor='white')
        ax = fig.subplots(1,1)
    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black',labelsize=15)
            #    grid_color='r', grid_alpha=0.5)
    # for j, ax in enumerate(axes.ravel()):
    j=1
    column = j % 2
    row = j // 2
    if xlabel==None:
        xlabel = "Delay [ns]"
    ylabel = "Time Since JD%d [days]" % jd if column == 0 else "Fringe Rate [mHz]"
    ax.set_xlabel(xlabel, fontsize=16,color='black')
    ax.set_ylabel(ylabel, fontsize=16,color='black')
    
    xlimits, ylimits = None, None
        # if column == 0 and row == 0:
        #     use_data = data
        #     extent = (
        #         plot_freqs.min(), plot_freqs.max(), plot_times.max(), plot_times.min()
        #     )
        #     vis_label = r"$\log_{10}|V(\nu, t)|$ [Jy]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("freq", extent[:2])
        #         ylimits = plot_limits.get("time", extent[2:])
        # elif column == 0 and row == 1:
        #     use_data = data_dly
        #     extent = (dlys.min(), dlys.max(), plot_times.max(), plot_times.min())
        #     vis_label = r"$\log_{10}|\tilde{V}(\tau, t)|$ [Jy Hz]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("delay", extent[:2])
        #         ylimits = plot_limits.get("time", extent[2:])
        # elif column == 1 and row == 0:
        #     use_data = data_fr
        #     extent = (
        #         plot_freqs.min(), plot_freqs.max(), fringe_rates.max(), fringe_rates.min()
        #     )
        #     vis_label = r"$\log_{10}|\tilde{V}(\nu, f)|$ [Jy s]"
        #     if plot_limits is not None:
        #         xlimits = plot_limits.get("freq", extent[:2])
        #         ylimits = plot_limits.get("fringe_rate", extent[2:])
        # else:
    use_data = data_fr_dly
    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", extent[:2])
        ylimits = plot_limits.get("fringe_rate", extent[2:])
            
    xlimits = xlimits or extent[:2]
    ylimits = ylimits or extent[2:]
    
    if dynamic_range is not None and mode == 'log':
        vmax = np.log10(np.abs(use_data)).max()
        vmin = vmax - dynamic_range
    else:
        vmin, vmax = None, None
        
    clip_drng = False
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    if "time" in limit_drng:
        if column == 0:
            clip_drng = True
    if "freq" in limit_drng:
        if row == 0:
            clip_drng = True
    if "delay" in limit_drng:
        if row == 1:
            clip_drng = True
    if "fringe_rate" in limit_drng:
        if column == 1:
            clip_drng = True
            
    if not clip_drng:
        vmin, vmax = None, None
        
    cbar_label = vis_label if mode == 'log' else "Phase [rad]"
    fig.sca(ax)
    cax = waterfall(
        use_data, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if baseline is not None and row == 1:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')
    
    if colorbar_flag==True:
        _ = plt.colorbar(cax)
        _.set_label(cbar_label,c='black',fontsize=16)
        _.ax.tick_params(axis='y',which='both', color='black', labelcolor='black',labelsize=15)
    return cax,data_fr_dly

def plot_waterfalls_from_dlfr(data_dlfr, freqs, times,mode='log', fig=None,ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True):
    
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3 # mHz
    dlys = fourier_freqs(freqs) * 1e9 # ns
    jd = int(np.floor(times[0]))

    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9
    
    if ax==None:
        fig = plt.figure(figsize=(10,10),facecolor='white')
        ax = fig.subplots(1,1)
    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black',labelsize=15)
            #    grid_color='r', grid_alpha=0.5)
    # for j, ax in enumerate(axes.ravel()):
    j=1
    column = j % 2
    row = j // 2
    if xlabel==None:
        xlabel = "Delay [ns]"
    ylabel = "Time Since JD%d [days]" % jd if column == 0 else "Fringe Rate [mHz]"
    ax.set_xlabel(xlabel, fontsize=16,color='black')
    ax.set_ylabel(ylabel, fontsize=16,color='black')
    
    xlimits, ylimits = None, None
    use_data = data_dlfr
    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", extent[:2])
        ylimits = plot_limits.get("fringe_rate", extent[2:])
            
    xlimits = xlimits or extent[:2]
    ylimits = ylimits or extent[2:]
    
    if dynamic_range is not None and mode == 'log':
        vmax = np.log10(np.abs(use_data)).max()
        vmin = vmax - dynamic_range
    else:
        vmin, vmax = None, None
        
    clip_drng = False
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    if "time" in limit_drng:
        if column == 0:
            clip_drng = True
    if "freq" in limit_drng:
        if row == 0:
            clip_drng = True
    if "delay" in limit_drng:
        if row == 1:
            clip_drng = True
    if "fringe_rate" in limit_drng:
        if column == 1:
            clip_drng = True
            
    if not clip_drng:
        vmin, vmax = None, None
        
    cbar_label = vis_label if mode == 'log' else "Phase [rad]"
    fig.sca(ax)
    cax = waterfall(
        use_data, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap
    )
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    if baseline is not None and row == 1:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')
    
    if colorbar_flag==True:
        _ = plt.colorbar(cax)
        _.set_label(cbar_label,c='black',fontsize=16)
        _.ax.tick_params(axis='y',which='both', color='black', labelcolor='black',labelsize=15)
    return cax
