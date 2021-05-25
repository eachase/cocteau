__doc__ = "Tools for common plot types."
__author__ = "Eve Chase <eachase@lanl.gov>"

import glob
import numpy as np

# Matplotlib settings
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
matplotlib.rcParams['xtick.labelsize'] = 24.0
matplotlib.rcParams['ytick.labelsize'] = 24.0
matplotlib.rcParams['axes.titlesize'] = 27.0
matplotlib.rcParams['axes.labelsize'] = 27.0
matplotlib.rcParams['legend.fontsize'] = 24.0
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
matplotlib.rcParams['font.serif'] = ['Computer Modern', 'Times New Roman']
matplotlib.rcParams['font.family'] = ['serif', 'STIXGeneral']
matplotlib.rcParams['legend.frameon'] = True
import matplotlib.pyplot as plt

from cocteau import filereaders, observations


default_bands = ['r-band','g-band', 'z-band', 'J-band', 'K-band']


def plot_many_lightcurves(filenames, bands=default_bands, 
    title=None, angles=0, angle_lines=False, **kwargs):
    """
    Plot multiple LightCurves in the same figure

    Parameters
    ----------
    fileneames : string OR array of strings
        path to the filename(s) of light curves to plot

    bands: array of strings
        labels for bands ot plot

    title: string (optional)
        Title of plot

    mag_col: integer
        which column (corresponding to viewing angle) to
        use for magnitudes

    Returns
    -------
    ax: Axes object
        contains figure information
    """

    if isinstance(filenames, str):
        filenames = np.array([filenames])

    # Check for valid angles
    if isinstance(angles, float):
        # Check for round number then convert to float
        if angles % 1 != 0:
            raise ValueError("Angles must be round integer values.")
        else:
            angles = int(angles)
    if isinstance(angles, int):
        if angles == 0:
            angles = [0]
        else:
            assert angles > 0 and angles <= 54
            angles = np.arange(angles)

    for angle in angles:
        assert angle >= 0 and angle <= 54
        if angle % 1 != 0:
            raise ValueError("Angles must be round integer values.")

    # Lines for each angle
    if angle_lines:
        ls_list = ['solid', 'dotted', 'dashed', 'dashdot']
        if len(angles) > len(ls_list):
            raise ValueError("angle_lines not supported for this many angles")
            

    
    fr = filereaders.LANLFileReader()
    ax = None
    for i, angle in enumerate(angles):
        # Set angle linestyles
        if angle_lines:
            ls = ls_list[i]
        else:
            ls = 'solid'

        for filename in filenames:
            for bandname in bands:
                lightcurve = fr.read_lightcurve(filename, bandname=bandname, 
                    angle_col=angle+2)
                ax = lightcurve.plot(bandname, ax=ax, title=title, 
                    ls=ls, **kwargs)

    # Set up legend in super hacky way

    for band in bands:
        ax.axhline(y=10, label=band,
            color=observations.colors[band], **kwargs)
    if angle_lines:
        for i, angle in enumerate(angles):
            ax.axhline(y=10, label=(f'{angle/54*180:.2f} deg off-axis'),
                color='k', ls=ls_list[i], **kwargs)
               
    ax.legend(fontsize=14)

    return ax



def kilonova_param_plot(data_dir, plot_params, title=None, **kwargs):
    """
    Plot several lightcurves with a detailed legend.

    Parameters
    ----------
    data_dir: string
        path to directory containing data cube

    plot_params: dictionary
        dictionary with kilonova parameters. 
        The following keys are required:
        - 'morph': 'P', 'S', or '*'
        - 'wind': 1, 2, or '*'
        - 'md': 0.1, 0.01, 0.001, or '*'
        - 'vd': 0.05, 0.3, or '*'
        - 'mw': 0.1, 0.01, 0.001, or '*'
        - 'vw': 0.05, 0.3, or '*'

    title: string, optional
        Title of plot
    """

    morph = plot_params['morph']
    wind = plot_params['wind']
    md = plot_params['md']
    vd = plot_params['vd']
    mw = plot_params['mw']
    vw = plot_params['vw']
    angles = plot_params['angles']    

    filenames = glob.glob(f"{data_dir}Run_T{morph}_dyn_all_lanth_wind{wind}_all_md{md}_vd{vd}_mw{mw}_vw{vw}_mags_*.dat")

    ax = plot_many_lightcurves(filenames, title=title, angles=angles, 
        **kwargs)
    
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height* 0.7])


    dyntitle = r'$\textbf{Dynamical Ejecta:} \\ \indent \indent$'
    if md == 0.1:
        md1 = r'Mass: $\textbf{0.1},$ '
        md2 = r'$0.01,$ '
        md3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    elif md == 0.01:    
        md1 = r'Mass: $0.1,$ '
        md2 = r'$\textbf{0.01},$ '
        md3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    elif md == 0.001:
        md1 = r'Mass: $0.1,$ '
        md2 = r'$0.01,$ '
        md3 = r'$\textbf{0.001} M_{\odot}  \\ \indent \indent$'
    else:
        md1 = r'Mass: $0.1,$ '
        md2 = r'$0.01,$ '
        md3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    if vd == 0.05:
        vd1 = r'Velocity: $ \textbf{0.05},$ '
        vd2 = r'$0.3c$'
    elif vd == 0.3:
        vd1 = r'Velocity: $ 0.05,$ '
        vd2 = r'$\textbf{0.3}c$'
    else:    
        vd1 = r'Velocity: $ 0.05,$ '
        vd2 = r'$0.3c$'

    dynamicalstr = dyntitle + md1 + md2 + md3 + vd1 + vd2


    windtitle = r'$\textbf{Wind Ejecta:} \\ \indent \indent$'
    if morph == 'P':
        morph1 = r'Morphology: $\mathrm{Spherical},$ '
        morph2 = r'$\mathrm{\textbf{Peanut}} \\ \indent \indent$'    
    elif morph == 'S':
        morph1 = r'Morphology: $\mathrm{\textbf{Spherical}},$ '
        morph2 = r'$\mathrm{Peanut} \\ \indent \indent$'
    else:    
        morph1 = r'Morphology: $\mathrm{Spherical},$ '
        morph2 = r'$\mathrm{Peanut} \\ \indent \indent$'
    if wind == 1:
        comp1 = r'Composition: $\mathrm{\textbf{Wind1}},$ '
        comp2 = r'$\mathrm{Wind2} \\ \indent \indent$'
    elif wind == 2:
        comp1 = r'Composition: $\mathrm{Wind1},$ '
        comp2 = r'$\mathrm{\textbf{Wind2}} \\ \indent \indent$'    
    else:
        comp1 = r'Composition: $\mathrm{Wind1},$ '
        comp2 = r'$\mathrm{Wind2} \\ \indent \indent$'
    if mw == 0.1:
        mw1 = r'Mass: $\textbf{0.1},$ '
        mw2 = r'$0.01,$ '
        mw3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    elif mw == 0.01:
        mw1 = r'Mass: $0.1,$ '
        mw2 = r'$\textbf{0.01},$ '
        mw3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    elif mw == 0.001:
        mw1 = r'Mass: $0.1,$ '
        mw2 = r'$0.01,$ '
        mw3 = r'$\textbf{0.001} M_{\odot}  \\ \indent \indent$'
    else:
        mw1 = r'Mass: $0.1,$ '
        mw2 = r'$0.01,$ '
        mw3 = r'$0.001 M_{\odot}  \\ \indent \indent$'
    if vw == 0.05:
        vw1 = r'Velocity: $ \textbf{0.05},$ '
        vw2 = r'$0.3c$'
    elif vw == 0.3:
        vw1 = r'Velocity: $ 0.05,$ '
        vw2 = r'$\textbf{0.3}c$'
    else:    
        vw1 = r'Velocity: $ 0.05,$ '
        vw2 = r'$0.3c$'

    windstr = windtitle + morph1 + morph2 + comp1 + comp2 + \
        mw1 + mw2 + mw3 + vw1 + vw2 


    keystr = 'Bold parameters are fixed. If a \n line has no bold, parameters \n span all possible combinations.'


    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.70, dynamicalstr, fontsize=18, 
        transform=ax.transAxes, bbox=props)
    ax.text(1.05, 0.325, windstr, fontsize=18, 
        transform=ax.transAxes, bbox=props)
    ax.text(1.05, 0.05, keystr, fontsize=18, 
        transform=ax.transAxes, bbox=props)


    return ax




