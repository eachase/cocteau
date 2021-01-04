#!/usr/bin/env python

__author__ = 'Eve Chase <eachase@lanl.gov>'

import argparse
from astropy import units as u
from astropy.cosmology import Planck18_arXiv_v2
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import root, fminbound

# Matplotlib settings
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12.0, 8.0)
matplotlib.rcParams['xtick.labelsize'] = 24.0
matplotlib.rcParams['ytick.labelsize'] = 24.0
matplotlib.rcParams['axes.titlesize'] = 27.0
matplotlib.rcParams['axes.labelsize'] = 27.0
matplotlib.rcParams['legend.fontsize'] = 24.0
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
#matplotlib.rcParams['font.serif'] = ['Computer Modern', 'Times New Roman']
#matplotlib.rcParams['font.family'] = ['serif', 'STIXGeneral']
matplotlib.rcParams['legend.frameon'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


# Local imports
from cocteau import matrix
from cocteau import upperlimit_utils as utils

prop_colors = ["#004949","#ff6db6","#006ddb","#009292",
        "#ffb6db","#490092","#b66dff","#6db6ff","#b6dbff",
        "#920000","#924900","#db6d00","#24ff24","#ffff6d"]

titles = {
    'white': 'UVOT/White',
    'V': 'UVOT/V',
    'GeminiR': 'Gemini/r-band',
    'F606W': 'HST/F606W',
    'F125W': 'HST/F125W',
    'F160W': 'HST/F160W',
    'i-band': 'i-band',
    'K-band': 'K-band',
    'UVW2': 'Swift/UVOT.UVW2',
    'F070W': 'JWST/NIRCam.F070W',
    'F090W': 'JWST/NIRCam.F090W',
    'F115W': 'JWST/NIRCam.F115W',
    'F140M': 'JWST/NIRCam.F140M',
    'F150W': 'JWST/NIRCam.F150W',
    'F200W': 'JWST/NIRCam.F200W',
    'F277W': 'JWST/NIRCam.F277W',
    'F356W': 'JWST/NIRCam.F356W', 
    'F444W': 'JWST/NIRCam.F444W',
    'F480M': 'JWST/NIRCam.F480M',
    'F560W': 'JWST/MIRI.F560W',
    'F770W': 'JWST/MIRI.F770W',
    'F1000W': 'JWST/MIRI.F1000W',
    'F1280W': 'JWST/MIRI.F1280W',
    'F1800W': 'JWST/MIRI.F1800W',
    'F2100W': 'JWST/MIRI.F2100W',
    'F2550W': 'JWST/MIRI.F2550W',
    'r-band': 'r-band'
}

prop_titles = {
    'morph' : 'Wind Morphology',
    'wind' : 'Wind Composition',
    'md' : r'Dynamical Ejecta Mass [$M_{\odot}$]',
    'vd' : r'Dynamical Ejecta Velocity [$c$]',
    'mw' : r'Wind Mass [$M_{\odot}$]',
    'vw' : r'Wind Velocity [$c$]',
    'angle': 'Angular Bin'
}

band_colors = {
    #'F606W': 'blue',
    #'F125W': 'red',
    'GeminiR': 'blue',
    'F606W': 'purple',
    'F160W': 'orange',
    'F125W': 'red',  
    'UVW2': 'purple'
}


def set_figname(fig_dir, band=None, prop=None, propval=None, 
    color_prop=None, color_massthresh=None, vel_thresh=False,
    angles=None, contour=False, redshift=None):

    figname = f'{fig_dir}/lightcurves'

    # Set band
    if band is not None:
        figname += f'_{band}'

    # List redshift
    if redshift is not None:
        figname += f'_{redshift}'

    # Set prop
    if prop is not None and propval is not None:
        figname += f'_{prop}_{propval}'
    else:
        figname += '_allprop'

    # Set colorprop
    if color_prop is not None:
        figname += f'_color_{color_prop}'

    # Set angles
    if angles is not None:
        figname += f"_angles_{('_').join([str(i) for i in angles])}"

    # Set mass threshold
    if color_massthresh is not None:
        figname += f'_massthresh{color_massthresh:.3}'

    # Velocity threshold
    if vel_thresh:
        figname += '_curtain'

    # If contour bands are shown
    if contour:
        figname += '_contours'

    figname += '.png'

    return figname



# FIXME: this would all make more sense as a class
def plot_bands(bandnames, magmatrix, fig_dir,
    upperlimits=None, upperlimits_wf=None, 
    prop=None, propval=None, color_prop=None, 
    color_massthresh=None, dist_lum=None,
    angles=None, vel_thresh=False, contour=False,
    redshift=0, gw170817data=None, lim_mag=None,
    legend=False):

    # Set up multipanel figure -- FIXME don't hardcode subplot dimensions
    n_row = len(bandnames) #(len(bandnames) + 1) // 2
    fig, ax = plt.subplots(1, 1,
        dpi=100, linewidth=2, figsize=(12, 10))

    # Flatten axes array
    #ax_arr = ax.flatten()
    ax_arr = [ax]

    # For each bandname
    for idx, band in enumerate(bandnames):
        ax = ax_arr[idx]

        ax_arr[idx] = plot_lightcurves(band, magmatrix, fig_dir,
            upperlimits, upperlimits_wf, prop, propval,
            color_prop, color_massthresh, vel_thresh, dist_lum,
            angles, save_fig=False, ax=ax, legend=legend,
            inconsistent_color='k', lim_mag=lim_mag,
            contour=contour, redshift=redshift)
        
        if lim_mag is not None:
            ax_arr[idx].axhline(y=lim_mag, c='r', lw=2)
        ax_arr[idx].axvline(x=14, c='k', lw=2, ls='-.')


        if gw170817data is None:
            pass
        else:
            if band in gw170817data.keys():
                band_data = gw170817data[band]
                #ax_arr[idx].scatter(band_data['time'],
                #    band_data['m160624A'],
                #    marker='d', color='k')

                # Interpolate data
                times = band_data['time']
                mags = band_data['m']

                spl = UnivariateSpline(times, mags)

                ax_arr[idx].plot(times,
                    spl(times), color='k', lw=5, ls='--')

                if idx == 0:
                    label = 'AT2017gfo'
                    for legend in [c for c in \
                        ax_arr[idx].get_children() \
                        if isinstance(c, matplotlib.legend.Legend)]:

                        # Grab legend properties
                        ax_lgd = legend.axes
                        handles, labels = ax_lgd.get_legend_handles_labels()                  

                        # Make a scatterpoint off the plot
                        scatt = ax_arr[idx].scatter(-10, 10, marker='d',
                            color='k')
                        extra_line, = ax_arr[idx].plot([-10,-11],
                            [10,11], color='k', lw=5, ls='--')

                        # Manually add scatterpoint to legend
                        handles.append(extra_line)
                        labels.append('AT2017gfo')

                        # Overwrite the legend
                        ax_arr[idx].legend(handles, labels)


    # Save figure
    figname = set_figname(fig_dir, prop=prop, band=band,
        propval=propval, color_prop=color_prop, 
        color_massthresh=color_massthresh, 
        vel_thresh=vel_thresh, angles=angles,
        contour=contour, redshift=redshift)
    plt.savefig(figname)
    print(f'Saved to {figname}')

    return ax_arr


def plot_lightcurves(band, magmatrix, fig_dir,
    upperlimits=None, upperlimits_wf=None, 
    prop=None, propval=None, color_prop=None, 
    color_massthresh=None, vel_thresh=False, 
    dist_lum=None, angles=None, save_fig=True, 
    legend=True, ax=None, inconsistent_color='#bf80ff',
    contour=False, redshift=0, lim_mag=None):
    """
    Plot several light-curves in one figure

    Parameters
    ----------
    band: str
        bandname

    magmatrix: MagMatrix object
        collection of light-curves

    fig_dir: string
        path to directory to store figure

    upperlimits: pandas data frame
        LDT upper limits

    upperlimits_wf: pandas data frame
        wide-field upper limits, used in parameter constraints

    prop: str
        limit light-curves to only those with prop
        with a value of propval

    propval: float
        value of prop to include in plot

    color_prop: str
        property to color light-curves by

    color_massthresh: float
        value of wind mass to differentiate lightcurves by

    Returns
    _______
    ax: figure
    """

    # Copy magmatrix
    sim_matrix = magmatrix.matrix.copy()
    knprops = magmatrix.knprops.copy()

    # Trim data to only angles of interest
    if angles is not None:
        # Find indices for each angle
        angle_idx = np.array([])
        for angle in angles:
            angle_idx = np.append(angle_idx,
                np.where(knprops['angle'] == angle)[0])
        angle_idx = np.unique(angle_idx)

        # Trim data frames
        sim_matrix = sim_matrix.loc[angle_idx]
        knprops = knprops.loc[angle_idx]

    # Trim based on velocity threshold
    if vel_thresh:
        vel_idx = np.where(knprops['vd'] > knprops['vw'])[0]
        knprops = knprops.loc[vel_idx]
        sim_matrix = sim_matrix.loc[vel_idx] 

    # Restrict light curves to only points with appropriate property
    if prop is not None:
        prop_idx = np.where((
            knprops[prop] == propval).values)[0]
        if len(prop_idx) > 0:
            sim_matrix = sim_matrix.loc[prop_idx].dropna()

    # Restrict to band of interest
    prop_band_magmatrix = sim_matrix[band]
    
    # Define times
    times = np.unique(magmatrix.times.values)

    # Set up colors
    if color_prop:
        # Get array of unique values for result
        color_prop_uniquevals = np.sort(np.unique(
            knprops[color_prop]))
        
        # Set up dictionary of color
        color_prop_dict = {}
        for idx, color_prop_val in enumerate(color_prop_uniquevals):
            # OMG these variable names are a disaster
            color_prop_dict[color_prop_val] = prop_colors[idx]

    # Select all light curves consistent with upper limits
    consistent_obs = prop_band_magmatrix.copy()

    # Set up array of consistent indices
    inconsistent_idx_arr = []


    # Exclude parameters based on wide-field upperlimits
    if upperlimits_wf is not None:
        upperlimits_wf = upperlimits_wf[upperlimits_wf['band'] == f'{band}']
        # For each upper limit measurement
        for i, obs in upperlimits_wf.iterrows():

            if obs.time >= np.min(times):
                # Find time in magmatrix closest to this value
                closest_time_idx = np.argmin(np.fabs(times - obs.time))
                closest_time = times[closest_time_idx]

                # Observations which contain the given time
                try:
                    obs_at_time = consistent_obs.loc(
                        axis=0)[:, closest_time_idx]
                    idx_at_time = obs_at_time.index.get_level_values(
                        level=0)

                    # Identify which are eliminated by upper limit
                    inconsistent_obs_idx = idx_at_time[obs_at_time.values < obs.m]
                    num_eliminated_obs = inconsistent_obs_idx.shape[0]

                    # Restrict observations to only viable indices
                    if num_eliminated_obs > 0:
                        consistent_obs = consistent_obs.drop(
                            inconsistent_obs_idx)
                    print(i, band, closest_time, obs.time, obs.m, 
                        num_eliminated_obs)
                except KeyError:
                    pass

    # Record indices of consistent observations        
    consistent_idx = np.unique(consistent_obs.index.get_level_values(level=0))
    print(consistent_idx.shape)
    
    # Plot the light curves
    if ax is None:
        fig, ax = plt.subplots(dpi=100, linewidth=2)

    # Set up to plot contours instead
    if contour:

        if color_prop is not None:
            contour_dict = {}
            for color_prop_val, color in color_prop_dict.items():
                contour_lower = 50 * np.ones_like(times)
                contour_upper = np.zeros_like(times)
                contour_dict[color_prop_val] = {
                    'lower': contour_lower,
                    'upper': contour_upper,
                }
        # For each time step record the maximum extent of 
        # consistent and inconsistent light curves
        inconsistent_lower = 50 * np.ones_like(times)
        inconsistent_upper = np.zeros_like(times)
        consistent_lower = 50 * np.ones_like(times)
        consistent_upper = np.zeros_like(times)


    # Iterative over each lightcurve
    lc_legend = True
    for idx, lc in prop_band_magmatrix.groupby(level=0):
        
        # Set times
        time_idx = lc.index.get_level_values(level=1).values
        times_with_units = times[time_idx] * u.day
        #times_to_plot = times[time_idx]        
        times_to_plot = magmatrix.times.loc[[idx]].values

        label = None
        zorder = 0 
        if color_massthresh is not None:
            # Compute the wind mass
            lc_params = knprops.iloc[idx]
            wind_mass = lc_params['mw'] 
            
            if wind_mass >= color_massthresh:
                alpha = 0.2
                #color = '#bf80ff'
                if idx in consistent_idx:
                    color = '#f2b602'
                else:
                    color = '#b81414'
            else:
                color = '0.75'
                alpha = 0.05

                    
            # Make label
            if lc_legend:
                over_label = r'Wind Ejecta Mass $\geq$' + \
                    f'{color_massthresh:.3} ' + r'$M_{\odot}$'
                under_label = r'Wind Ejecta Mass $<$' + \
                    f'{color_massthresh:.3} ' + r'$M_{\odot}$'
                ax.plot(-times_to_plot, lc.values, 
                    alpha=1, color='#f2b602', 
                    label=('Consistent: ' + over_label))
                ax.plot(-times_to_plot, lc.values, 
                    alpha=1, color='#b81414', 
                    label=('Inconsistent: ' + over_label))
                ax.plot(-times_to_plot, -lc.values, 
                    alpha=1, color='0.75', label=under_label) 
               
    
                lc_legend = False
                
                
        # Plot consistent light curves with no differentiation
        # based on mass        
        elif idx in consistent_idx:
            # No coloring based on property
            if color_prop is None:
                color = '0.75'
            # Consistent light curves with colors based on property
            else:
                color = color_prop_dict[knprops.iloc[idx][color_prop]]
            alpha = 0.2

        # Inconsistent light curves that are not colored by property
        elif color_prop is None:
            color = inconsistent_color
            alpha = 0.4

        # Inconsistent light curves that are colored by property
        else:
            color = color_prop_dict[knprops.iloc[idx][color_prop]]
            alpha = 0.2

        if contour:
            # Record indices of timesteps
            time_len = times_to_plot.shape[0]

            if np.any(np.isinf(lc.values)):
                time_len = np.where(lc.values == np.inf)[0][0]
                lc_to_plot = lc.values[:time_len]
            else:
                lc_to_plot = lc.values

            # Plot by property
            if color_prop is not None:
                contour_lower, contour_upper = \
                    contour_dict[knprops.iloc[idx][color_prop]].values()
                idx_to_change = np.where(
                    contour_upper[:time_len] < lc_to_plot)[0]
                contour_upper[idx_to_change] = lc_to_plot[idx_to_change]

                idx_to_change = np.where(
                    contour_lower[:time_len] > lc_to_plot)[0]
                contour_lower[idx_to_change] = lc_to_plot[idx_to_change]



            # Check if light curve is consistent:
            elif idx in consistent_idx:
                # Compare to existing consistent contour
                idx_to_change = np.where(
                    consistent_upper[:time_len] < lc_to_plot)[0]
                consistent_upper[idx_to_change] = lc_to_plot[idx_to_change]

                idx_to_change = np.where(
                    consistent_lower[:time_len] > lc_to_plot)[0]
                consistent_lower[idx_to_change] = lc_to_plot[idx_to_change]

            else:
                # Compare to existing inconsistent contour
                idx_to_change = np.where(
                    inconsistent_upper[:time_len] < lc_to_plot)[0]
                inconsistent_upper[idx_to_change] = lc_to_plot[idx_to_change]

                idx_to_change = np.where(
                    inconsistent_lower[:time_len] > lc_to_plot)[0]
                inconsistent_lower[idx_to_change] = lc_to_plot[idx_to_change]

        else:
            ax.plot(times_to_plot, lc.values, alpha=alpha, 
                zorder=zorder, c=color, label=label)
    if contour:
        # Plot contours
        if color_prop is not None:
            for color_prop_val, color in color_prop_dict.items():
                title_split = prop_titles[color_prop].split(' ')
                if color_prop == 'wind':
                    label = color_prop_val

                else:
                    label = f"{(' ').join(title_split[:-1])}: {color_prop_val:.3} {title_split[-1][1:-1]}"
                contour_lower, contour_upper = \
                    contour_dict[color_prop_val].values()

                            
                #ax.plot(times, contour_upper, c=color)
                ax.plot(times, contour_lower, c=color, label=label)

                if color_prop != 'wind':
                    ax.plot(times, contour_upper, c=color)
                    ax.fill_between(times, contour_upper, contour_lower,
                        color=color, alpha=0.2)

                # Record time each constraint is made
                if lim_mag is not None and color_prop_val != 0.001:
                    min_idx = int(25*(1 + redshift/10))
                    max_idx = min(65, len(times) - 1)
                    time_constraint = times[min_idx:max_idx][np.argmin(
                        np.fabs(contour_upper[min_idx:max_idx] - lim_mag))]
                    if time_constraint > times[min_idx] and \
                        time_constraint < times[max_idx-1]:
                        print(color_prop_val, time_constraint)
                        ax.axvline(time_constraint, lw=2, ls='-.', color=color)

        else:
            # Don't plot contour if no points are excluded
            if consistent_idx.shape[0] != knprops.shape[0]:
                ax.plot(times, inconsistent_upper, c=inconsistent_color)
                ax.plot(times, inconsistent_lower, c=inconsistent_color)
                ax.fill_between(times, inconsistent_upper, 
                    inconsistent_lower, color=inconsistent_color, 
                    alpha=0.2)

            ax.plot(times, consistent_upper, c='0.75')
            ax.plot(times, consistent_lower, c='0.75')
            ax.fill_between(times, consistent_upper, consistent_lower,
                color='0.75', alpha=0.2)

    # Make legend for plots colored by properties
    if color_prop is not None and contour is False:
        if color_prop == 'angle':
            for angle_idx, color in color_prop_dict.items():
                label = f'Angular Bin {int(angle_idx)}'
                ax.plot(-times[time_idx], -lc.values, alpha=1, 
                    color=color, label=label)


        else:
            for color_prop_val, color in color_prop_dict.items():
                title_split = prop_titles[color_prop].split(' ')
                label = f"{(' ').join(title_split[:-1])}: {color_prop_val:.3} {title_split[-1][1:-1]}"

                ax.plot(-times[time_idx], -lc.values, alpha=1, 
                    color=color, label=label)

    # Scatterplot of galaxy-targeted upper limits
    if upperlimits is not None:
        upperlimits = upperlimits[upperlimits['band'] == f'{band}']
        if len(upperlimits) > 1:
            ax.scatter(upperlimits.time, upperlimits.m, 
                marker='v', s=380, c='k', zorder=1, label='LDT')

    # Scatterplot of widefield upper limits
    if upperlimits_wf is not None:
        label = 'Upper Limit'
        ax.scatter(upperlimits_wf.time, upperlimits_wf.m, 
            marker='v', s=380, lw=2, c='none', edgecolor='k', 
            zorder=1, label=label)

    # Include horizontal line for JWST
    if band == 'JWST' and dist_lum is not None:
        limit = utils.absMag(28.8, dist_lum)
        ax.axhline(y=limit, ls='-.', color='k', 
            label='Limiting Magnitude')

    ax.set_xscale('log')
    ax.set_xlabel('Observer frame time since merger (d)')
    y_ticks = np.linspace(-20, 0, 5)
    if band in ['white', 'V']:
        ax.set_xlim([1e-3, 20])
        ax.set_ylim([-25, 5])
        xtick_vals = [1e-3, 1e-2, 1e-1, 1, 10]
    else:
        ax.set_xlim([0.125 * (1 + redshift), 30 * (1 + redshift)])
        if redshift == 0:
            xtick_vals = [0.125, 1, 10]
        else:
            xtick_vals = [1, 10]
        ax.set_ylim([15, 33]) #47])
        #ax.set_ylim([22.23, 45])
        y_ticks = np.linspace(15, 30, 6) #33, 7)
        #y_ticks = np.linspace(23, 43, 10)
       
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_vals)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
    
    ax.minorticks_on()
    ax.tick_params(which='major', direction='in', length=7, 
        width=0.5, top=True, bottom=True, right=True, left=True)
    ax.tick_params(which='minor', direction='in', length=4, 
        width=0.5, top=True, bottom=True, right=True, left=True)
    
    ax.invert_yaxis()

    ax.set_ylabel(f'Apparent magnitude')
    if legend:
        ax.legend()#loc='lower left')
    

    # Set title based on band name
    ax.set_title(titles[band])

    # Set figure name based on properties
    if save_fig:
        if prop is not None:

            if color_prop is not None:
                plt.savefig(f'{fig_dir}/lightcurves_{band}_{prop}_{propval}_color_{color_prop}.png')
            elif angles is not None:
                plt.savefig(f"{fig_dir}/lightcurves_{band}_{prop}_{propval}_angles_{('_').join([str(i) for i in angles])}.png")
         
            else:
                plt.savefig(f'{fig_dir}/lightcurves_{band}_{prop}_{propval}.png')
            
        elif color_massthresh is not None:
            plt.savefig(f'{fig_dir}/lightcurves_{band}_allprop_massthresh{color_massthresh:.3}.png')
        elif color_prop is not None:
            plt.savefig(f'{fig_dir}/lightcurves_{band}_allprop_color_{color_prop}.png')
        elif angles is not None:
            plt.savefig(f"{fig_dir}/lightcurves_{band}_allprop_angles_{('_').join([str(i) for i in angles])}.png")
     

        else:
            plt.savefig(f'{fig_dir}/lightcurves_{band}_allprop.png')
            
    return ax








if __name__ == '__main__':

    # Input parameters
    parser = argparse.ArgumentParser()

    # Set input directory
    parser.add_argument('-i', '--input-dir', type=str,
        default='./')

    # Set output directory for plots
    parser.add_argument('--fig-dir', type=str,
        default='figures/')

    # Provide distance to kilonova
    parser.add_argument('--redshift', type=float, default=0.0098)

    parser.add_argument('-l', '--lim-mag', type=float,
        default=None)

    parser.add_argument('-r', '--rest-band', type=str,
        default=None)

    parser.add_argument('-f', '--filters', nargs='+', type=str)

    args = parser.parse_args()

    # Convert redshift to distance
    dist_lum = Planck18_arXiv_v2.luminosity_distance(args.redshift)
    print(dist_lum)

    # Read in the upper limits
    #upperlimits = utils.read_upperlimits(
    #    'upperlimits.csv', dist_lum, usecols=[0,1,2])

    # Set bandnames
    bandnames = args.filters

    # Reconstruct magmatrix from pickles
    redshift = args.redshift
    # FIXME - this is too hardcode-y
    band = bandnames[0]
    magmatrix = pd.read_pickle(
        glob.glob(f'{args.input_dir}magmatrix_*{band}*z{redshift}.pkl')[0])
    times = pd.read_pickle(
        glob.glob(f'{args.input_dir}times_*{band}*z{redshift}.pkl')[0])
    knprops = pd.read_pickle(
        glob.glob(f'{args.input_dir}knprops_*{band}*z{redshift}.pkl')[0])
    
    # Redshift the times
    redshift_factor = 1 + redshift
    #upperlimits['time'] /= redshift_factor
    times *= redshift_factor

    # Convert absolute magnitudes to apparent
    magmatrix = utils.appMag(magmatrix, dist_lum)

    # Manually overwrite angles in knprops
    knprops.loc[knprops['angle'] == 30, 'angle'] = 9
    knprops.loc[knprops['angle'] == 60, 'angle'] = 18
    knprops.loc[knprops['angle'] == 90, 'angle'] = 27
    magmatrix = matrix.MagMatrix(magmatrix, bandnames,
        times, knprops)


    # Read in GW170817 data
    data_dir = '/Users/r349989/Documents/kilonovae/data/GW170817/'
    gw170817 = {}

    # Redshifted bandpairs
    rest_band = args.rest_band
    if rest_band is not None:
        gw170817_data = pd.read_csv(
            f'{data_dir}GW170817_{rest_band}band.csv', 
            usecols=[1,2])

        # Convert to absolute mangitude
        gw170817_data['M'] = gw170817_data.apply(
            lambda row: utils.absMag(row['m'], 43.2 * u.Mpc),
            axis=1)

        # Convert to apparent magnitude at plotted redshift
        gw170817_data['m'] = gw170817_data.apply(
            lambda row: utils.appMag(row['M'], dist_lum),
            axis=1)

        # Redshift the times
        gw170817_data['time'] *= redshift_factor

        # Add to dictionary
        gw170817[band] = gw170817_data

    plot_bands(bandnames, magmatrix, args.fig_dir,
        upperlimits_wf=None, dist_lum=dist_lum,
        angles=[0], contour=False, redshift=redshift,
        lim_mag=args.lim_mag, gw170817data=gw170817)

    plot_bands(bandnames, magmatrix, args.fig_dir,
        upperlimits_wf=None, dist_lum=dist_lum,
        angles=[0], contour=False, redshift=redshift,
        lim_mag=args.lim_mag, gw170817data=gw170817,
        color_prop='md', legend=True)


    plot_bands(bandnames, magmatrix, args.fig_dir,
        upperlimits_wf=None, dist_lum=dist_lum,
        angles=[0], contour=True, redshift=redshift,
        lim_mag=args.lim_mag, gw170817data=gw170817,
        color_prop='md', legend=True)


