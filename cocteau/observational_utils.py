__doc__ = "Tools for common plot types."
__author__ = "Eve Chase <eachase@lanl.gov>"

from astropy import units as u
try:
    from astropy.cosmology import Planck18_arXiv_v2
except:
    from astropy.cosmology import Planck15

from astropy.time import Time
import glob
import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Local imports
from cocteau import filereaders, observations

def absMag(mag, distance):
    """
    Convert apparent magnitude to absolute magnitude
    """
    # Convert distance to parsec
    distance = distance.to(u.pc)
    
    # Convert to absolute magnitude
    return (mag - 2.5 * np.log10((distance / (
        10 * u.pc))**2)).value

def appMag(mag, distance):
    """
    Convert absolute magnitude into apparent
    """
    # Convert distance to parsec
    distance = distance.to(u.pc)
    
    # Convert to apparent magnitude
    return (mag + 2.5 * np.log10((distance / (
        10 * u.pc))**2))


def mjd_to_days(mjd_time, mergertime_gps=1249852257.0130):

    # Store time in astropy MJD format
    time = Time(mjd_time, format='mjd')

    # Convert to GPS time
    time.format = 'gps'

    # Compare to gps time of merger
    postmergertime = (time.value - mergertime_gps) * u.second

    # Convert time post-merger to days
    return postmergertime.to(u.day)


def read_upperlimits(filename, distance, usecols=[2,3,4],
    names=['time','band','m']):
    """
    Format upperlimits into a specific dataframe
    """

    # Read in the observations
    upperlimits = pd.read_csv(filename, usecols=usecols, 
        skiprows=1, names=names)

    # If time is NAN, replace with time above
    upperlimits = upperlimits.fillna(method='ffill')

    # Convert data types
    upperlimits['m'] = upperlimits['m'].astype(float)
    upperlimits['time'] = upperlimits['time'].astype(float)
    upperlimits['M'] = upperlimits.apply(lambda row: absMag(
        row['m'], distance), axis=1)

    return upperlimits


def read_kn_data(data_dir, angle_factor=5, lum=False,
    bandnames=None):

    angles = np.arange(0, 54, angle_factor)
    print(angles)
    plot_params = {
        'morph' : '*',
        'wind' : '*',
        'md' : '*',
        'vd' : '*',
        'mw' : '*',
        'vw' : '*',
    }
    
    if bandnames is None:
        bandnames = ['r-band','i-band','J-band','K-band']
    num_bands = len(bandnames)

    morph = plot_params['morph']
    wind = plot_params['wind']
    md = plot_params['md']
    vd = plot_params['vd']
    mw = plot_params['mw']
    vw = plot_params['vw']

    # Read in the file
    if lum:
        filetype = 'lums'
    else:
        filetype = 'mags'

    filenames = glob.glob(f"{data_dir}Run_T{morph}_dyn_all_lanth_wind{wind}_all_md{md}_vd{vd}_mw{mw}_vw{vw}_{filetype}_*.dat")

    fr = filereaders.LANLFileReader()
    return fr.read_magmatrix(filenames, bandnames, angles=angles,
        lum=lum)


def read_blackbody(bb_file, distance):
    """
    Read blackbody file

    Assumes units are provided in milliJansky
    """

    # Read txt file
    bb_data = pd.read_csv(bb_file, 
        delim_whitespace=True)

    # Process flux data
    updated_cols = []
    for col in bb_data.columns:
        if 'Fnu' in col:
            # Convert to Jansky
            bb_data[col] /= 1000

            updated_cols.append(
                f"flux{col.split('[')[1].split(']')[0][:-1]}")
        else:
            updated_cols.append(col)

    # Overwrite column names
    bb_data.columns = updated_cols

    # Convert to magnitudes
    for col in bb_data.columns:
        if 'flux' in col:
            percentage = float(col[4:])
            if percentage % 1 == 0:
                percentage = int(percentage)

            m_name = f'm{percentage}'

            # Convert to apparent magnitudes
            bb_data[m_name] = bb_data.apply(
                lambda row: -2.5 * np.log10(
                row[col] / 3631), axis=1)        

            # Convert to absolute magnitudes
            bb_data[f'M{percentage}'] = bb_data.apply(
                lambda row: absMag(row[m_name], distance),
                axis=1)

    return bb_data


def compute_at2017gfo(data_file, band, lim_mag, 
    redshifts=np.array(
        [0.001, 0.005, 0.0098, 0.0125, 0.025, 0.05, 0.075, 0.1, 
        0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7])):
    """
    Compute detectability contour for AT2017gfo in a given band
    """

    # Read in GW170817 spectra
    fr = filereaders.ObservationalFileReader()
    spectra = fr.read_spectra(data_file)

    max_z_in_epoch = {}
    epochs = np.floor(spectra.times)
    for epoch in np.unique(epochs):
        max_z_in_epoch[epoch] = 0

    # Output data from scatterplot
    obs_time_arr = []
    redshift_arr = []
    detectable_arr = []

    # Get wavelength ranges from band
    band_wl = band.wavelength_arr.value
    min_wl = band_wl[0]
    max_wl = band_wl[-1]


    fig, ax = plt.subplots()
    for obs_time, spectrum in spectra.spectra.items():

        rest_time = spectrum.rest_time.to(u.day).value

        epoch = np.floor(rest_time) 
        
        
        for redshift in redshifts:
            
            # Determine if spectrum is in range
            wavelength_arr = spectrum.wavelength_arr.value * (1 + redshift)
            min_wl_spec = np.min(wavelength_arr)
            max_wl_spec = np.max(wavelength_arr)

            in_range = False
            if min_wl_spec <= min_wl:
                if max_wl_spec >= max_wl:
                    # Restrict to positive flux values
                    flux_arr = spectrum.flux_density_arr.value
                    flux_in_range = flux_arr[np.where((wavelength_arr >= min_wl) & (wavelength_arr <= max_wl))[0]]
            
                    if np.all(flux_in_range >= 0):
                        in_range = True

        
        
            # Plot if in range
            if in_range:

                abs_mag = observations.compute_magnitude_at_timestep(spectrum,
                    band, redshift=redshift)

                # Redshift the times
                dist_lum = Planck18_arXiv_v2.luminosity_distance(redshift)

                # Convert absolute magnitudes to apparent
                app_mag = appMag(abs_mag.value, dist_lum)


                obs_time_arr.append(rest_time * (1 + redshift))
                redshift_arr.append(redshift)

                # Detectable
                if app_mag < lim_mag:
                    ax.scatter(rest_time * (1 + redshift), redshift, 
                        marker='o', color='k')
                    if redshift > max_z_in_epoch[epoch]:
                        max_z_in_epoch[epoch] = redshift 
                    detectable_arr.append(1)

                # Not detectable
                else:
                    ax.scatter(rest_time * (1 + redshift), redshift, 
                        marker='x', color='r')
                    detectable_arr.append(0)

    # Maximum z contour
    epoch_times = []
    max_z_arr = []
    for epoch in np.unique(epochs):
        max_z = max_z_in_epoch[epoch]
        if max_z > 0:
            epoch_times.append(np.mean(spectra.times[np.where(epochs == epoch)]) * (
                1 + max_z))
            max_z_arr.append(max_z)

    # Plot actual maximum z line        
    ax.plot(epoch_times, max_z_arr)            

    # Plot smoothed maximum z contour
    try:
        smooth_interp = savgol_filter(max_z_arr, 11, 3)
    except:
        try:
            smooth_interp = savgol_filter(max_z_arr, 7, 3)
        except:
            try:
                smooth_interp = savgol_filter(max_z_arr, 3, 1)
            except:
                smooth_interp = None

    return epoch_times, smooth_interp




