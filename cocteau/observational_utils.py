__doc__ = "Tools for common plot types."
__author__ = "Eve Chase <eachase@lanl.gov>"

from astropy import units as u
from astropy.time import Time
import glob
import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad

# Local imports
from cocteau import filereaders

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







