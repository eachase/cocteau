__doc__ = "Tools to read data files containing spectra, bands, and light curves"
__author__ = "Eve Chase <eachase@lanl.gov>"

from astropy import units
import json
import numpy as np
import pandas as pd
import os.path

from cocteau import matrix, observations

morph_keys = {
    'TS': 0,
    'TP': 1,
    'ST': 2,
    'SS': 3,
    'SP': 4,
    'PS': 5,
    'H': 6,
    'P': 7,
    'R': 8,
    'S': 9,
    'T': 10,
}

# Key to filenames -- FIXME: there should be a better way to organize this
# Also it's odd that the filters themselves aren't located in this repository
band_files = {
    'SIBEX':  ['norm', 'SIBEX', 'SIBEX/SIBEX_UV.dat'],
    'u-band': ['norm', 'VRO', 'VRO/u_LSST.dat'],
    'g-band': ['norm', 'VRO', 'VRO/g_LSST.dat'],
    'r-band': ['norm', 'VRO', 'VRO/r_LSST.dat'],
    'i-band': ['norm', 'VRO', 'VRO/i_LSST.dat'],
    'z-band': ['norm', 'VRO', 'VRO/z_LSST.dat'],
    'y-band': ['norm', 'VRO', 'VRO/y_LSST.dat'],
    'J-band': ['tab',  '2MASS', 'J_2MASS.dat'],
    'H-band': ['tab',  '2MASS', 'H_2MASS.dat'],
    'K-band': ['tab',  '2MASS', 'K_2MASS.dat'],
    'B-band': ['norm', 'Swift', 'Swift/Swift_UVOT.B.dat'],
    'U-band': ['norm', 'Swift', 'Swift/Swift_UVOT.U.dat'],
    'UVM2'  : ['norm', 'Swift', 'Swift/Swift_UVOT.UVM2.dat'],
    'UVW1'  : ['norm', 'Swift', 'Swift/Swift_UVOT.UVW1.dat'],
    'UVW2'  : ['norm', 'Swift', 'Swift/Swift_UVOT.UVW2.dat'],
    'V-band': ['norm', 'Swift', 'Swift/Swift_UVOT.V.dat'],
    'white':  ['norm', 'Swift', 'Swift/Swift_UVOT.white.dat'],
    'GeminiR': ['norm', 'Gemini', 'Gemini_GMOS-N.r.dat'],
    'F606W' : ['norm', 'HST', 'HST_ACS_WFC.F606W.dat'],
    'F125W' : ['norm', 'HST', 'HST_WFC3_IR.F125W.dat'],
    'F160W' : ['norm', 'HST', 'HST_WFC3_IR.F160W.dat'],
    'F070W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F070W.dat'],
    'F090W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F090W.dat'],
    'F115W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F115W.dat'],
    'F140M' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F140M.dat'],
    'F150W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F150W.dat'],
    'F200W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F200W.dat'],
    'F277W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F277W.dat'],
    'F356W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F356W.dat'],
    'F444W' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F444W.dat'],
    'F480M' : ['norm', 'JWST', 'JWST/JWST_NIRCam.F480M.dat'],
    'F560W': ['norm', 'JWST', 'JWST/JWST_MIRI.F560W.dat'],
    'F770W': ['norm', 'JWST', 'JWST/JWST_MIRI.F770W.dat'],
    'F1000W': ['norm', 'JWST', 'JWST/JWST_MIRI.F1000W.dat'],
    'F1130W': ['norm', 'JWST', 'JWST/JWST_MIRI.F1130W.dat'],
    'F1280W': ['norm', 'JWST', 'JWST/JWST_MIRI.F1280W.dat'],
    'F1500W': ['norm', 'JWST', 'JWST/JWST_MIRI.F1500W.dat'],
    'F1800W': ['norm', 'JWST', 'JWST/JWST_MIRI.F1800W.dat'],
    'F2100W': ['norm', 'JWST', 'JWST/JWST_MIRI.F2100W.dat'],
    'F2550W': ['norm', 'JWST', 'JWST/JWST_MIRI.F2550W.dat'],
    'RomanR': ['norm', 'Roman', 'Roman/WFIRST_WFI.R062.dat'],
    'RomanZ': ['norm', 'Roman', 'Roman/WFIRST_WFI.Z087.dat'],
    'RomanY': ['norm', 'Roman', 'Roman/WFIRST_WFI.Y106.dat'],
    'RomanJ': ['norm', 'Roman', 'Roman/WFIRST_WFI.J129.dat'],
    'RomanH': ['norm', 'Roman', 'Roman/WFIRST_WFI.H158.dat'],
    'RomanF': ['norm', 'Roman', 'Roman/WFIRST_WFI.F184.dat'],
    'ZTF_g' : ['norm', 'ZTF', 'ZTF/Palomar_ZTF.g.dat'],
    'ZTF_r' : ['norm', 'ZTF', 'ZTF/Palomar_ZTF.r.dat'],
    'ZTF_i' : ['norm', 'ZTF', 'ZTF/Palomar_ZTF.i.dat'],
    'Dorado': ['norm', 'Dorado', 'Dorado/nuv_d.dat'],
    'g-bg': ['norm', 'BlackGEM', 'BlackGEM/g-bg.dat'],
    'i-bg': ['norm', 'BlackGEM', 'BlackGEM/i-bg.dat'],
    'r-bg': ['norm', 'BlackGEM', 'BlackGEM/r-bg.dat'],
    'u-bg': ['norm', 'BlackGEM', 'BlackGEM/u-bg.dat'],
    'vr-bg': ['norm', 'BlackGEM', 'BlackGEM/vr-bg.dat'],
    'z-bg': ['norm', 'BlackGEM', 'BlackGEM/z-bg.dat'],
    'DECam_i': ['norm', 'DECam', 'DECam/CTIO_DECam.i.dat'],
    'DECam_z': ['norm', 'DECam', 'DECam/CTIO_DECam.z.dat'],
    'ULTRASAT': ['norm', 'ULTRASAT', 'ULTRASAT/nuv.dat'],
    'VistaJ': ['norm', 'VISTA', 'VISTA/Paranal_VISTA.J.dat'],
    'VistaKs': ['norm', 'VISTA', 'VISTA/Paranal_VISTA.Ks.dat'],
    'VistaY': ['norm', 'VISTA', 'VISTA/Paranal_VISTA.Y.dat'],
    'VistaZ': ['norm', 'VISTA', 'VISTA/Paranal_VISTA.Z.dat'],
    'VistaH': ['norm', 'VISTA', 'VISTA/Paranal_VISTA.H.dat'],
    'w-band': ['norm', 'DDOTI', 'DDOTI/SLOAN_SDSS.r.dat'],
    'GOTO': ['norm', 'GOTO', 'GOTO/L-band.dat'],
    'PrimeZ': ['norm', 'PRIME', 'PRIME/Z-band.dat'],
    'PrimeY': ['norm', 'PRIME', 'PRIME/Y-band.dat'],
    'PrimeJ': ['norm', 'PRIME', 'PRIME/J-band.dat'],
    'PrimeH': ['norm', 'PRIME', 'PRIME/H-band.dat'],
    'WinterY': ['norm', 'WINTER', 'WINTER/Y-band.dat'],
    'WinterJ': ['norm', 'WINTER', 'WINTER/J-band.dat'],
    'WinterH': ['norm', 'WINTER', 'WINTER/H-band.dat'],
    'Gamow0' : ['norm', 'Gamow',  'Gamow/Gamow0.dat'],
    'Gamow1' : ['norm', 'Gamow',  'Gamow/Gamow1.dat'],
    'Gamow2' : ['norm', 'Gamow',  'Gamow/Gamow2.dat'],
    'Gamow3' : ['norm', 'Gamow',  'Gamow/Gamow3.dat'],
    'Gamow4' : ['norm', 'Gamow',  'Gamow/Gamow4.dat'],
    'RAPTOR' : ['norm', 'RAPTOR', 'RAPTOR/optical.dat'],
    'GenericU': ['norm', 'Generic', 'Generic/Generic_Bessell.U.dat'],
    'GenericR': ['norm', 'Generic', 'Generic/Generic_Bessell.R.dat'],
    'GenericI': ['norm', 'Generic', 'Generic/Generic_Bessell.I.dat'],
    'GenericK': ['norm', 'Generic', 'Generic/Generic_Bessell.K.dat'],
}


class FileReader(object):
    """
    For all your file reading needs
    """
    def __init__(self):
        pass

    def read_spectrum(self, filename):
        pass

    def read_spectra(self, filename):
        pass

    def read_lightcurve(self, filename):
        pass

    def read_band(self, filename, bandname=None, wl_units=None):
        """
        Read in band data.

        Parameters
        ----------
        filename: string
            path to band file

        Returns
        -------
        wavelengths: array_like
            wavelengths in cm

        transmissions: array_like
            Wavelength-dependent transmission percentages.
            Values should vary between 0 and 1.

        wl_units: astropy Unit
            Units of provided wavelengths
        """
        band = None

        try:
            band_data = pd.read_csv(filename, sep=' ',
                names=['wavelength', 'transmission'], dtype=float)
        except:
            print("Use a specialized reader function.")
        else:
            wavelengths = band_data['wavelength'].values * wl_units
            transmissions = band_data['transmission'].values * \
                units.dimensionless_unscaled

            band = observations.Band(bandname, wavelengths,
                transmissions)

        assert band is not None

        #FIXME: assert transmissions between 0 and 1 (put in Band object)

        # Assert wavelength units are in units of length
        wl_units.to(units.cm)

        return band


class TabFileReader(FileReader):
    """Read files set up with tabs"""
    def read_band(self, filename, bandname=None, wl_units=None):
        """
        Read in band data.

        Parameters
        ----------
        filename: string
            path to band file

        Returns
        -------
        wavelengths: array_like
            wavelengths in cm

        transmissions: array_like
            Wavelength-dependent transmission percentages.
            Values should vary between 0 and 1.

        wl_units: astropy Unit
            Units of provided wavelengths
        """
        band = None
        try:
            band_data = pd.read_csv(filename, sep='\t',
                names=['wavelength', 'transmission'], dtype=float)
        except:
            print("Use a specialized reader function.")
        else:
            wavelengths = band_data['wavelength'].values * wl_units
            transmissions = band_data['transmission'].values * \
                units.dimensionless_unscaled

            band = observations.Band(bandname, wavelengths,
                transmissions)

        assert band is not None
        # Assert wavelength units are in units of length
        wl_units.to(units.cm)

        return band





class LANLFileReader(FileReader):
    """
    Read files in the specific format used by LANL's CTA
    """
    def read_spectrum(self, filename, timesteps_in_file, 
        nrows, timestep, wl_units=units.cm, angle=0, remove_zero=False,
        fd_units=(units.erg / units.s / units.cm**2 / units.angstrom)):
        """
        Read in spectra at one timesteps
        for Even et al. (2019) and subsequent 
        paper data format.

        Parameters
        ----------
        filename: string
            path to spectrum file

        timesteps_in_file: dictionary

        nrows: int

        timestep: float


        Returns
        -------
        spectra: dictionary
            - time in days as keys
            - each time contains a dictionary with 
            a wavelength array in cm and a flux density
            array in erg / s / cm^3
        """


        rows_to_skip = np.arange(timesteps_in_file[timestep.value])

        # Check that units are appropriate
        wl_units.to(units.angstrom)
        fd_units.to(units.erg / units.s / units.cm**2 / units.angstrom)


        spectrum = pd.read_csv(filename, skiprows=rows_to_skip,
            names=['wavelength_low', 'wavelength_high', 'flux_density'],
            usecols=[0, 1, angle+2], nrows=nrows,
            delim_whitespace=True, dtype='float')

        # Add column for average wavelength in bin
        spectrum['wavelength_mid'] = 0.5 * (
            spectrum['wavelength_low'] + spectrum['wavelength_high'])

        # Remove all points where the spectrum is zero
        if remove_zero:
            spectrum = spectrum.drop(spectrum[spectrum.flux_density == 0].index)

        # Store wavelengths in cm
        wavelengths = spectrum['wavelength_mid'].values * wl_units

        # Store flux density in erg / s / cm^3
        flux_density_arr = spectrum['flux_density'].values * fd_units

        return observations.Spectrum(timestep,
                wavelengths, flux_density_arr)



    def read_spectra(self, filename, time_units=units.day,
        wl_units=units.cm, angles=[0], remove_zero=False,
        fd_units=(units.erg / units.s / units.cm**2 / units.angstrom)):
        """
        Read in spectra at multiple timesteps
        for Even et al. (2019) and subsequent 
        paper data format.

        Parameters
        ----------
        filename: string
            path to spectrum file

        Returns
        -------
        spectra: dictionary
            - time in days as keys
            - each time contains a dictionary with 
            a wavelength array in cm and a flux density
            array in erg / s / cm^3
        """

        assert os.path.isfile(filename)

        # Check that units are appropriate
        wl_units.to(units.angstrom)
        fd_units.to(units.erg / units.s / units.cm**2 / units.angstrom)


        # Determine time steps in file
        nrows, timesteps_in_file = self.parse_file(
            filename, key='time')

        if len(timesteps_in_file) == 0:
            raise IOError("File not read. Check file type.")

        # Set up properties to collect spectra
        timesteps = np.array(list(timesteps_in_file.keys())) * time_units
        spectra_arr = np.zeros(len(timesteps), dtype=object)

        col_names = ['wavelength_low', 'wavelength_high']
        spectra = {}
        num_angles = len(angles)
        for angle in angles:
            col_names.append(f'spec_angle{angle}')
            spectra[angle] = observations.SpectraOverTime(
                timesteps=timesteps, num_angles=num_angles)

        col_idx = np.concatenate([np.array([0,1]),
            np.asarray(angles) + 2])

        # Read in the spectrum at a given timestep
        for i, time in enumerate(timesteps):
            rows_to_skip = np.arange(timesteps_in_file[time.value])
                        
            spectrum_df = pd.read_csv(filename, 
                skiprows=rows_to_skip, names=col_names, 
                usecols=col_idx, nrows=nrows, 
                delim_whitespace=True, dtype='float')
            # Store each angular bin separately
            for angle in angles:
                col_name = f'spec_angle{angle}'
                spectrum_copy = spectrum_df.copy()

                # Remove all points where the spectrum is zero
                if remove_zero:
                    spectrum_copy = spectrum_copy.drop(
                        spectrum_copy[spectrum_copy[col_name] == 0].index)

                # Compute average wavelength in bin
                wavelengths = 0.5 * (
                    spectrum_copy['wavelength_low'] + \
                    spectrum_copy['wavelength_high']).values * \
                    wl_units

                # Make Spectrum object 
                flux_density_arr = spectrum_copy[col_name].values * \
                    fd_units

                spectra[angle].spectra = np.append(
                    spectra[angle].spectra,
                    observations.Spectrum(time, wavelengths, 
                    flux_density_arr))

        assert timesteps.size > 0
        assert spectra_arr.size > 0

        return spectra 
            


    def read_lightcurve(self, filename, bandname='r-band',
        time_units=units.day, mag_units=units.ABmag, angle_col=2):

        assert os.path.isfile(filename)

        # Determine bands in file
        if bandname == 'bolometric':
            key = 'bolometric'
        else:
            key = 'band'
        nrows, bands_in_file = self.parse_file(
            filename, key=key)

        if len(bands_in_file) == 0:
            raise IOError("File not read. Check file type.")

        # Check that units are appropriate
        time_units.to(units.s)
        if mag_units != units.ABmag and mag_units != units.erg / units.s:
            raise ValueError("Magnitudes must be AB magnitudes or erg/s")

        # Use appropriate bandname
        if bandname not in bands_in_file.keys():
            raise ValueError("Use one of these bandnames: f{bands_in_file.keys()}")

        # Set which rows to read
        rows_to_skip = np.arange(bands_in_file[bandname] + 1)

        # Use appropriate angle_col
        assert angle_col != 1

        # Read in light curve for specific band
        if angle_col == 'all':
            num_angles = 54
            titles = ['time']
            for i in range(num_angles):
                titles.append(f'mag{i}')
            lightcurve_data = pd.read_csv(filename, skiprows=rows_to_skip, sep=' ',
                skipinitialspace=True, usecols=np.arange(1,num_angles+2), names=titles,
                dtype='float', nrows=nrows)

            # FIXME: make a light curve for each magnitude object

        else:
            lightcurve_data = pd.read_csv(filename, skiprows=rows_to_skip, sep=' ',
                skipinitialspace=True, usecols=[1,angle_col], names=['time', 'mag'],
                dtype='float', nrows=nrows)

        times = lightcurve_data['time'].values * time_units
        magnitudes = lightcurve_data['mag'].values * mag_units

        return observations.LightCurve(times, magnitudes)





    def parse_file(self, filename, key='band'):
        """
        Tool for reading data from the Wollaeger et al. (2018)
        and subsequent paper data format. 

        Used to determine the number of rows for a given passband 
        filter or timestep

        Parameters
        ----------
        filename: string
            path to magnitude file

        key: string
            key to search for in file. Options: 'band', 'time'

        Returns
        -------
        nrows: int
            Number of rows between successive appearances of key

        keys_in_file: dictionary
            Dictionary where keys are each occurance of the selected
            keyword (i.e. each bandname or each timestep) and values
            are the line number to start searching for that value in.
        """

        assert key in ['time', 'band', 'bolometric']

        keys_in_file = {}

        # Find out how many rows are in each band
        counter = 0  # Set up a counter
        with open(filename, 'r') as datafile:
            # Read each line until the key appears
            count = 1
            key_count = 0  # Line number where key appears
            line = datafile.readline()
            if key in ['band', 'bolometric']:
                line = datafile.readline()
            while line:
                if key in line:
                    key_count = count
                    if key in ['band', 'bolometric']:
                        keys_in_file[line.split()[1]] = count
                        if key == 'bolometric':
                            key = 'band'
                    elif key == 'time':
                        keys_in_file[float(line.split()[-1])] = count
                line = datafile.readline()

                count += 1
        #if key == 'band':
        #    nrows = count - key_count - 4
        #elif key == 'time':
        nrows = count - key_count - 3

        return nrows, keys_in_file



    def read_magmatrix(self, filenames, bandnames, angles, 
        time_units=units.day, mag_units=units.ABmag, lum=False):

        assert len(filenames) > 0

        # Handle only one filename passed
        if isinstance(filenames, str):
            filenames = [filenames]

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

        # Set search key
        if lum:
            searchkey = 'bolometric'
        else:
            searchkey = 'band'


        # Check that units are appropriate
        time_units.to(units.s)
        if mag_units != units.ABmag:
            raise ValueError("Magnitudes must be AB magnitudes")

        # Set up KN properties dataframe
        knprops = pd.DataFrame(columns=['angle','morph','wind','md','vd','mw','vw'])

        lightcurve_collection = []
        count = 1
        for i, filename in enumerate(filenames):
            # Ensure that file exists
            assert os.path.isfile(filename)

            # Determine bands in file
            nrows, bands_in_file = self.parse_file(
                filename, key=searchkey)

            # Ensure that file is in expected format
            if len(bands_in_file) == 0:
                raise IOError("File not read. Check file type.")

            # Extract properties from filename
            fileprops = self.get_knprops_from_filename(filename)

            # Treat each angle as a separate event
            for angle_col in angles:

                # Report angle in degrees to knprops
                fileprops_angle = fileprops.copy()
                fileprops_angle['angle'] =  np.degrees(
                    np.arccos(1 - angle_col/27))
                #angle_col * 180 / 54
                knprops = knprops.append(fileprops_angle, 
                    ignore_index=True)

                lightcurves_per_band = None
                for j, bandname in enumerate(bandnames):

                    # Read in the magnitudes for the bandname
                    rows_to_skip = np.arange(bands_in_file[bandname] + 1)
                    lightcurve = pd.read_csv(filename, skiprows=rows_to_skip, sep=' ',
                        skipinitialspace=True, usecols=[1, angle_col+2], 
                        names=['time', bandname], dtype='float', nrows=nrows)

                    # Combine magnitudes for each band
                    if lightcurves_per_band is None:
                        lightcurves_per_band = lightcurve
                    else:
                        lightcurves_per_band = pd.merge(lightcurves_per_band, 
                            lightcurve, on='time')

                # Combine magnitudes for each angle
                lightcurve_collection.append(lightcurves_per_band)
                count += 1

        # Combine all magnitudes
        lightcurves = pd.concat(lightcurve_collection, keys=np.arange(count))

        # Store lightcurves as MagMatrix object
        return matrix.MagMatrix(lightcurves.drop(columns=['time']), 
            bandnames=bandnames, times=lightcurves['time'], 
            knprops=knprops, time_units=time_units, mag_units=mag_units)


    def get_knprops_from_filename(self, filename):
        """
        Read the standard LANL filename format.

        Typically this looks something like this:
        'Run_TP_dyn_all_lanth_wind2_all_md0.1_vd0.3_mw0.001_vw0.05_mags_2020-01-04.dat'

        Parameters
        ----------
        filename: str
            string representation of filename
        """

        wind = None
        morph = None
        md = None
        vd = None
        mw = None
        vw = None
        num_comp = 2

        # Reduce filename to last part
        try:
            filename = filename.split('/')[-1]
        except:
            pass

        for info in filename.split('_'):
            # Record morphology
            if morph is None:
                if 'TS' in info:
                    morph = 0
                elif 'TP' in info:
                    morph = 1
                elif 'ST' in info:
                    morph = 2
                elif 'SS' in info:
                    morph = 3
                elif 'SP' in info:
                    morph = 4
                elif 'PS' in info:
                    morph = 5
                elif 'H' in info:
                    morph = 6
                    num_comp = 1
                elif 'P' in info:
                    morph = 7
                    num_comp = 1
                elif 'R' in info and 'Run' not in info:
                    morph = 8
                    num_comp = 1
                elif 'S' in info:
                    morph = 9
                    num_comp = 1
                elif 'T' in info:
                    morph = 10
                    num_comp = 1

            # Record velocity and mass for two component models    
            if num_comp == 2:
                # Record wind
                if 'wind' in info:
                    wind = int(info[-1])
                
                # Record dynamical ejecta mass
                elif 'md' in info:
                    md = float(info[2:])
                    if '.' not in info:
                        if '1' in info:
                            md /= 100
                        else:
                            md /= 1000
               

                # Record dynamical ejecta velocity
                elif 'vd' in info:
                    vd = float(info[2:])
                    if '.' not in info:
                        if '5' in info:
                            vd /= 100
                        else:
                            vd /= 10
                   
                # Record wind ejecta mass
                elif 'mw' in info:
                    mw = float(info[2:])
                    if '.' not in info:
                        if '1' in info:
                            mw /= 100
                        else:
                            mw /= 1000
                    
                # Record wind ejecta velocity
                elif 'vw' in info:
                    vw = float(info[2:])
                    if '.' not in info:
                        if '5' in info:
                            vw /= 100
                        else:
                            vw /= 10
 
            # Record velocity and ejecta mass for single component models
            elif num_comp == 1:
                if 'm' in info and 'v' in info:                
                    mass, vel = info.split('v')
                    md = float(mass[2:])
                    vd = float(vel)

                # Record mass
                elif 'm' == info[0] and info != 'mags':
                    md = float(info[2:])
                    # Recast masses
                    if md in [1, 5]:
                        md /= 100
                    elif md in [2]:
                        md /= 1000

                # Record velocity
                elif 'v' == info[0]:
                    vd = float(info[1:]) / 100

                # Record composition
                elif 'Ye' == info[:2]:
                    wind = float(info[2:]) / 100

        param_values = {
            'morph' : morph,
            'wind' : wind,
            'md' : md,
            'vd' : vd,
            'mw' : mw,
            'vw' : vw,
        }
        knprops = {}
        for prop in ['morph', 'wind', 'md', 'vd', 'mw', 'vw']:
            prop_value = param_values[prop]
            if prop_value is not None:
                knprops[prop] = prop_value
 
        return knprops

class ObservationalFileReader(FileReader):
    """Read observational astronomy data products"""
    def read_spectra(self, filename):

        # Read in observational json
        with open(filename, "r") as read_file:
            data = json.load(read_file)

        spectra = observations.ObservedSpectraCollection()

        for idx in np.arange(50):
            spec = data['GW170817']['spectra'][idx]
            
            
            spec_data = np.asarray(
                data['GW170817']['spectra'][idx]['data']).astype(
                'float')

            spectra.add_spectrum(observations.ObservedSpectrum(
                     obs_time=spec['time'], 
                     wavelengths=spec_data[:,0],
                     fluxes=spec_data[:,1], 
                     u_time=spec['u_time'],
                     u_wavelength=spec['u_wavelengths'],
                     source=spec['source'],
                     u_flux=spec['u_fluxes']))

        return spectra





