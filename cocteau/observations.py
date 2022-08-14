__doc__ = "Store magnitudes, passbands, and spectra series."
__author__ = "Eve Chase <eachase@lanl.gov>"

from astropy import constants, units
try:
    from astropy.cosmology import Planck18_arXiv_v2
except:
    from astropy.cosmology import Planck15

import numpy as np
from scipy.integrate import fixed_quad, quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize, brentq

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

colors = {
    'g-band':'green',
    'r-band':'red',
    'z-band':'magenta',
    'J-band':'orange',
    'K-band':'0.75',
    'i-band':'purple'
}


class Band(object):
    """
    Wavelength-dependent passband filters
    """
    def __init__(self, bandname=None,
        wavelengths=None, transmissions=None):
        """
        Parameters
        ----------
        bandname: string
            Key for filter. Options include None,
            'r-band', 'g-band', 'z-band', 'J-band',
            and 'K-band'.

        wavelengths: array_like
            input wavelengths in cm

        transmissions: array_like
            Wavelength-dependent transmission percentages.
            Values should vary between 0 and 1.
        """
        # Check imports
        if wavelengths is None:
            assert transmissions is None
        elif transmissions is None:
            assert wavelengths is None
        else:
            assert len(wavelengths) == len(transmissions)
                    
        assert type(bandname) == str or bandname == None

        # Store bandname
        self.bandname = bandname


        # Store wavelengths and transmissions
        # sorted by increasing wavelength
        self.wavelength_arr = np.sort(wavelengths)  # cm
        self.transmission_arr = transmissions[np.argsort(wavelengths)]

    def interpolate(self):
        """
        Interpolate a functional form of the transmission
        array as a function of wavelength
     
        Returns
        -------
        band_func: scipy.interpolate.interpolate.interp1d
            functional representation of band
        """

        # Determine bounds of interpolation
        self.min = self.wavelength_arr[0]
        self.max = self.wavelength_arr[-1]
        assert self.max > self.min

        # Interpolate the passband filter as a function of wavelength
        # Requires wavelengths in cm as input
        self.func = interp1d(self.wavelength_arr.cgs, self.transmission_arr,
            bounds_error=False, fill_value=0)
        return self.func


    def plot(self, ax=None, **kwargs):
        """
        Plot band in format similar to Even et al. (2019)

        Returns:
        --------
        ax: Axes object
           contains figure information
        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.wavelength_arr, self.transmission_arr,
            **kwargs)
        ax.set_xlabel('Wavelength (cm)')
        ax.set_ylim([0,1])
        ax.set_ylabel('Transmission Percentage')
        return ax

    def effective_wavelength(self):
        """
        Compute effective wavelength
        """

        # Interpolate band
        self.interpolate()

        # Set bounds of integration in cm -- FIXME this is dumb
        bound_min = self.min.cgs.value
        bound_max = self.max.cgs.value

        # Integrate to find effective wavelength
        _numerator_func = lambda wavelength : \
            wavelength * self.func(wavelength)
        _denominator_func = lambda wavelength : \
            self.func(wavelength)
        numerator = quad(
            _numerator_func, bound_min, bound_max)[0]
        denominator = quad(
            _denominator_func, bound_min, bound_max)[0]
           
        self.wavelength_eff = (numerator / denominator * \
            units.cm).to(units.Angstrom)   

        return self.wavelength_eff



class BandCollector(object):
    """
    Stores multiple Bands.
    """
    def __init__(self, bands={}):
        self.bandnames = np.array(list(bands.keys()))
        self.bands_dict = bands

    def append(self, band):
        self.bands_dict[band.bandname] = band
        self.bandnames = np.array(list(self.bands_dict.keys()))





class SpectraOverTime(object):
    """
    A collection of spectra at successive timesteps
    """
    def __init__(self, timesteps=np.array([]), spectra=np.array([]),
        num_angles=1):
        """
        Parameters:
        -----------
        timesteps: array
            array of timesteps in days

        spectra: array
            array of Spectrum objects, each at corresponding timestep           
        num_angles: int
            number of angular bins. This assumes each
            bin spans equal solid angle.
        """
        # FIXME: this seems like the wrong datastructure here
        self.timesteps = timesteps
        self.spectra = spectra
        self.num_angles = num_angles



class Spectrum(object):
    """
    Spectrum as a function of wavelength
    """
    def __init__(self, timestep=None,
        wavelengths=None, flux_density=None):
        """
        Parameters:
        -----------
        timestep: float
            time in days corresponding to spectrum

        wavelengths: array
            input wavelengths in cm

        flux_density: array
            flux at R=10pc in units of erg / s / cm^3
        """
        # FIXME: assert that wavelengths are sorted
        self.timestep = timestep 
        self.wavelength_arr = wavelengths.cgs.value
        self.flux_density_arr = flux_density.cgs.value



    def interpolate(self):
        """
        Interpolate a functional form of the flux density
        array as a function of wavelength

        Returns
        -------
        spectrum_func: scipy.interpolate.interpolate.interp1d
            functional representation of spectrum
        """

        # Values must be in cgs for interpolation
        return interp1d(self.wavelength_arr.cgs,
            self.flux_density_arr.cgs, bounds_error=False,
            fill_value=0)



    def plot(self, ax=None, **kwargs):
        """
        Plot spectrum in format similar to Even et al. (2019)

        Returns:
        --------
        ax: Axes object
            contains figure information
        """

        fig, ax = plt.subplots()
        ax.plot(self.wavelength_arr * 1e4,
            np.log10(self.flux_density_arr * (4 * np.pi * (10 * 3.08567758e18)**2)),
            **kwargs)

        ax.set_ylabel(r'$\log_{10}$ dL\d$\lambda$  (erg s$^-1 \AA^{-1}$) + const. ')
        ax.set_xlabel('Wavelength (Microns)')
        ax.set_xscale('log')
        ax.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.xaxis.set_tick_params(which='minor', bottom=False)

        return ax


class LightCurve(object):
    """
    Magnitudes over time for a specific passband filter
    """
    def __init__(self, times=None, magnitudes=None,
        spectra=None, band=None, redshift=0):
        """
        Parameters
        ----------
        times: array_like
            input times in days

        magnitudes: array_like
            input AB magnitudes

        spectra: SpectraOverTime
            collection of spectra (flux densities vs.
            wavelengths) for different timesteps

        band: Band
            passband filter

        Notes
        -----
        - times and magnitudes must be of same length
        - either times and magnitudes must be provided 
        or spectra and band must be provided
        - if all four parameters are provided, times and
        magnitudes will be read by default
        """
        self.redshift = redshift

        if times is not None and magnitudes is not None:
            assert len(times) == len(magnitudes)
            self.times = times # days
            self.magnitudes = magnitudes

        elif spectra is not None and band is not None:
            assert isinstance(spectra, SpectraOverTime)
            assert isinstance(band, Band)
            self.compute_lightcurve(spectra, band)

        else:
            raise ValueError("Provide either times and magnitudes or spectra and bands")



    def interpolate(self):
        """
        Interpolate a functional form of the light curve
        array as a function of time

        Returns
        -------
        lightcurve_func: scipy.interpolate.interpolate.interp1d
            functional representation of light curve
        """

        # Values must be in cgs for interpolation
        return interp1d(self.times.to(units.day),
            self.magnitudes, bounds_error=False,
            fill_value=0)


    def compute_lightcurve(self, spectra, band):
        """
        Generate an AB magnitude for each timestep
        and corresponding spectrum, for a given band

        Parameters
        ----------
        spectra: SpectraOverTime
            collection of spectra (flux densities vs.
            wavelengths) for different timesteps

        band: Band
            passband filter
        """
      
        # Determine times from spectra
        self.times = spectra.timesteps

        # For each timestep, compute the magnitude
        self.magnitudes = np.zeros_like(spectra.spectra)
        lc_valid = True

        for i, spectrum in enumerate(spectra.spectra):
            if lc_valid:
                mag = compute_magnitude_at_timestep(
                    spectrum, band, num_angles=spectra.num_angles,
                    redshift=self.redshift).value
                if np.isinf(mag):
                    lc_valid = False
            else:
                mag = np.inf

            self.magnitudes[i] = mag

        self.magnitudes *= units.ABmag



    def get_peaktime(self, magdiff=2, 
        peak_guess=2, return_timescale=False):
        """
        Find the time at which the lightcurve peaks
        and the timescale for the magnitude to dim by
        the amount specified in magdiff

        Parameters
        ----------
        magdiff: float
            The desired difference in magnitude that a
            timescale will be reported at. For example,
            a magdiff of 2 indiciates that we'll search for
            the time it takes for the band to dim by 2 mags.
            Negative numbers correspond to searching for brightening.


        Returns
        -------
        peak_time: float
            Time of peak brightness

        timescale_magdiff: float
            Time required for the magnitude to fall by magdiff 
        """

        # Interpolate the lightcurve
        lc_func = self.interpolate()

        # Find time of peak magnitude
        
        peak_time = minimize(lc_func, peak_guess, 
            method = 'Nelder-Mead').x[0]
        mag_at_peak = lc_func(peak_time)

        # Find the timescale that the magnitude differs by magdiff
        if return_timescale:
            mag2 = lambda time: lc_func(time) - (mag_at_peak + magdiff)
            try:
                time_mag2 = brentq(mag2, peak_time*1.05, peak_time*5)
            except:
                try:
                    time_mag2 = brentq(mag2, peak_time*1.05, peak_time*50)
                except:
                    time_mag2 = peak_time

            # Only return a valid timescale
            if np.isclose(lc_func(time_mag2), mag_at_peak+magdiff):
                timescale_magdiff = time_mag2 - peak_time
            else:
                timescale_magdiff = None

            return peak_time, mag_at_peak, timescale_magdiff
        else:
            return peak_time, mag_at_peak



    def plot(self, bandname='r-band', ax=None, 
        title=None, **kwargs):
        """
        Plot light curve

        Returns
        -------
        ax: Axes object
            contains figure information
        """
        set_figure = False
        if ax is None:
            set_figure = True
            fig, ax = plt.subplots()

        ax.plot(self.times.to(units.day).value, 
            self.magnitudes.to(units.ABmag).value,
            **kwargs)

        if set_figure:
            ax.set_xscale('log')
            ax.set_xlabel('Time (days)')

            ax.set_ylim([-20, 0])
            ax.set_ylabel('AB Mag')
            
            ax.set_title(title)    

            fig.gca().invert_yaxis()
        return ax


class ObservedSpectrum(Spectrum):
    
    def __init__(self, obs_time, wavelengths, 
        fluxes, u_time, u_wavelength, u_flux, 
        redshift=0.009727, merger_time=57982.528524,
        source=None):
        

        # Store source (reference) for observation
        self.reference = source

        # Store redshift
        self.redshift = redshift
        
        # Convert redshift to luminosity distance
        self.dist_lum = Planck18_arXiv_v2.luminosity_distance(redshift).to(units.pc).value
               
        # Set up time
        if u_time == 'MJD':
            self.obs_time = (float(obs_time) - merger_time) * units.day
        else:
            raise ValueError('Time units not supported.')
        self.rest_time = self.obs_time / (1 + self.redshift)
        
        # Set up wavelengths
        if u_wavelength == 'Angstrom':
            wl_units = units.Angstrom
        else:
            raise ValueError('Wavelength units not supported.')
                        
        self.obs_wavelengths = wavelengths * wl_units
        self.rest_wavelengths = self.obs_wavelengths / (1 + self.redshift)
        
        # Set up fluxes
        if u_flux == 'erg/s/cm^2/Angstrom':
            flux_units = units.erg / units.s / units.cm**2 / units.Angstrom
        else:
            raise ValueError('Flux units not supported.')
            
            
        # Restrict flux values to positive numbers
        self.flux_density_arr = fluxes * flux_units * (self.dist_lum / 10)**2
        
        # Set up parameters to match Spectrum class
        self.timestep = self.rest_time
        self.wavelength_arr = self.rest_wavelengths

class ObservedSpectraCollection(object):
    
    def __init__(self):
        
        self.times = np.array([])
        self.spectra = {}
        
    def add_spectrum(self, spectrum):
        
        time = spectrum.rest_time.to(units.day).value
        self.times = np.append(self.times, time)
        self.spectra[time] = spectrum
        







def compute_magnitude_at_timestep(spectrum, band, 
    num_angles=1, redshift=0):
    """
    Compute AB magnitude for a given pass band filter
    and spectrum at a particular time

    Parameters
    ----------
    spectrum: Spectrum
        flux density over wavelength

    band: Band
        passband filter

    num_angles: int
        number of angular bins. This assumes each
        bin spans equal solid angle.

    Returns
    -------
    magnitude: float
        AB magnitude
    """
    # Interpolate the passband filter as a function of wavelength
    band_func = band.interpolate()

    # Interpolate the spectrum as a function of wavelength
    spectrum_func = spectrum.interpolate()

    # Set up functions and parameters for integration
    # Note: Integration must be in cgs units
    c = constants.c.cgs
    redshift_factor = 1 + redshift
    _numerator_func = lambda wavelength : \
        spectrum_func(wavelength / redshift_factor) * band.func(
        wavelength) * wavelength / redshift_factor
    _denominator_func = lambda wavelength : \
        band.func(wavelength) / wavelength

    # Bounds of integration are converted to cm, 
    # then only used in values 
    bound_min = band.min.cgs.value
    bound_max = band.max.cgs.value

    # Integrate to compute the spectral flux density
    denominator = fixed_quad(
            _denominator_func, bound_min, bound_max)[0]
    numerator = fixed_quad(
            _numerator_func, bound_min, bound_max)[0]
 
    #if spectrum.wavelength_arr[0].value > (
    #    bound_min / redshift_factor) or (
    #    spectrum.wavelength_arr[-1].value < (
    #    bound_max / redshift_factor)):
    #    numerator = 0

    assert denominator != 0

    # Reintroduce units
    numerator *= units.erg / units.s / units.cm
    # FIXME: make a merge request in astropy to allow *=
    denominator = denominator * units.dimensionless_unscaled

    # FIXME: supply number of angular bins
    fv = num_angles * numerator / denominator  / c


    # Return a magnitude
    if fv == 0:
        return np.inf * units.ABmag
    else:
        return fv.to(units.ABmag)

