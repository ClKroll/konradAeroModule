# -*- coding: utf-8 -*-
"""Module containing classes describing different atmosphere models.
"""

__all__ = [
    'Atmosphere',
    'AtmosphereFixedVMR',
    'AtmosphereFixedRH',
    'AtmosphereConvective',
    'AtmosphereMoistConvective',
    'AtmosphereConUp',
    'AtmosphereConvectiveFlux',
]


import abc
import collections
import logging

import typhon
import netCDF4
import numpy as np
from scipy.interpolate import interp1d
from xarray import Dataset, DataArray

import conrad
from conrad import constants
from conrad import utils

logger = logging.getLogger()

atmosphere_variables = [
    'T',
    'H2O',
    'N2O',
    'O3',
    'CO2',
    'CO',
    'CH4',
]


class Atmosphere(Dataset, metaclass=abc.ABCMeta):
    """Abstract base class to define requirements for atmosphere models."""
    @abc.abstractmethod
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust atmosphere according to given heatingrate."""

    @classmethod
    def from_atm_fields_compact(cls, atmfield, **kwargs):
        """Convert an ARTS atm_fields_compact [0] into an atmosphere.

        [0] http://arts.mi.uni-hamburg.de/docserver-trunk/variables/atm_fields_compact

        Parameters:
            atmfield (typhon.arts.types.GriddedField4): A compact set of
                atmospheric fields.
        """
        # Create a Dataset with time and pressure dimension.
        plev = atmfield.grids[1]
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            # Get ARTS variable name from variable description.
            arts_key = constants.variable_description[var].get('arts_name')

            # Extract profile from atm_fields_compact
            profile = typhon.arts.atm_fields_compact_get(
                [arts_key], atmfield).squeeze()

            d[var] = DataArray(profile[np.newaxis, :], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_xml(cls, xmlfile, **kwargs):
        """Read atmosphere from XML file containing an ARTS atm_fields_compact.

        Parameters:
            xmlfile (str): Path to XML file.
        """
        # Read the content of given XML file.
        griddedfield = typhon.arts.xml.load(xmlfile)

        # Check if the XML file contains an atm_fields_compact (GriddedField4).
        arts_type = typhon.arts.utils.get_arts_typename(griddedfield)
        if arts_type != 'GriddedField4':
            raise TypeError(
                f'XML file does not contain "GriddedField4" but "{arts_type}".'
            )

        return cls.from_atm_fields_compact(griddedfield, **kwargs)

    @classmethod
    def from_dict(cls, dictionary, **kwargs):
        """Create an atmosphere model from dictionary values.

        Parameters:
            dictionary (dict): Dictionary containing ndarrays.
        """
        # TODO: Currently working for good-natured dictionaries.
        # Consider allowing a more flexibel user interface.

        # Create a Dataset with time and pressure dimension.
        plev = dictionary['plev']
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(dictionary[var], dims=('time', 'plev',))

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    @classmethod
    def from_netcdf(cls, ncfile, timestep=-1, **kwargs):
        """Create an atmosphere model from a netCDF file.

        Parameters:
            ncfile (str): Path to netCDF file.
            timestep (int): Timestep to read (default is last timestep).
        """
        data = netCDF4.Dataset(ncfile).variables

        # Create a Dataset with time and pressure dimension.
        plev = data['plev'][:]
        phlev = utils.calculate_halflevel_pressure(plev)
        d = cls(coords={'plev': plev,  # pressure level
                        'time': [0],  # time dimension
                        'phlev': phlev,  # pressure at halflevels
                        },
                **kwargs,
                )

        for var in atmosphere_variables:
            d[var] = DataArray(
                data=data[var][[timestep], :],
                dims=('time', 'plev',)
            )

        # Calculate the geopotential height.
        d.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(d)

        return d

    def to_atm_fields_compact(self):
        """Convert an atmosphere into an ARTS atm_fields_compact."""
        # Store all atmosphere variables including geopotential height.
        variables = atmosphere_variables + ['z']

        # Get ARTS variable name from variable description.
        species = [constants.variable_description[var].get('arts_name')
                   for var in variables]

        # Create a GriddedField4.
        atmfield = typhon.arts.types.GriddedField4()

        # Set grids and their names.
        atmfield.gridnames = ['Species', 'Pressure', 'Longitude', 'Latitude']
        atmfield.grids = [
            species, self['plev'].values, np.array([]), np.array([])
        ]

        # The profiles have to be passed in "stacked" form, as an ndarray of
        # dimensions [species, pressure, lat, lon].
        atmfield.data = np.vstack(
            [self[var].values.reshape(1, self['plev'].size, 1, 1)
             for var in variables]
        )
        atmfield.dataname = 'Data'

        # Perform a consistency check of the passed grids and data tensor.
        atmfield.check_dimension()

        return atmfield

    def refine_plev(self, pgrid, axis=1, **kwargs):
        """Refine the pressure grid of an atmosphere object.

        Note:
              This method returns a **new** object,
              the original object is maintained!

        Parameters:
              pgrid (ndarray): New pressure grid [Pa].
              axis (int): Index of pressure axis (should be 1).
                This keyword is only there for possible changes in future.
            **kwargs: Additional keyword arguments are collected
                and passed to :func:`scipy.interpolate.interp1d`

        Returns:
              Atmosphere: A **new** atmosphere object.
        """
        # Initialize an empty directory to fill it with interpolated data.
        # The dictionary is later used to create a new object using the
        # Atmosphere.from_dict() classmethod. This allows to circumvent the
        # fixed dimension size in xarray.DataArrays.
        datadict = dict()

        datadict['plev'] = pgrid  # Store new pressure grid.
        # Loop over all atmospheric variables...
        for variable in atmosphere_variables:
            # and create an interpolation function using the original data.
            f = interp1d(self['plev'].values, self[variable],
                         axis=axis, **kwargs)

            # Store the interpolated new data in the data directory.
            datadict[variable] = DataArray(f(pgrid), dims=('time', 'plev'))

        # Create a new atmosphere object from the filled data directory.
        new_atmosphere = type(self).from_dict(datadict)

        # Calculate the geopotential height.
        new_atmosphere.calculate_height()

        # Append variable descriptions to the Dataset.
        utils.append_description(new_atmosphere)

        return new_atmosphere

    # TODO: This function could handle the nasty time dimension in the future.
    # Allowing to set two-dimensional variables using a 1d-array, if one
    # coordinate has the dimension one.
    def set(self, variable, value):
        """Set the values of a variable.

        Parameters:
            variable (str): Variable key.
            value (float or ndarray): Value to assign to the variable.
                If a float is given, all values are filled with it.
        """
        if isinstance(value, collections.Container):
            self[variable].values[0, :] = value
        else:
            self[variable].values.fill(value)

    def get_values(self, variable, keepdims=True):
        """Get values of a given variable.

        Parameters:
            variable (str): Variable key.
            keepdims (bool): If this is set to False, single-dimensions are
                removed. Otherwise dimensions are keppt (default).

        Returns:
            ndarray: Array containing the values assigned to the variable.
        """
        if keepdims:
            return self[variable].values
        else:
            return self[variable].values.ravel()

    def calculate_height(self):
        """Calculate the geopotential height."""
        g = constants.earth_standard_gravity

        plev = self['plev'].values  # Air pressure at full-levels.
        phlev = self['phlev'].values  # Air pressure at half-levels.
        T = self['T'].values  # Air temperature at full-levels.

        # Calculate the air density from current atmospheric state.
        rho = typhon.physics.density(plev, T)

        # Use the hydrostatic equation to calculate geopotential height from
        # given pressure, density and gravity.
        z = np.cumsum(-np.diff(phlev) / (rho * g))

        # If height is already in Dataset, update its values.
        if 'z' in self:
            self['z'].values[0, :] = np.cumsum(-np.diff(phlev) / (rho * g))
        # Otherwise create the DataArray.
        else:
            self['z'] = DataArray(z[np.newaxis, :], dims=('time', 'plev'))

    @property
    def relative_humidity(self):
        """Return the relative humidity of the current atmospheric state."""
        vmr, p, T = self['H2O'], self['plev'], self['T']
        return typhon.atmosphere.relative_humidity(vmr, p, T)

    @relative_humidity.setter
    def relative_humidity(self, RH):
        """Set the water vapor mixing ratio to match given relative humidity.

        Parameters:
            RH (ndarray or float): Relative humidity.
        """
        logger.debug('Adjust VMR to preserve relative humidity.')
        self['H2O'].values = typhon.atmosphere.vmr(RH, self['plev'], self['T'])

    def get_lapse_rates(self):
        """Calculate the temperature lapse rate at each level."""
        lapse_rate = np.diff(self['T'][0, :]) / np.diff(self['z'][0, :])
        lapse_rate = typhon.math.interpolate_halflevels(lapse_rate)
        lapse_rate = np.append(lapse_rate[0], lapse_rate)
        return np.append(lapse_rate, lapse_rate[-1])

    def get_potential_temperature(self, p0=1000e2):
        """Calculate the potential temperature.

        .. math::
            \theta = T \cdot \left(\frac{p_0}{P}\right)^\frac{2}{7}

        Parameters:
              p0 (float): Pressure at reference level [Pa].

        Returns:
              ndarray: Potential temperature [K].
        """
        # Get view on temperature and pressure arrays.
        T = self['T'].values[0, :]
        p = self['plev'].values

        # Calculate the potential temperature.
        return T * (p0 / p) ** (2 / 7)

    def get_static_stability(self):
        """Calculate the static stability.

        .. math::
            \sigma = - \frac{T}{\Theta} \frac{\partial\Theta}{\partial p}

        Returns:
              ndarray: Static stability [K/Pa].
        """
        # Get view on temperature and pressure arrays.
        t = self['T'].values[0, :]
        p = self['plev'].values

        # Calculate potential temperature and its vertical derivative.
        theta =  self.get_potential_temperature()
        dtheta = np.diff(theta) / np.diff(p)

        return -(t / theta)[:-1] * dtheta

    def get_diabatic_subsidence(self, radiative_cooling):
        """Calculate the diabatic subsidence.

        Parameters:
              radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!

        Returns:
            ndarray: Diabatic subsidence [Pa/day].
        """
        sigma = self.get_static_stability()

        return -radiative_cooling[:-1] / sigma

    def get_subsidence_convergence_max_index(self, radiative_cooling,
                                             pmin=10e2):
        """Return index of maximum subsidence convergence.

        Parameters:
            radiative_cooling (ndarray): Radiative cooling rates.
                Positive values for heating, negative values for cooling!
            pmin (float): Lower pressure threshold. The cold point has to
                be below (higher pressure, lower height) that value.

        Returns:
              int: Layer index.
        """
        plev = self['plev'].values
        omega = self.get_diabatic_subsidence(radiative_cooling)
        domega = np.diff(omega) / np.diff(plev[:-1])

        return np.argmax(domega[plev[:-2]>pmin])

    def adjust_relative_humidity(self, heatingrates, rh_surface=0.8,
                                 rh_tropo=0.3):
        """Adjust the relative humidity according to the vertical structure."""
        # TODO: Humidity values should be attributes of Atmosphere objects.

        # Find the level (index) of maximum diabatic subsidence convergence.
        # This level is associated with a second peak in the relative humidity.
        scm = self.get_subsidence_convergence_max_index(heatingrates[0, :])
        self['convergence_max'] = DataArray([scm], dims=('time',))

        # Determine relative humidity profile with second maximum at the
        # level of maximum subsidence convergence.
        plev = self['plev'].values
        rh = utils.create_relative_humidity_profile(
            plev=plev,
            rh_surface=rh_surface,
            rh_tropo=rh_tropo,
            p_tropo=plev[scm],
        )

        # Overwrite the current humidity profile with the new values.
        self.relative_humidity = rh

    @property
    def cold_point_index(self, pmin=1e2):
        """Return the pressure index of the cold point tropopause.

        Parameters:
              pmin (float): Lower pressure threshold. The cold point has to
              be below (higher pressure, lower height) that value.

        Returns:
            int: Layer index.
        """
        return int(np.argmin(self['T'][:, self['plev'] > pmin]))

    def apply_H2O_limits(self, vmr_max=1.):
        """Adjust water vapor VMR values to follow physical limitations.

        Parameters:
            vmr_max (float): Maximum limit for water vapor VMR.
        """
        # Keep water vapor VMR values above the cold point tropopause constant.
        i = self.cold_point_index
        self['H2O'].values[0, i:] = self['H2O'][0, i]

        # NOTE: This has currently no effect, as the vmr_max is set to 1.
        # Limit the water vapor mixing ratios to a given threshold.
        too_high = self['H2O'].values > vmr_max
        self['H2O'].values[too_high] = vmr_max


class AtmosphereFixedVMR(Atmosphere):
    """Atmosphere model with fixed volume mixing ratio."""
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature.

        Adjust the atmospheric temperature profile by simply adding the given
        heatingrates.

        Parameters:
            heatingrates (float or ndarray): Heatingrate [K /day].
            timestep (float): Width of a timestep [day].
        """
        # Apply heatingrates to temperature profile.
        self['T'] += heatingrate * timestep

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereFixedRH(Atmosphere):
    """Atmosphere model with fixed relative humidity.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio.

    Parameters:
        heatingrates (float or ndarray): Heatingrate [K /day].
        timestep (float): Width of a timestep [day].
    """
    def adjust(self, heatingrate, timestep, **kwargs):
        """Adjust the temperature and preserve relative humidity.

        Parameters:
            heatingrate (float or ndarray):
                Heatingrate (already scaled with timestep) [K].
        """
        # Apply heatingrates to temperature profile.
        self['T'] += heatingrate * timestep

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


def energy_difference(T_2, T_1, sst_2, sst_1, dp, eff_Cp_s):
    """
    Calculate the energy difference between two atmospheric profiles (2 - 1).

    Parameters:
        T_2: atmospheric temperature profile (2)
        T_1: atmospheric temperature profile (1)
        sst_2: surface temperature (2)
        sst_1: surface temperature (1)
        dp: pressure thicknesses of levels,
            must be the same for both atmospheric profiles
        eff_Cp_s: effective heat capacity of surface
    """
    Cp = constants.isobaric_mass_heat_capacity
    g = constants.g

    dT = T_2 - T_1  # convective temperature change of atmosphere
    dT_s = sst_2 - sst_1  # of surface
    termdiff = - np.sum(Cp/g * dT * dp) + eff_Cp_s * dT_s

    return termdiff

class AtmosphereConvective2(Atmosphere):
    """Atmosphere model with preserved RH and fixed temperature lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a simple
    convection parameterization is used.
    """
    def __init__(self, *args, lapse=0.0065, **kwargs):

        super().__init__(*args, **kwargs)
        if isinstance(lapse, float):
            # make an array of lapse rate values, corresponding to the half
            # pressure levels
            lapse_array = lapse * np.ones((1, self['phlev'].size))
            self['lapse'] = DataArray(lapse_array, dims=('time', 'phlev'))
        elif isinstance(lapse, np.ndarray):
            # Here the input lapse rate is given on the full pressure levels,
            # we need to convert it, so that it is on the half levels.
            lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
            self['lapse'] = DataArray(lapse_phlev, dims=('time', 'phlev'))

        utils.append_description(self)  # Append variable descriptions.

    def save_profile(self, surface, T_con, surfaceT):
        """
        Update the surface and atmospheric temperatures to surfaceT and T_con.

        Parameters:
            surfaceT: float
            T_con: ndarray
        """
        surface['temperature'][0] = surfaceT
        self['T'].values = T_con[np.newaxis, :]

    def convective_adjustment(self, surface, timestep=0.1):
        """
        Find the energy-conserving temperature profile using a iterative
        procedure with test profiles. Update the atmospheric temperature
        profile to this one.

        Parameters:
            timestep: float, only required for slow convection
        """
        near_zero = 0.00001
        T_rad = self['T'][0, :]
        p = self['plev']
        lapse = self.lapse[0, :]
        density1 = typhon.physics.density(p, T_rad)
        density = utils.calculate_halflevel_pressure(density1.values)

        g = constants.g
        lp = -lapse[:].values / (g*density)

        # find energy difference if there is no change to surface temp due to
        # convective adjustment. in this case the new profile should be
        # associated with an increase in energy in the atmosphere.
        surfaceTpos = surface.temperature.values
        T_con, diffpos = self.test_profile(surface, surfaceTpos, lp,
                                           timestep=timestep)
        # this is the temperature profile required if we have a set-up with a
        # fixed surface temperature, then the energy does not matter.
        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self['T'].values = T_con.values[np.newaxis, :]
        # for other cases, if we find a decrease or approx no change in energy,
        # the atmosphere is not being warmed by the convection,
        # as it is not unstable to convection, so no adjustment is applied
        if diffpos < near_zero:
            return None

        # if the atmosphere is unstable to convection, a fixed surface temp
        # produces an increase in energy (as convection warms the atmosphere).
        # this surface temperature is an upper bound to the energy-conserving
        # surface temperature.
        # now we reduce surface temperature until we find an adjusted profile
        # that is associated with an energy loss.
        surfaceTneg = surfaceTpos - 1
        T_con, diffneg = self.test_profile(surface, surfaceTneg, lp,
                                           timestep=timestep)
        # good guess for energy-conserving profile
        if np.abs(diffneg) < near_zero:
            self.save_profile(surface, T_con, surfaceTneg)
            return None
        # if surfaceTneg = surfaceTpos - 1 is not negative enough to produce an
        # energy loss, keep reducing surfaceTneg to find a lower bound for the
        # uncertainty range of the energy-conserving surface temperature
        while diffneg > 0:
            diffpos = diffneg
            # update surfaceTpos to narrow the uncertainty range
            surfaceTpos = surfaceTneg
            surfaceTneg -= 1
            T_con, diffneg = self.test_profile(surface, surfaceTneg, lp,
                                               timestep=timestep)
            # again for the case that this surface temperature happens to
            # be a good guess (sufficiently close to energy conserving)
            if np.abs(diffneg) < near_zero:
                self.save_profile(surface, T_con, surfaceTneg)
                return None

        # Now we have a upper and lower bound for the surface temperature of
        # the energy conserving profile. Iterate to get closer to the energy-
        # conserving temperature profile.
        counter = 0
        while diffpos >= near_zero and -diffneg >= near_zero:
            surfaceT = surfaceTneg + (surfaceTpos - surfaceTneg) * (-diffneg) / (-diffneg + diffpos)
            T_con, diff = self.test_profile(surface, surfaceT, lp,
                                            timestep=timestep)
            if diff > 0:
                diffpos = diff
                surfaceTpos = surfaceT
            else:
                diffneg = diff
                surfaceTneg = surfaceT
            # to avoid getting stuck in a loop if something weird is going on
            counter += 1
            if counter == 100:
                raise ValueError(
                        "No energy conserving convective profile can be found"
                        )

        # save new temperature profile
        self.save_profile(surface, T_con, surfaceT)

    def test_profile(self, surface, surfaceT, lp, timestep=0.1):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, following the specified lapse rate (lp) for the region where
        the convectively adjusted atmosphere is warmer than the radiative one.

        Parameters:
            surfaceT: float, surface temperature of the new profile
            lp: lapse rate in K/Pa
            timestep: float, not required in this case

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        T_rad = self['T'][0, :]
        p = self['plev']
        phlev = self['phlev']
        # dp, thicknesses of atmosphere layers, for energy calculation
        dp = np.diff(phlev)
        # for lapse rate integral
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        T_con = surfaceT - np.cumsum(dp_lapse * lp[:-1])
        if np.any(T_con > T_rad):
            contop = np.max(np.where(T_con > T_rad))
            T_con[contop+1:] = T_rad[contop+1:]
        else:
            T_con = T_rad
            contop = 0
        self['convective_top'] = DataArray([contop], dims=('time',))

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff)

    def adjust(self, heatingrates, timestep, surface):

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()

class AtmosphereSlowConvective(AtmosphereConvective2):
    """
    Atmosphere with a time dependent convective adjustment.

    Here the convective adjustment occurs throughout the whole atmosphere, but
    tau (the convective timescale) should be chosen to be very large in the
    middle and upper atmosphere.
    """
    def __init__(self, *args, tau=0, **kwargs):
        super().__init__(*args, **kwargs)

        self['convective_tau'] = DataArray(tau[np.newaxis, :], dims=('time', 'plev'))

        utils.append_description(self)

    def test_profile(self, surface, surfaceT, lp, timestep):
        """
        Assuming a particular surface temperature (surfaceT), create a new
        profile, using the convective timescale and specified lapse rate (lp).

        Parameters:
            surfaceT: float, surface temperature of the new profile
            lp: lapse rate in K/Pa
            timestep: float, timestep of simulation

        Returns:
            ndarray: new atmospheric temperature profile
            float: energy difference between the new profile and the old one
        """
        T_rad = self['T'][0, :]
        p = self['plev']
        phlev = self['phlev']
        dp = np.diff(phlev)
        dp_lapse = np.hstack((np.array([p[0] - phlev[0]]), np.diff(p)))

        tau = self['convective_tau'][0]
        tf = 1 - np.exp(-timestep/tau)
        T_con = T_rad*(1 - tf) + tf*(surfaceT - np.cumsum(dp_lapse * lp[:-1]))
        T_con = T_con.values

        eff_Cp_s = surface.rho * surface.cp * surface.dz

        diff = energy_difference(T_con, T_rad, surfaceT, surface.temperature,
                                 dp, eff_Cp_s)

        return T_con, float(diff.values)

    def adjust(self, heatingrates, timestep, surface):

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep=timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()

class AtmosphereMoistConvective2(AtmosphereConvective2):
    """Atmosphere model with preserved RH and a temperature and humidity
    -dependent lapse rate.

    This atmosphere model preserves the initial relative humidity profile by
    adjusting the water vapor volume mixing ratio. In addition, a convection
    parameterization is used, which sets the lapse rate to the moist adiabat,
    calculated from the previous temperature and humidity profiles.
    """
    def moistlapse(self):
        """Updates the atmospheric lapse rate for the convective adjustment
        according to the moist adiabat, which is calculated from the
        atmospheric temperature and humidity profiles. The lapse rate is in
        units of K/km.
        """
        g = constants.g
        Lv = 2501000
        R = 287
        eps = 0.62197
        Cp = constants.isobaric_mass_heat_capacity
        VMR = self['H2O'][0, :]
        T = self['T'][0, :]
        lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
        lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
        self['lapse'][0] = lapse_phlev

    def adjust(self, heatingrates, timestep, surface):

        self.moistlapse()

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()

class AtmosphereSlowMoistConvective(AtmosphereSlowConvective):

    def moistlapse(self):
        """Updates the atmospheric lapse rate for the convective adjustment
        according to the moist adiabat, which is calculated from the
        atmospheric temperature and humidity profiles. The lapse rate is in
        units of K/km.
        Parameters:
            a: atmosphere
        """
        g = constants.g
        Lv = 2501000
        R = 287
        eps = 0.62197
        Cp = constants.isobaric_mass_heat_capacity
        VMR = self['H2O'][0, :]
        T = self['T'][0, :]
        lapse = g*(1 + Lv*VMR/R/T)/(Cp + Lv**2*VMR*eps/R/T**2)
        lapse_phlev = utils.calculate_halflevel_pressure(lapse.values)
        self['lapse'][0] = lapse_phlev

    def adjust(self, heatingrates, timestep, surface):

        self.moistlapse()

        # Apply heatingrates to temperature profile.
        self['T'] += heatingrates * timestep

        # Apply convective adjustment
        self.convective_adjustment(surface, timestep)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()



class AtmosphereConUp(AtmosphereConvective2):
    """
    Requires testing. Do not use.

    Atmosphere model with preserved RH and fixed temperature lapse rate,
    that includes a cooling term due to upwelling in the statosphere.
    """
    def upwelling_adjustment(self, ctop, timestep, w=0.0005):
        """Stratospheric cooling term parameterizing large-scale upwelling.

        Parameters:
            ctop (float): array index,
                the bottom level for the upwelling
                at and above this level, the upwelling is constant with height
            w (float): upwelling velocity
        """
        Cp = constants.isobaric_mass_heat_capacity
        g = constants.earth_standard_gravity

        actuallapse = self.get_lapse_rates()

        Q = -w * (-actuallapse + g / Cp)  # per second
        Q *= 24 * 60 * 60  # per day
        Q[:ctop] = 0

        self['T'] += Q * timestep

    def adjust(self, heatingrates, timestep, surface, w=0.0001, **kwargs):
        # TODO: Wrtie docstring.
        self['T'] += heatingrates * timestep

        if isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            # TODO: Output convective top for fixed_surface_temperature case
            ct = self.convective_adjustment_fixed_surface_temperature(surface=surface)
        else:
            ct, tdn, tdp = self.convective_top(surface=surface,
                                               timestep=timestep)

        self.upwelling_adjustment(ct, timestep, w)

        if not isinstance(surface, conrad.surface.SurfaceFixedTemperature):
            self.convective_adjustment(
                    ct, tdn, tdp,
                    surface=surface,
                    timestep=timestep
                )

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()


class AtmosphereConvectiveFlux(Atmosphere):
    """Convective flux."""
    def adjust(self, heatingrates, timestep, surface, w=0.0001, **kwargs):
        self['T'] += heatingrates * timestep

        Cp = conrad.constants.Cp
        p = self['plev'].values
        T = self['T'].values
        z = self['z'].values

        critical_lapse_rate = self.lapse[0, 1:-1]
        w = 0.01

        lapse_rate = -np.diff(T[0, :]) / np.diff(z[0, :])

        flux_divergence = w * (lapse_rate - critical_lapse_rate)
        dT = flux_divergence * timestep * 24 * 3600
        self['T'].values[0, :-1] += dT.values

        print(dT.values)

        # Preserve the initial relative humidity profile.
        self.relative_humidity = self['initial_rel_humid'].values

        # Adjust stratospheric VMR values.
        self.apply_H2O_limits()

        # Calculate the geopotential height field.
        self.calculate_height()
