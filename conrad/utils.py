# -*- coding: utf-8 -*-
"""Common utility functions.
"""
import logging

import numpy as np
import typhon
from netCDF4 import Dataset

from conrad import constants


__all__ = [
    'append_timestep_netcdf',
    'create_relative_humidity_profile',
    'ensure_decrease',
    'calculate_halflevel_pressure',
    'append_description',
    'refined_pgrid',
]

logger = logging.getLogger(__name__)


def append_timestep_netcdf(filename, data, timestamp):
    """Append a timestep to an existing netCDF4 file.

    Notes:
        The variables to append to have to exist in the netCDF4 file!

    Parameters:
        filename (str): Path to the netCDF4.
        data (dict{ndarray}): Dict-like object containing the data arrays.
            The key is the variable name and the value is an ``ndarray``, a
            ``pandas.Series`` or an ``xarray.DataArray`` e.g.:
                >>> data = {'T': np.array([290, 295, 300])}

        timestamp (float): Timestamp of values appended.
    """
    # Open netCDF4 file in `append` mode.
    with Dataset(filename, 'a') as nc:
        logging.debug('Append timestep to "{}".'.format(filename))
        t = nc.dimensions['time'].size  # get index to store data.
        nc.variables['time'][t] = timestamp  # append timestamp.

        # Append data for each variable in ``data``.
        for var in data:
            # Append variable if it has a `time` dimension and is no
            # dimension itself.
            if 'time' in nc[var].dimensions and var not in nc.dimensions:
                # TODO: Find a cleaner way to handle different data dimensions.
                if 'plev' in nc[var].dimensions:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t, :] = data[var].values
                    else:
                        nc.variables[var][t, :] = data[var]
                else:
                    if hasattr(data[var], 'values'):
                        nc.variables[var][t] = data[var].values
                    else:
                        nc.variables[var][t] = data[var]


def create_relative_humidity_profile(p, rh_s=0.75):
    """Create an exponential relative humidity profile.

    Parameters:
        p (ndarray): Pressure.
        rh_s (float): Relative humidity at first pressure level.

    Returns:
        ndarray: Relative humidtiy."""
    rh = rh_s / (np.exp(1) - 1) * (np.exp(p / p[0]) - 1)
    return np.round(rh, decimals=4)


def ensure_decrease(array):
    """Ensure that a given array is decreasing.

    Parameters:
        array (ndarray): Input array.

    Returns:
        ndarray: Monotonously decreasing array.
    """
    for i in range(1, len(array)):
        if array[i] > array[i-1]:
            array[i] = array[i-1]
    return array


def calculate_halflevel_pressure(fulllevels):
    """Returns the linear interpolated halflevels for given array.

    Parameters:
        fulllevels (ndarray): Pressure at fullevels.

    Returns:
        ndarray: Coordinates at halflevel.

    """
    inter = (fulllevels[1:] + fulllevels[:-1]) / 2
    bottom = fulllevels[0] - 0.5 * (fulllevels[1] - fulllevels[0])
    top = 0
    return np.hstack((bottom, inter, top))


def append_description(dataset, description=None):
    """Append variable attributes to a given dataset.

    Parameters:
          dataset (xarray.Dataset): Dataset including variables to describe.
          description (dict): Dictionary containing variable descriptions.
            The keys are the variable keys used in the Dataset.
            The values are dictionaries themselves containing attributes
            and their names as keys, e.g.:
                desc = {'T': {'units': 'K', 'standard_name': 'temperature'}}
    """
    if description is None:
        description = constants.variable_description

    for key in dataset.keys():
        if key in description:
            dataset[key].attrs = constants.variable_description[key]


def refined_pgrid(start, stop, num=200, threshold=100e2):
    """Create a pressure grid with two spacing regimes.

    This functions creates a pressure grid with two different spacings. The
    bottom part (troposphere) uses a linear spacing. Above a given threshold,
    a logarithmic spacing is used. This allows for higher sampling close to the
    surface while still covering hihger altitudes.

    Parameters:
        start (float): Pressure at lowest atmosphere layer.
        stop (float): Pressure at highest atmosphere layer.
        num (int): Number of pressure layers.
        threshold (float): Pressure layer at which to swich spacing.

    Returns:
        ndarray: Pressure grid.
    """
    # Create a linear spaced pressure grid for the lower atmosphere.
    p_tropo = np.linspace(start, threshold, np.ceil(num / 2), endpoint=False)

    # Above a given threshold, use a logarithmic spacing to create the grid.
    p_strato = typhon.math.nlogspace(threshold, stop, np.floor(num / 2))

    return np.hstack([p_tropo, p_strato])