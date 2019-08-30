import os
import abc
import xarray as xr
import scipy as sc
import numpy as np
import typhon.physics as ty
from sympl import DataArray

#from konrad import constants
from konrad.cloud import get_waveband_data_array




class Aerosol(metaclass=abc.ABCMeta):
    def __init__(self,atmNumlevels, aerosol_type='no_aerosol', aerosolLevelShiftInput=0,includeSWForcing=True,includeLWForcing=True,includeScattering=True,includeAbsorption=True):
         
        a = get_waveband_data_array(0, units='dimensionless', numlevels=atmNumlevels, sw=True)   #called ext_sun in files
        b = get_waveband_data_array(0, units='dimensionless', numlevels=atmNumlevels, sw=True)    #called omega_sun in files
        c = get_waveband_data_array(0, units='dimensionless', numlevels=atmNumlevels, sw=True)         #called g_sun in files
        d = get_waveband_data_array(0, units='dimensionless', numlevels=atmNumlevels, sw=False)     #called ext_earth in files
        self._aerosol_type = aerosol_type
        self.includeSWForcing=includeSWForcing
        self.includeLWForcing=includeLWForcing
        self.aerosolLevelShift=aerosolLevelShiftInput
        self.includeScattering=includeScattering
        self.includeAbsorption=includeAbsorption
        self.optical_thickness_due_to_aerosol_sw = a.T
        self.single_scattering_albedo_aerosol_sw = b.T
        self.asymmetry_factor_aerosol_sw = c.T
        self.optical_thickness_due_to_aerosol_lw = d.T

    #################################################################
    #To do: time step updating
    #For now the aerosols are left constant and are not updated
    #add numlevels to init
    #implementation for a changing lapse rate, for now it is implemented only for a fixed lapse rate
    ################################################################
    def update_aerosols(self, time, atmosphere):
        return
    
    def calculateHeightLevels(self, atmosphere):
        return


class VolcanoAerosol(Aerosol):
    def __init__(self,atmNumlevels, aerosolLevelShiftInput=0):
        super().__init__(atmNumlevels,aerosol_type='all_aerosol_properties')
        self.aerosolLevelShift=aerosolLevelShiftInput

    def update_aerosols(self, time, atmosphere):
        if not np.count_nonzero(self.optical_thickness_due_to_aerosol_sw.values):
            if self.includeLWForcing:
                extEarth = xr.open_dataset(
                        os.path.join(
                                os.path.dirname(__file__),
                                'data/aerosolData/23dataextEarth1991.nc'
                                #'data/aerosolData/zonAverageExtEarthbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc'
                                ))
            if self.includeSWForcing:
                extSun = xr.open_dataset(
                        os.path.join(
                                os.path.dirname(__file__),
                                'data/aerosolData/23dataextSun1991.nc'
                                #'data/aerosolData/zonAverageExtSunbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc'
                                ))
                gSun = xr.open_dataset(
                        os.path.join(
                                os.path.dirname(__file__),
                                'data/aerosolData/23datagSun1991.nc'
                                #'data/aerosolData/zonAveragegSunbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc'
                                ))
                omegaSun = xr.open_dataset(
                        os.path.join(
                                os.path.dirname(__file__),
                                'data/aerosolData/23dataomegaSun1991.nc'
                                #'data/aerosolData/zonAverageOmegaSunbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc'
                                ))
            heights = self.calculateHeightLevels(atmosphere)
            #the input data has to be scaled to fit to model levels
            #for compatability with rrtmg input format
            scaling=np.gradient(heights)
            
            if self.aerosolLevelShift:
                self.aerosolLevelShiftArray=self.aerosolLevelShift*np.ones(np.shape(extEarth.altitude[:]))
                if self.includeLWForcing:
                    extEarth.altitude.values=extEarth.altitude.values+self.aerosolLevelShiftArray
                if self.includeSWForcing:
                    extSun.altitude.values=extSun.altitude.values+self.aerosolLevelShiftArray
                    gSun.altitude.values=gSun.altitude.values+self.aerosolLevelShiftArray
                    omegaSun.altitude.values=omegaSun.altitude.values+self.aerosolLevelShiftArray
            
            if self.includeLWForcing:
                for lw_band in range(np.shape(extEarth.terrestrial_bands)[0]):
                    self.optical_thickness_due_to_aerosol_lw[lw_band, :] = \
                        sc.interpolate.interp1d(
                            extEarth.altitude.values,
                            extEarth.ext_earth[8,lw_band, :].values,
                            #extEarth.ext_earth[lw_band, :, 1].values,
                            bounds_error=False,
                            fill_value=0)(heights)*scaling
                        
            if self.includeSWForcing:
                for sw_band in range(np.shape(extSun.solar_bands)[0]):
                    self.optical_thickness_due_to_aerosol_sw[sw_band, :] = \
                        sc.interpolate.interp1d(
                            extSun.altitude.values,
                            extSun.ext_sun[sw_band,8, :].values,
                            #extSun.ext_sun[sw_band, :, 1],
                            bounds_error=False,
                            fill_value=0)(heights)*scaling
                    self.asymmetry_factor_aerosol_sw[sw_band, :] = \
                        sc.interpolate.interp1d(
                            gSun.altitude.values,
                            gSun.g_sun[sw_band,8, :].values,
                            #gSun.g_sun[sw_band, :, 1].values,
                            bounds_error=False,
                            fill_value=0)(heights)
                    self.single_scattering_albedo_aerosol_sw[sw_band, :] = \
                        sc.interpolate.interp1d(
                            omegaSun.altitude.values,
                            omegaSun.omega_sun[sw_band,8, :].values,
                            #omegaSun.omega_sun[sw_band, :, 1].values,
                            bounds_error=False,
                            fill_value=0)(heights)
            
                if not self.includeScattering: '''only absorption'''
                    try:
                        self.optical_thickness_due_to_aerosol_sw= \
                                    np.multiply(self.optical_thickness_due_to_aerosol_sw,
                                                np.subtract(np.ones_like(self.optical_thickness_due_to_aerosol_sw),
                                                            self.single_scattering_albedo_aerosol_sw))
                        self.asymmetry_factor_aerosol_sw=np.zeros_like(self.asymmetry_factor_aerosol_sw)
                        self.single_scattering_albedo_aerosol_sw=np.zeros_like(self.single_scattering_albedo_aerosol_sw)
                        if not self.includeAbsorption:
                            raise ValueError('For aerosols scattering and absorption can not both be deactivated')
                    except (ValueError):
                        exit('Please choose valid input data.')
                        
                if not self.includeAbsorption: '''only scattering'''
                    try:
                        self.optical_thickness_due_to_aerosol_sw= \
                                    np.multiply(self.optical_thickness_due_to_aerosol_sw,
                                                self.single_scattering_albedo_aerosol_sw)
                        self.single_scattering_albedo_aerosol_sw=np.ones_like(self.single_scattering_albedo_aerosol_sw)
                        if not self.includeScattering:
                            raise ValueError('For aerosols scattering and absorption can not both be deactivated')
                    except (ValueError):
                        exit('Please choose valid input data.')
                    

    def calculateHeightLevels(self, atmosphere):
        heights = ty.pressure2height(atmosphere['plev'], atmosphere['T'][0, :])/1000
        return heights


class NoAerosol(Aerosol):
    def __init__(self,atmNumlevels):
        super().__init__(atmNumlevels,aerosol_type='no_aerosol')
        
    def update_aerosols(self, time, atmosphere):
        return
    
    def calculateHeightLevels(self,atmosphere):
        return
