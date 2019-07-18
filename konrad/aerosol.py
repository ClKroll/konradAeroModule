import abc
import xarray as xr
import scipy as sc
import numpy as np
import typhon.physics as ty
from sympl import DataArray

#from konrad import constants
from konrad.cloud import get_waveband_data_array


 
   

class Aerosol(metaclass=abc.ABCMeta):
    def __init__(self, aerosol_type='no_aerosol'):#, numlevels):
        a=get_waveband_data_array(0,units='dimensionless', numlevels=200,sw=True)   #called ext_sun in files
        b=get_waveband_data_array(0,units='dimensionless',numlevels=200,sw=True)    #called omega_sun in files
        c=get_waveband_data_array(0,units='dimensionless', numlevels=200,sw=True)         #called g_sun in files
        d=get_waveband_data_array(0,units='dimensionless',numlevels=200,sw=False)     #called ext_earth in files
        self._aerosol_type=aerosol_type
        self.optical_thickness_due_to_aerosol_sw=a.T
        self.single_scattering_albedo_aerosol_sw=b.T
        self.asymmetry_factor_aerosol_sw=c.T
        self.optical_thickness_due_to_aerosol_lw=d.T
        
        

    #################################################################
    #To do: time step updating
    #For now the aerosols are left constant and are not updated
    #add numlevels to init
    #implementation for a changing lapse rate, for now it is implemented only for a fixed lapse rate
    ################################################################
    def update_aerosols(self, time,atmosphere):
        return
    
    def calculateHeightLevels(self,atmosphere):
        return

class VolcanoAerosol(Aerosol):
    def __init__(self):
        super().__init__(aerosol_type='all_aerosol_properties')


    def update_aerosols(self, time,atmosphere):
        if not np.count_nonzero(self.optical_thickness_due_to_aerosol_sw.values):
            extEarth=xr.open_dataset('~/Documents/konrad/konrad/data/aerosolData/zonAverageExtEarthbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc')
            extSun=xr.open_dataset('~/Documents/konrad/konrad/data/aerosolData/zonAverageExtSunbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc')	
            gSun=xr.open_dataset('~/Documents/konrad/konrad/data/aerosolData/zonAveragegSunhbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc')	
            omegaSun=xr.open_dataset('~/Documents/konrad/konrad/data/aerosolData/zonAverageOmegaSunbc_aeropt_cmip6_volc_lw_b16_sw_b14_1992.nc')
            
            heights=self.calculateHeightLevels(atmosphere)
            
            for lw_band in range(np.shape(extEarth.terrestrial_bands)[0]):
                self.optical_thickness_due_to_aerosol_lw[lw_band,:]=sc.interpolate.interp1d(extEarth.altitude.values,extEarth.ext_earth[1,:,lw_band].values,fill_value='extrapolate')(heights)
            for sw_band in range(np.shape(extSun.solar_bands)[0]):
                self.optical_thickness_due_to_aerosol_sw[sw_band,:]=sc.interpolate.interp1d(extSun.altitude.values,extSun.ext_sun[1,:,sw_band],fill_value='extrapolate')(heights)
                self.asymmetry_factor_aerosol_sw[sw_band,:]=sc.interpolate.interp1d(gSun.altitude.values,gSun.g_sun[1,:,sw_band].values,fill_value='extrapolate')(heights)				 					  		                  
                self.single_scattering_albedo_aerosol_sw[sw_band,:]=sc.interpolate.interp1d(omegaSun.altitude.values,omegaSun.omega_sun[1,:,sw_band].values,fill_value='extrapolate')(heights)
                
    def calculateHeightLevels(self,atmosphere):
        heights=ty.pressure2height(atmosphere['plev'],atmosphere['T'][0,:])/1000
        return heights

class NoAerosol(Aerosol):
    def __init__(self):
        super().__init__(aerosol_type='no_aerosol')
        
    def update_aerosols(self, time,atmosphere):
        return
    
    def calculateHeightLevels(self,atmosphere):
        return