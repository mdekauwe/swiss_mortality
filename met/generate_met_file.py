#!/usr/bin/env python

"""
Generate NC file for CABLE

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.09.2020)"
__email__ = "mdekauwe@gmail.com"

import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt

def main(lat, lon, df):

    out_fname = "swiss_met.nc"

    ndim = 1
    n_timesteps = len(df)
    times = []
    secs = 0.0
    for i in range(n_timesteps):
        times.append(secs)
        secs += 3600.

    # create file and write global attributes
    f = nc.Dataset(out_fname, 'w', format='NETCDF4')
    f.description = 'Swiss met data, created by Martin De Kauwe'
    f.history = "Created by: %s" % (os.path.basename(__file__))
    f.creation_date = "%s" % (datetime.datetime.now())
    f.contact = "mdekauwe@gmail.com"

    # set dimensions
    f.createDimension('time', None)
    f.createDimension('z', ndim)
    f.createDimension('y', ndim)
    f.createDimension('x', ndim)
    #f.Conventions = "CF-1.0"

    # create variables
    time = f.createVariable('time', 'f8', ('time',))
    time.units = "seconds since %s 00:00:00" % (df.index[0])
    time.long_name = "time"
    time.calendar = "standard"

    z = f.createVariable('z', 'f8', ('z',))
    z.long_name = "z"
    z.long_name = "z dimension"

    y = f.createVariable('y', 'f8', ('y',))
    y.long_name = "y"
    y.long_name = "y dimension"

    x = f.createVariable('x', 'f8', ('x',))
    x.long_name = "x"
    x.long_name = "x dimension"

    latitude = f.createVariable('latitude', 'f8', ('y', 'x',))
    latitude.units = "degrees_north"
    latitude.missing_value = -9999.
    latitude.long_name = "Latitude"

    longitude = f.createVariable('longitude', 'f8', ('y', 'x',))
    longitude.units = "degrees_east"
    longitude.missing_value = -9999.
    longitude.long_name = "Longitude"

    SWdown = f.createVariable('SWdown', 'f8', ('time', 'y', 'x',))
    SWdown.units = "W/m^2"
    SWdown.missing_value = -9999.
    SWdown.long_name = "Surface incident shortwave radiation"
    SWdown.CF_name = "surface_downwelling_shortwave_flux_in_air"

    Tair = f.createVariable('Tair', 'f8', ('time', 'z', 'y', 'x',))
    Tair.units = "K"
    Tair.missing_value = -9999.
    Tair.long_name = "Near surface air temperature"
    Tair.CF_name = "surface_temperature"

    Rainf = f.createVariable('Rainf', 'f8', ('time', 'y', 'x',))
    Rainf.units = "mm/s"
    Rainf.missing_value = -9999.
    Rainf.long_name = "Rainfall rate"
    Rainf.CF_name = "precipitation_flux"

    Qair = f.createVariable('Qair', 'f8', ('time', 'z', 'y', 'x',))
    Qair.units = "kg/kg"
    Qair.missing_value = -9999.
    Qair.long_name = "Near surface specific humidity"
    Qair.CF_name = "surface_specific_humidity"

    Wind = f.createVariable('Wind', 'f8', ('time', 'z', 'y', 'x',))
    Wind.units = "m/s"
    Wind.missing_value = -9999.
    Wind.long_name = "Scalar windspeed" ;
    Wind.CF_name = "wind_speed"

    PSurf = f.createVariable('PSurf', 'f8', ('time', 'y', 'x',))
    PSurf.units = "Pa"
    PSurf.missing_value = -9999.
    PSurf.long_name = "Surface air pressure"
    PSurf.CF_name = "surface_air_pressure"

    LWdown = f.createVariable('LWdown', 'f8', ('time', 'y', 'x',))
    LWdown.units = "W/m^2"
    LWdown.missing_value = -9999.
    LWdown.long_name = "Surface incident longwave radiation"
    LWdown.CF_name = "surface_downwelling_longwave_flux_in_air"

    CO2air = f.createVariable('CO2air', 'f8', ('time', 'z', 'y', 'x',))
    CO2air.units = "ppm"
    CO2air.missing_value = -9999.
    CO2air.long_name = ""
    CO2air.CF_name = ""

    # write data to file
    x[:] = ndim
    y[:] = ndim
    z[:] = ndim
    time[:] = times
    latitude[:] = lat
    longitude[:] = lon

    SWdown[:,0,0] = df.swdown.values.reshape(n_timesteps, ndim, ndim)
    Tair[:,0,0,0] = df.tair.values.reshape(n_timesteps, ndim, ndim, ndim)
    Rainf[:,0,0] = df.rainf.values.reshape(n_timesteps, ndim, ndim)
    Qair[:,0,0,0] = df.qair.values.reshape(n_timesteps, ndim, ndim, ndim)
    Wind[:,0,0,0] = df.wind.values.reshape(n_timesteps, ndim, ndim, ndim)
    PSurf[:,0,0] = df.psurf.values.reshape(n_timesteps, ndim, ndim)
    LWdown[:,0,0] = df.lwdown.values.reshape(n_timesteps, ndim, ndim)
    CO2air[:,0,0] = df.co2.values.reshape(n_timesteps, ndim, ndim, ndim)

    f.close()

def convert_rh_to_qair(rh, tair, press):
    """
    Converts relative humidity to specific humidity (kg/kg)

    Params:
    -------
    tair : float
        deg C
    press : float
        pa
    rh : float
        [0-1]
    """
    tairC = tair - 273.15

    # Sat vapour pressure in Pa
    esat = calc_esat(tairC)

    # Specific humidity at saturation:
    ws = 0.622 * esat / (press - esat)

    # specific humidity
    qair = rh * ws

    return qair

def calc_esat(tair):
    """
    Calculates saturation vapour pressure

    Params:
    -------
    tair : float
        deg C

    Reference:
    ----------
    * Jones (1992) Plants and microclimate: A quantitative approach to
    environmental plant physiology, p110
    """

    esat = 613.75 * np.exp(17.502 * tair / (240.97 + tair))

    return esat

def estimate_lwdown(tair, rh):
    """
    Synthesises downward longwave radiation based on Tair RH

    Params:
    -------
    tair : float
        K C
    rh : float
        [0-1]

    Reference:
    ----------
    * Abramowitz et al. (2012), Geophysical Research Letters, 39, L04808

    """
    zeroC = 273.15

    sat_vapress = 611.2 * np.exp(17.67 * ((tair - zeroC) / (tair - 29.65)))
    vapress = np.maximum(0.05, rh) * sat_vapress
    lw_down = 2.648 * tair + 0.0346 * vapress - 474.0

    return lw_down

if __name__ == "__main__":

    lat = 47.43805556
    lon = 7.77694444

    fname = "../data/swiss site meto data 2017_18.xlsx"
    df = pd.read_excel(open(fname, 'rb'), sheet_name='meteo data 2018',
                       na_values="na")
    #df = pd.read_excel(open(fname, 'rb'), sheet_name='meteo data 2017',
    #                   na_values="na")


    # Clean up the column names
    df = df.rename(columns={'date / time utc':'dates',
                                    'air temp (°C)':'tair',
                                    'air humidity (%)':'rh',
                                    'global rad (W/m2)':'swdown',
                                    'wind (m/s)':'wind',
                                    'precip (mm)':'rainf',
                                    'vpd (hPa)':'vpd'})

    # Clean up the dates
    df['dates'] = df['dates'].astype(str).str[:-4]
    date = pd.to_datetime(df['dates'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('dates')

    # fix units
    hpa_2_kpa = 0.1
    kpa_2_pa = 1000.
    deg_2_kelvin = 273.15
    df.vpd *= hpa_2_kpa
    df.tair += deg_2_kelvin
    df.rainf /= 3600. # kg m-2 s-1

    # sort out negative values
    df.swdown = np.where(df.swdown < 0.0, 0.0, df.swdown)
    df.vpd = np.where(df.vpd < 0.0, 0.0, df.vpd)
    df.rh = np.where(df.rh < 0.0, 0.0, df.rh)
    df.rh = np.where(df.rh > 100.0, 100.0, df.rh)
    df.rh /= 100.



    # Add co2
    df_co2 = pd.read_csv("AmaFACE_co2npdepforcing_1850_2100_AMB.csv", sep=";")
    df_co2.rename(columns={'CO2 [ppm]':'co2'}, inplace=True)
    co2 = df_co2[df_co2.Year == 2018].co2.values[0]
    df['co2'] = co2

    # Add pressure
    df['psurf'] = 101.325 * kpa_2_pa

    # Add LW
    df['lwdown'] = estimate_lwdown(df.tair.values, df.rh.values)

    # Add qair
    df['qair'] = convert_rh_to_qair(df.rh.values, df.tair.values,
                                    df.psurf.values)

    # drop the single 2019 entry
    df.drop(df.tail(1).index, inplace=True)

    """
    #df = df.fillna(method='ffill')
    df['tair'].interpolate(method ='linear', limit_direction ='forward',
                          inplace=True)
    df['lwdown'].interpolate(method ='linear', limit_direction ='forward',
                            inplace=True)
    df['qair'].interpolate(method ='linear', limit_direction ='forward',
                            inplace=True)
    df['rainf'].fillna(0.0, inplace=True)
    df = df.fillna(df.mean()) # wind
    """



    ### this will look OK on the daily, but won't have a diurnal cycle,
    ### so need to use a diurnal fill
    #df['rainf'].fillna(0.0, inplace=True)
    #df['tair'].fillna(16.0+deg_2_kelvin, inplace=True)
    #df['swdown'].fillna(90., inplace=True)
    #df['lwdown'].fillna(320., inplace=True)
    #df = df.fillna(df.mean())

    main(lat, lon, df)
