#!/usr/bin/env python

"""
Turn the MAESPA input file into a CABLE netcdf file. Aim to swap MAESPA data
for the raw data later when I have more time...

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (04.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt

fname = "swiss_met.nc"

vars_to_keep = ['Tair','Qair','Rainf','Wind','PSurf','LWdown','SWdown','CO2air']
ds = xr.open_dataset(fname, decode_times=False)
freq = "H"
units, reference_date = ds.time.attrs['units'].split('since')
df = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True).to_dataframe()
start = reference_date.strip().split(" ")[0].replace("-","/")
df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)
df = df.set_index('dates')

df.Tair -= 273.15
df.SWdown *= 2.3

df = df.resample("D").agg("mean")


fig = plt.figure(figsize=(15,10))
fig.subplots_adjust(hspace=0.4)
fig.subplots_adjust(wspace=0.2)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Helvetica"
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

colours = plt.cm.Set2(np.linspace(0, 1, 7))

ax1 = fig.add_subplot(3,3,1)
ax2 = fig.add_subplot(3,3,2)
ax3 = fig.add_subplot(3,3,3)
ax4 = fig.add_subplot(3,3,4)
ax5 = fig.add_subplot(3,3,5)
ax6 = fig.add_subplot(3,3,6)
ax7 = fig.add_subplot(3,3,7)
ax8 = fig.add_subplot(3,3,8)

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

for a, v in zip(axes, vars_to_keep):
    a.set_title(v)


    #a.plot(df2[v].index.to_pydatetime(), df2[v].rolling(window=7).mean(), c=colours[2],
    #       lw=1.5, ls="-", label="cable", alpha=0.8)

    a.plot(df[v].index.to_pydatetime(), df[v].rolling(window=7).mean(), c=colours[1],
           lw=1.5, ls="-", label="narclim")

fig.autofmt_xdate()
#for a in axes:
#    a.set_xlim([datetime.date(1992,1,1), datetime.date(1993, 1, 1)])

plt.show()
