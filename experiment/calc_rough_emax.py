#!/usr/bin/env python

"""
Plot SWP

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import xarray as xr
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob

depth = 0.2 # m
m_2_mm = 1000.

fname = "../data/swiss site soil water content 2018.xlsx"
df_swc = pd.read_excel(open(fname, 'rb'), sheet_name='soil water content')
df_swc = df_swc.rename(columns={'timestamp':'dates',
                        'mean soil water content (m3/m3)':'swc',
                        'air humidity (%)':'rh',
                        'global radiation (W/m2)':'swdown',
                        'wind speed (m/s)':'wind',
                        'precip (mm)':'rainf',
                        'vapor presure (hPa)':'vpd'})
df_swc = df_swc.drop(['sample size'], axis=1)
df_swc = df_swc.set_index('dates')

max_delta = -9999.
all_delta = []
for i in range(1, len(df_swc)):

    delta = np.abs(df_swc.swc[i] - df_swc.swc[i-1])
    if delta > max_delta:
        max_delta = delta
    all_delta.append(delta)

all_delta = np.asarray(all_delta)
all_delta *= depth * m_2_mm

plt.hist(all_delta)
plt.show()


print(max_delta * depth * m_2_mm)
print(np.percentile(all_delta, 99))
