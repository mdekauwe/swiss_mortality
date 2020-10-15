#!/usr/bin/env python

"""
Plot visual benchmark (average seasonal cycle) of old vs new model runs.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob
import xarray as xr

def main(fname1, fname2, plot_fname=None):

    df1 = read_cable_file(fname1, type="CABLE")
    df1 = resample_timestep(df1, type="CABLE")

    df2 = read_cable_file(fname2, type="CABLE")
    df2 = resample_timestep(df2, type="CABLE")

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.1)
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

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    axx1 = ax1.twinx()
    axx2 = ax2.twinx()
    axx3 = ax3.twinx()

    axes = [ax1, ax2, ax3]

    axes2 = [axx1, axx2, axx3]
    vars = ["GPP", "TVeg", "LAI"]
    for a, x, v in zip(axes, axes2, vars):

        #a.plot(df1[v].index.to_pydatetime(), df1[v].rolling(window=7).mean(), c=colours[0],
        #       lw=1.5, ls="-", label="Standard")
        if v == "TVeg":
            a.plot(df2[v].index.to_pydatetime(), df2["ESoil"].rolling(window=3).mean(), c=colours[0],
                   lw=1.5, ls="-", label="Standard")
            a.plot(df2[v].index.to_pydatetime(), df2[v].rolling(window=3).mean(), c=colours[2],
                   lw=1.5, ls="-", label="Hydraulics")
        else:
            a.plot(df2[v].index.to_pydatetime(), df2[v].rolling(window=3).mean(), c=colours[2],
                   lw=1.5, ls="-", label="Hydraulics")

        x.bar(df2.index, df2["Rainf"], alpha=0.3, color="black")


    ax2.set_ylim(0, 7.5)
    labels = ["GPP (g C m$^{-2}$ d$^{-1}$)", "E (mm d$^{-1}$)", "LAI (m$^{2}$ m$^{-2}$)"]
    for a, l in zip(axes, labels):
        a.set_ylabel(l, fontsize=12)

    for x, l in zip(axes2, labels):
        x.set_ylabel("Rainfall (mm d$^{-1}$)", fontsize=12)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    #ax1.legend(numpoints=1, loc="best")


    #for a in axes:
    #    a.set_xlim([datetime.date(2018,1,1), datetime.date(2018, 12, 31)])
        #a.set_xlim([datetime.date(2002,12,1), datetime.date(2003, 4, 1)])
    #    a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])
        #a.set_xlim([datetime.date(2012,7,1), datetime.date(2013, 8, 1)])
        #a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])
    #ax2.set_ylim(0, 3)

    if plot_fname is None:
        plt.show()
    else:
        #fig.autofmt_xdate()
        fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)

    ofname = "/Users/mdekauwe/Desktop/timeseries.png"
    fig.savefig(ofname, dpi=150, bbox_inches='tight', pad_inches=0.1)

def read_cable_file(fname, type=None):

    if type == "CABLE":
        vars_to_keep = ['GPP','Qle','LAI','TVeg','ESoil','NEE','Rainf']
    elif type == "FLUX":
        vars_to_keep = ['GPP','Qle']
    elif type == "MET":
        vars_to_keep = ['Rainf']

    ds = xr.open_dataset(fname, decode_times=False)

    time_jump = int(ds.time[1].values) - int(ds.time[0].values)
    if time_jump == 3600:
        freq = "H"
    elif time_jump == 1800:
        freq = "30M"
    else:
        raise("Time problem")

    units, reference_date = ds.time.attrs['units'].split('since')
    df = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True).to_dataframe()
    start = reference_date.strip().split(" ")[0].replace("-","/")
    df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)
    df = df.set_index('dates')

    return df


def resample_timestep(df, type=None):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_HLFHOUR = 1800.
    SEC_2_HOUR = 3600.

    if type == "CABLE":
        # umol/m2/s -> g/C/30min
        #df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HLFHOUR
        df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HOUR

        # kg/m2/s -> mm/30min
        df['ESoil'] *= SEC_2_HOUR
        df['TVeg'] *= SEC_2_HOUR
        df['Rainf'] *= SEC_2_HOUR

        method = {'GPP':'sum', 'TVeg':'sum', "Qle":"mean", "LAI":"mean",
                  "ESoil":"sum", 'Rainf':'sum'}
    elif type == "FLUX":
        # umol/m2/s -> g/C/30min
        #df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HLFHOUR
        df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HOUR

        method = {'GPP':'sum', "Qle":"mean"}

    elif type == "MET":
        # kg/m2/s -> mm/30min
        df['Rainf'] *= SEC_2_HLFHOUR

        method = {'Rainf':'sum'}

    df = df.resample("D").agg(method)

    return df


    return dates

if __name__ == "__main__":



    fname1 = "outputs/standard.nc"
    fname2 = "outputs/hydraulics.nc"
    main(fname1, fname2)
