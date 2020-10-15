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

def main(plot_fname=None, fpath=None):

    fig = plt.figure(figsize=(14,6))
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
    colours = plt.cm.YlOrRd_r(np.linspace(0, 1, 11))

    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    zse = np.array([.022, .058, .134, .189])
    depth = np.sum(zse)
    print(depth)
    cnt = 0

    when_amb = []
    when_eco2 = []
    when_evpd = []
    when_eco2_evpd = []
    depth_amb = []
    depth_eco2 = []
    depth_evpd = []
    depth_eco2_evpd = []
    for dx in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:

        print(dx)
        fname = "outputs/hydraulics_root_%.1f.nc" % (dx)
        df = read_cable_file(fname)
        dfd = df.resample("D").agg("sum")
        dfm = df.resample("D").agg("max")

        fname = "outputs_eco2/hydraulics_root_%.1f.nc" % (dx)
        dfx = read_cable_file(fname)
        dfm_eco2 = dfx.resample("D").agg("max")

        fname = "outputs_evpd/hydraulics_root_%.1f.nc" % (dx)
        dfx = read_cable_file(fname)
        dfm_evpd = dfx.resample("D").agg("max")

        fname = "outputs_eco2_evpd/hydraulics_root_%.1f.nc" % (dx)
        dfx = read_cable_file(fname)
        dfm_eco2_evpd = dfx.resample("D").agg("max")

        actual_depth = depth / dx

        df1 = df[(df.index.hour >= 5) & (df.index.hour < 6) &
                 (df.index.minute >= 30)].copy()
        df2 = df[(df.index.hour >= 12) & (df.index.hour < 13) &
                 (df.index.minute < 30)].copy()

        window = 3

        axes = [ax1]

        ax1.plot(df2["psi_leaf"], df2["plc"], c=colours[cnt],
                 lw=1.5, ls=" ", marker=".", label="%.2f m" % (actual_depth))


        dfm = dfm[dfm.plc == 88]
        if len(dfm) > 0.0:
            when_amb.append(dfm.plc.index[0].dayofyear)
            depth_amb.append(actual_depth)

        dfm_eco2 = dfm_eco2[dfm_eco2.plc == 88]
        if len(dfm_eco2) > 0.0:
            when_eco2.append(dfm_eco2.plc.index[0].dayofyear)
            depth_eco2.append(actual_depth)

        dfm_evpd = dfm_evpd[dfm_evpd.plc == 88]
        if len(dfm_evpd) > 0.0:
            when_evpd.append(dfm_evpd.plc.index[0].dayofyear)
            depth_evpd.append(actual_depth)

        dfm_eco2_evpd = dfm_eco2_evpd[dfm_eco2_evpd.plc == 88]
        if len(dfm_eco2_evpd) > 0.0:
            when_eco2_evpd.append(dfm_eco2_evpd.plc.index[0].dayofyear)
            depth_eco2_evpd.append(actual_depth)

        #ax1.set_ylabel("$\Theta$ (m$^{3}$ m$^{-3}$)")
        #ax2.set_ylabel("Water potential (MPa)")
        #ax3.set_ylabel("E (mm d$^{-1}$)")
        #axx2.set_ylabel("Rainfall (mm d$^{-1}$)")


        #ax1.set_ylim(0, 0.4)
        #ax2.set_ylim(-8, 0.0)
        #ax3.set_ylim(0.0, 2.0)

        ax1.legend(numpoints=1, loc="best", fontsize=8, frameon=False)
        #ax2.legend(numpoints=1, loc="best", fontsize=8)

        #for a in axes:
        #    a.set_xlim([datetime.date(2018,1,1), datetime.date(2018, 12, 31)])

        #for a in axes:
        #    a.set_xlim([datetime.date(1993,1,1), datetime.date(1997, 1, 1)])
        #    #a.set_xlim([datetime.date(2004,1,1), datetime.date(2004, 8, 1)])
        #    a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])

        #plt.setp(ax1.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_xticklabels(), visible=False)

        #fig.autofmt_xdate()

        cnt += 1

    ax1.set_xlim(-5, 0)

    ax2.plot(depth_amb, when_amb, "ks", label="Amb")
    ax2.plot(depth_eco2, when_eco2, "go", label="eCO$_2$ (x2)")
    ax2.plot(depth_evpd, when_evpd, "ro", label="eVPD (x1.5)")
    ax2.plot(depth_eco2_evpd, when_eco2_evpd, "bo", label="eCO$_2$ + eVPD")

    ax2.legend(numpoints=1, loc="best", fontsize=8, frameon=False)


    ax1.set_ylabel("Max PLC (%)")
    ax1.set_xlabel("Water potential (MPa)")

    ax2.set_ylabel("Day of year carked it")
    ax2.set_xlabel("Depth (m)")

    if fpath is None:
        fpath = "./"
    ofname = os.path.join(fpath, plot_fname)
    fig.savefig(ofname, dpi=150, bbox_inches='tight', pad_inches=0.1)

def read_cable_file(fname):

    vars_to_keep = ['weighted_psi_soil','psi_leaf','psi_soil',\
                    'Rainf','SoilMoist', 'TVeg', 'plc']

    ds = xr.open_dataset(fname, decode_times=False)

    time_jump = int(ds.time[1].values) - int(ds.time[0].values)
    if time_jump == 3600:
        freq = "H"
    elif time_jump == 1800:
        freq = "30M"
    else:
        raise("Time problem")

    units, reference_date = ds.time.attrs['units'].split('since')
    ds = ds[vars_to_keep].squeeze(dim=["x","y","patch"], drop=True)


    ds['psi_soil1'] = ds['psi_soil'][:,0]
    ds['psi_soil2'] = ds['psi_soil'][:,1]
    ds['psi_soil3'] = ds['psi_soil'][:,2]
    ds['psi_soil4'] = ds['psi_soil'][:,3]
    ds['psi_soil5'] = ds['psi_soil'][:,4]
    ds['psi_soil6'] = ds['psi_soil'][:,5]
    ds = ds.drop("psi_soil")

    ds['Rainf'] *= float(time_jump)
    ds['TVeg'] *= float(time_jump)

    # layer thickness
    #zse = [.022, .058, .154, .409, 1.085, 2.872]
    zse = [.022, .058, .134, .189, 1.325, 2.872] # Experiment with top 4 layer = 40 cm

    frac1 = zse[0] / (zse[0] + zse[1])
    frac2 = zse[1] / (zse[0] + zse[1])
    frac3 = zse[2] / (zse[2] + zse[3])
    frac4 = zse[3] / (zse[2] + zse[3])
    frac5 = zse[4] / (zse[4] + zse[4])
    frac6 = zse[5] / (zse[5] + zse[5])

    ds['theta1'] = (ds['SoilMoist'][:,0] * frac1) + \
                   (ds['SoilMoist'][:,1] * frac2)
    ds['theta2'] = (ds['SoilMoist'][:,2] * frac3) + \
                   (ds['SoilMoist'][:,3] * frac4)
    ds['theta3'] = (ds['SoilMoist'][:,4] * frac5) + \
                   (ds['SoilMoist'][:,5] * frac6)




    """
    froot = np.array([0.0343, 0.0302, 0.2995, 0.636, 0.0, 0.0])
    frac1 = froot[0] / np.sum(froot[0:4])
    frac2 = froot[1] / np.sum(froot[0:4])
    frac3 = froot[2] / np.sum(froot[0:4])
    frac4 = froot[3] / np.sum(froot[0:4])
    frac5 = froot[4] / np.sum(froot[0:4])
    frac6 = froot[5] / np.sum(froot[0:4])

    ds['theta_weight'] = (ds['SoilMoist'][:,0] * froot[0]) + \
                         (ds['SoilMoist'][:,1] * froot[1]) + \
                         (ds['SoilMoist'][:,2] * froot[2]) + \
                         (ds['SoilMoist'][:,3] * froot[3]) #+ \
                         #(ds['SoilMoist'][:,4] * frac5) + \
                        # (ds['SoilMoist'][:,5] * frac6)
    """



    ds = ds.drop("SoilMoist")
    df = ds.to_dataframe()

    start = reference_date.strip().split(" ")[0].replace("-","/")
    df['dates'] = pd.date_range(start=start, periods=len(df), freq=freq)
    df = df.set_index('dates')

    return df


if __name__ == "__main__":


    fname = "outputs/hydraulics.nc"
    fpath = "/Users/mdekauwe/Desktop"

    plot_fname = "sensitivity_to_root_depth.png"
    main(plot_fname, fpath)
