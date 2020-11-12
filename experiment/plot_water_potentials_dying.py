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

def main(df_swc, df_psix, fname, plot_fname=None, fpath=None):


    df = read_cable_file(fname)
    dfd = df.resample("D").agg("sum")
    dfa = df.resample("A").agg("sum")
    #print(dfa.TVeg.values[0], (dfa.ECanop/dfa.Evap).values[0]*100.)
    print(np.max(dfd["TVeg"]), np.max(dfd["ESoil"]))
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

    df1 = df[(df.index.hour >= 5) & (df.index.hour < 6) &
             (df.index.minute >= 30)].copy()
    df2 = df[(df.index.hour >= 12) & (df.index.hour < 13) &
             (df.index.minute < 30)].copy()

    window = 3

    axes = [ax1, ax2, ax3]
    axes2 = [axx1, axx2, axx3]
    ax1.plot(df.index, df["theta1"].rolling(window=window).mean(), c=colours[1],
             lw=1.5, ls="-", label="1-2", zorder=1)
    ax1.plot(df.index, df["theta2"].rolling(window=window).mean(), c=colours[2],
             lw=1.5, ls="-", label="3-4")
    ax1.plot(df.index, df["theta3"].rolling(window=window).mean(), c=colours[0],
             lw=1.5, ls="-", label="5+6")
    #ax1.plot(df.index, df["theta_weight"].rolling(window=window).mean(), c=colours[5],
    #         lw=1.5, ls="-", label="weighted")


    ax1.plot(df_swc.index, df_swc["swc"].rolling(window=window).mean(), c=colours[4],
             lw=1., ls="-", label="OBS", alpha=0.8)

    lower = (df_swc["swc"] - df_swc["sd"]).rolling(window=window).mean()
    upper = (df_swc["swc"] + df_swc["sd"]).rolling(window=window).mean()
    ax1.fill_between(df_swc.index, lower, upper, color=colours[4],alpha=.5)

    #ax1.plot(df.index, df["theta4"], c=colours[4],
    #         lw=1.5, ls="-", label="4")
    #ax1.plot(df.index, df["theta5"], c=colours[5],
    #         lw=1.5, ls="-", label="5")
    #ax1.plot(df.index, df["theta6"], c=colours[6],
    #         lw=1.5, ls="-", label="6")
    #ax1.set_ylim(-0.3, 0.0)

    ax2.plot(df.index, df["weighted_psi_soil"].rolling(window=window).mean(), c=colours[2],
             lw=1.5, ls="-", label="Pre-dawn $\Psi$$_{s}$ weight")

    ax2.plot(df2.index, df2["psi_leaf"].rolling(window=window).mean(), c=colours[1],
             lw=1.5, ls="-", label="Midday $\Psi$$_{l}$")

    df_psi_survive = df_psix[df_psix["mortality observation"] == "surviving"]
    df_psi_dying = df_psix[df_psix["mortality observation"] == "dying "] # NB white space in file

    #ax2.plot(df_psi_survive.index, df_psi_survive["psi_x"], c="black", ls=" ",
    #         marker=".", label="Surviving", alpha=0.5)
    ax2.plot(df_psi_dying.index, df_psi_dying["psi_x"], c="red", ls=" ",
             marker=".", label="Obs. $\Psi$$_{x}$", alpha=0.5)

    ax3.plot(dfd.index, dfd["TVeg"].rolling(window=window).mean(), c=colours[0], lw=1.5, ls="-", label="Etr")
    #ax3.plot(dfd.index, dfd["ESoil"].rolling(window=window).mean(), c=colours[1], lw=1.5, ls="-", label="Esoil")
    #ax3.plot(dfd.index, dfd["TVeg"].cumsum(), c=colours[0], lw=1.5, ls="-")

    #dfx = read_cable_file("outputs/hydraulics_root_2.0.nc")
    #dfdx = dfx.resample("D").agg("sum")
    #ax3.plot(dfdx.index, dfdx["TVeg"].cumsum(), c="black", lw=1.5, ls="-")


    #ax3.set_ylim(0, 7.5)

    #ax3.set_ylim(-5, 0)
    axx1.bar(dfd.index, dfd["Rainf"], alpha=0.3, color="black")
    axx2.bar(dfd.index, dfd["Rainf"], alpha=0.3, color="black")
    axx3.bar(dfd.index, dfd["Rainf"], alpha=0.3, color="black")

    ax1.set_ylabel("$\Theta$ (m$^{3}$ m$^{-3}$)")
    ax2.set_ylabel("Water potential (MPa)")
    ax3.set_ylabel("E (mm d$^{-1}$)")
    axx2.set_ylabel("Rainfall (mm d$^{-1}$)")


    ax1.set_ylim(0, 0.4)
    #ax2.set_ylim(-8, 0.0)
    #ax3.set_ylim(0.0, 2.0)

    ax1.legend(numpoints=1, loc="best", fontsize=8)
    ax2.legend(numpoints=1, loc="best", fontsize=8)
    ax3.legend(numpoints=1, loc="best", fontsize=8)

    #for a in axes:
    #    a.set_xlim([datetime.date(2018,1,1), datetime.date(2018, 12, 31)])

    #for a in axes:
    #    a.set_xlim([datetime.date(1993,1,1), datetime.date(1997, 1, 1)])
    #    #a.set_xlim([datetime.date(2004,1,1), datetime.date(2004, 8, 1)])
    #    a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    #fig.autofmt_xdate()

    if fpath is None:
        fpath = "./"
    ofname = os.path.join(fpath, plot_fname)
    fig.savefig(ofname, dpi=150, bbox_inches='tight', pad_inches=0.1)

def read_cable_file(fname):

    vars_to_keep = ['weighted_psi_soil','psi_leaf','psi_soil',\
                    'Rainf','SoilMoist', 'TVeg', 'ESoil','Evap','ECanop']

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
    ds['ESoil'] *= float(time_jump)
    ds['Evap'] *= float(time_jump)
    ds['ECanop'] *= float(time_jump)

    # layer thickness
    #zse = [.022, .058, .154, .409, 1.085, 2.872]
    #zse = [.022, .058, .134, .189, 1.325, 2.872] # Experiment with top 4 layer = 40 cm
    zse = np.array([0.022, 0.058, 0.0193, 0.0292, 0.1775, 0.294]) # Experiment with top 6 layer = 60 cm

    #zse = np.array([.022, .058, .134, .189, 1.325, 2.872])
    #change = np.ones(6) * 1.0
    #change[4] = 1.0 # don't change two bottom layers with no roots
    #change[5] = 1.0 # don't change two bottom layers with no roots

    #zse /= change
    #print(zse, np.sum(zse[0:4]))

    # three layers
    frac1 = zse[0] / np.sum(zse[0:2])
    frac2 = zse[1] / np.sum(zse[0:2])
    frac3 = zse[2] / np.sum(zse[2:4])
    frac4 = zse[3] / np.sum(zse[2:4])
    frac5 = zse[4] / np.sum(zse[4:])
    frac6 = zse[5] / np.sum(zse[4:])

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

    fname = "../data/swiss site xylem pressure 2018.xlsx"
    df_psix = pd.read_excel(open(fname, 'rb'), sheet_name='xylem pressure 2018',
                            na_values="na")
    df_psix = df_psix.rename(columns={'date':'dates',
                            'xylem pressure (MPa)':'psi_x'})
    # psi_x = "-" we can't plot that, may as well drop
    df_psix = df_psix[df_psix["mortality observation"] != "dead"]

    #df_psix = df_psix.drop(['mortality observation','notes'], axis=1)
    df_psix = df_psix.set_index('dates')

    #print(df_obs.swc.max(), df_obs.swc.min())
    #sys.exit()
    fname = "outputs/CABLE_swiss_mortality_dying.nc"
    fpath = "/Users/mdekauwe/Desktop"

    plot_fname = "water_potentials_dying.png"
    main(df_swc, df_psix, fname, plot_fname, fpath)
