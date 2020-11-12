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

def main(df_obs, fname, plot_fname=None, fpath=None):


    df = read_cable_file(fname)

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

    ax1 = fig.add_subplot(111)

    df1 = df[(df.index.hour >= 5) & (df.index.hour < 6) &
             (df.index.minute >= 30)].copy()
    df2 = df[(df.index.hour >= 12) & (df.index.hour < 13) &
             (df.index.minute < 30)].copy()


    MMOL_2_MOL = 1E-03

    KG_2_G = 1000.
    G_WATER_TO_MOL_WATER = 1.0/18.02
    MOL_TO_MMOL = 1000.
    LAI = 4.8
    # mmol m-2 s-1 per unit leaf
    conv = KG_2_G * G_WATER_TO_MOL_WATER * MOL_TO_MMOL / LAI

    ax1.plot(df_obs["psi_x"], df_obs["E"], ls=" ", marker="o", c=colours[0],
             label="OBS", alpha=0.3, markersize=5)



    #df_obs_mu = df_obs.groupby([df_obs.index.year, df_obs.index.dayofyear]).mean()
    #print(df_obs_mu)
    #ax1.plot(df_obs_mu["psi_x"], df_obs_mu["E"], ls=" ", marker="o", c=colours[0],
    #         label="OBS")


    #df2 = df2[(df2.index.dayofyear >=df_obs.index.dayofyear[0]) &
    #          (df2.index.dayofyear <=df_obs.index.dayofyear[-1])]
    ax1.plot(df2["psi_leaf"], df2["TVeg"]*conv, ls=" ", marker="o",
            c=colours[1],  label="CABLE", alpha=0.5)


    ax1.set_ylabel("E (mmol m$^{2}$ s$^{-1}$)")
    ax1.set_xlabel("Water potential (MPa)")
    ax1.legend(numpoints=1, loc="best", fontsize=8)

    #fig.autofmt_xdate()

    if fpath is None:
        fpath = "./"
    ofname = os.path.join(fpath, plot_fname)
    fig.savefig(ofname, dpi=150, bbox_inches='tight', pad_inches=0.1)

def read_cable_file(fname):

    vars_to_keep = ['weighted_psi_soil','psi_leaf','psi_soil',\
                    'Rainf','SoilMoist', 'TVeg']

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

    # layer thickness
    #zse = [.022, .058, .154, .409, 1.085, 2.872]
    zse = [.022, .058, .134, .189, 1.325, 2.872] # Experiment with top 4 layer = 40 cm


    #zse = np.array([.022, .058, .134, .189, 1.325, 2.872])
    #change = np.ones(6) * 1.0
    #change[4] = 1.0 # don't change two bottom layers with no roots
    #change[5] = 1.0 # don't change two bottom layers with no roots

    #zse /= change
    #print(zse, np.sum(zse[0:4]))

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

    fname = "../data/swiss site gas exchange 2018_19_20.xlsx"
    df_gx = pd.read_excel(open(fname, 'rb'), sheet_name='2018')
    df_gx = df_gx.rename(columns={'date (dd.mm.yyyy)':'dates'})
    df_gx = df_gx.set_index('dates')

    # average over replicates on the same day
    df_gx = df_gx.groupby([df_gx.index, 'tree id']).mean()#.reset_index()
    #df_gx = df_gx.set_index('dates')

    fname = "../data/swiss site xylem pressure 2018.xlsx"
    df_psix = pd.read_excel(open(fname, 'rb'), sheet_name='xylem pressure 2018',
                            na_values="na")
    df_psix = df_psix.rename(columns={'date':'dates',
                            'xylem pressure (MPa)':'psi_x'})
    # psi_x = "-" we can't plot that, may as well drop
    df_psix = df_psix[df_psix["mortality observation"] != "dead"]

    df_psix = df_psix[df_psix["mortality observation"] == "surviving"]

    df_psix = df_psix.drop(['mortality observation','notes'], axis=1)
    df_psix = df_psix.set_index('dates')

    df_obs =  pd.merge(df_gx, df_psix, how='inner', on='dates')
    df_obs = df_obs.dropna()

    #fname = "outputs/hydraulics_root_1.0.nc"
    fname = "outputs/CABLE_swiss_mortality_living.nc"
    fpath = "/Users/mdekauwe/Desktop"

    plot_fname = "E_vs_water_potential.png"
    main(df_obs, fname, plot_fname, fpath)
