#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
import glob

#filenames_2020combo = glob.glob('/data/user/zzhang1/24h2021l3/h5file/2020/comboV01-01-00/*/*.h5')
#filenames_2020official = glob.glob('/data/user/zzhang1/24h2021l3/h5file/2020/official/*/*.h5')

start_time_2020combo = []
start_time_2020official = []

fill_ratio_2020combo = []
fill_ratio_2020official = []

rlogl_2020combo = []
rlogl_2020official = []

nstring_2020combo = []
nstring_2020official = []

energy_L3_2020combo = []
energy_L3_2020official = []

zenith_L3_2020combo = []
zenith_L3_2020official = []

azimuth_L3_2020combo = []
azimuth_L3_2020official = []

energy_redo_2020combo = []
energy_redo_2020official = []

zenith_redo_2020combo = []
zenith_redo_2020official = []

azimuth_redo_2020combo = []
azimuth_redo_2020official = []

'''
for filename in filenames_2020combo:
    cond1 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy.notna()
    cond2 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith.notna()
    cond3 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth.notna()
    cond4 = pd.read_hdf(filename, 'RedoMonopodAmpFit').energy.notna()
    cond5 = pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith.notna()
    cond6 = pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth.notna()

    cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    header = pd.read_hdf(filename,'I3EventHeader')
    current_start_time = list(header['time_start_utc_daq'])
    start_time_2020combo += current_start_time
    fill_ratio_2020combo += list(pd.read_hdf(filename,'CascadeFillRatio_L2')[cond].fillratio_from_mean)
    rlogl_2020combo += list(pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams')[cond].rlogL)
    nstring_2020combo += list(pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC')[cond].value)
    energy_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].energy)
    zenith_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].zenith)
    azimuth_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].azimuth)
    energy_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].energy)
    zenith_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].zenith)
    azimuth_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].azimuth)

    fill_ratio_2020combo += list(pd.read_hdf(filename,'CascadeFillRatio_L2').fillratio_from_mean)
    rlogl_2020combo += list(pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams').rlogL)
    nstring_2020combo += list(pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC').value)
    energy_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy)
    zenith_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith)
    azimuth_L3_2020combo += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth)
    energy_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').energy)
    zenith_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith)
    azimuth_redo_2020combo += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth)
'''

filename = "2020_combo_concat.h5"
start_time_2020combo =pd.read_hdf(filename, 'I3EventHeader').time_start_utc_daq
fill_ratio_2020combo = pd.read_hdf(filename,"CascadeFillRatio_L2").fillratio_from_mean
rlogl_2020combo = pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams').rlogL
nstring_2020combo = pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC').value
energy_L3_2020combo = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy
zenith_L3_2020combo = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith
azimuth_L3_2020combo = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth
energy_redo_2020combo = pd.read_hdf(filename, 'RedoMonopodAmpFit').energy
zenith_redo_2020combo = pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith
azimuth_redo_2020combo = pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth



'''
for filename in filenames_2020official:
    cond1 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy.notna()
    cond2 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith.notna()
    cond3 = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth.notna()
    cond4 = pd.read_hdf(filename, 'RedoMonopodAmpFit').energy.notna()
    cond5 = pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith.notna()
    cond6 = pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth.notna()

    cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    header = pd.read_hdf(filename,'I3EventHeader')
    current_start_time = list(header['time_start_utc_daq'])
    start_time_2020official += current_start_time

    fill_ratio_2020official += list(pd.read_hdf(filename,'CascadeFillRatio_L2')[cond].fillratio_from_mean)
    rlogl_2020official += list(pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams')[cond].rlogL)
    nstring_2020official += list(pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC')[cond].value)
    energy_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].energy)
    zenith_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].zenith)
    azimuth_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit')[cond].azimuth)
    energy_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].energy)
    zenith_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].zenith)
    azimuth_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit')[cond].azimuth)

    fill_ratio_2020official += list(pd.read_hdf(filename,'CascadeFillRatio_L2').fillratio_from_mean)
    rlogl_2020official += list(pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams').rlogL)
    nstring_2020official += list(pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC').value)
    energy_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy)
    zenith_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith)
    azimuth_L3_2020official += list(pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth)
    energy_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').energy)
    zenith_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith)
    azimuth_redo_2020official += list(pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth)
'''

filename = "2020_official_concat.h5"
start_time_2020official =pd.read_hdf(filename, 'I3EventHeader').time_start_utc_daq
fill_ratio_2020official = pd.read_hdf(filename,"CascadeFillRatio_L2").fillratio_from_mean
rlogl_2020official = pd.read_hdf(filename,'CascadeLlhVertexFit_ICParams').rlogL
nstring_2020official = pd.read_hdf(filename, 'NString_OfflinePulsesHLC_noDC').value
energy_L3_2020official = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').energy
zenith_L3_2020official = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').zenith
azimuth_L3_2020official = pd.read_hdf(filename, 'L3_MonopodFit4_AmptFit').azimuth
energy_redo_2020official = pd.read_hdf(filename, 'RedoMonopodAmpFit').energy
zenith_redo_2020official = pd.read_hdf(filename, 'RedoMonopodAmpFit').zenith
azimuth_redo_2020official = pd.read_hdf(filename, 'RedoMonopodAmpFit').azimuth

def get_ratio_error(n1,err1,n2,err2):
    eps = 1/np.power(10,10)
    ratio_error = n1/(n2+eps)*np.sqrt((err1/(n1+eps))**2+(err2/(n2+eps))**2)
    return ratio_error

interval24h = np.linspace(0,86400,145)
#total_time = 86400 # second per 24 hours
w2020combo = np.power(10,10)/(np.max(start_time_2020combo)-np.min(start_time_2020combo)) # to calculate rate
w2020official = np.power(10,10)/(np.max(start_time_2020official)-np.min(start_time_2020official)) # to calculate rate
eps = 1/np.power(10,10)

print("run live time 2020combo:{} [s]".format(w2020combo))
print("number of events 2020combo:",len(fill_ratio_2020combo))
print("run live time 2020official:{} [s]".format(w2020official))
print("number of events 2020official:",len(fill_ratio_2020official))

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(7,10,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(rlogl_2020combo, weights = w2020combo*np.ones(len(rlogl_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(rlogl_2020official, weights = w2020official*np.ones(len(rlogl_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)

plt.title('2021 24h check level3 rlogl')
ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'upper right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[CascadeLlhVertexFit_ICParams].rlogL')
plt.grid()
plt.savefig('rlogl.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(3,20,18)
n2020combo, bins2020combo, patch2020combo = plt.hist(nstring_2020combo, weights = w2020combo*np.ones(len(nstring_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(nstring_2020official, weights = w2020official*np.ones(len(nstring_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)

plt.title('2021 24h check level3 nstring')
ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'upper right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[NString_OfflinePulsesHLC_noDC].value')
plt.grid()
plt.savefig('nstring.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(0.55,1,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(fill_ratio_2020combo, weights = w2020combo*np.ones(len(fill_ratio_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(fill_ratio_2020official, weights = w2020official*np.ones(len(fill_ratio_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 fill_ratio')
ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'upper left')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[CascadeFillRatio_L2].fillratio_from_mean')
plt.grid()
plt.savefig('fill_ratio.png', dpi=400)
plt.clf()

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.logspace(1,8,31)
n2020combo, bins2020combo, patch2020combo = plt.hist(energy_L3_2020combo, weights = w2020combo*np.ones(len(energy_L3_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(energy_L3_2020official, weights = w2020official*np.ones(len(energy_L3_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 energy_L3')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'upper right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_xscale('log')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[L3_MonopodFit4_AmptFit].energy')
plt.grid()
plt.savefig('energy_L3.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(-1,1,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(np.cos(zenith_L3_2020combo), weights = w2020combo*np.ones(len(zenith_L3_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(np.cos(zenith_L3_2020official), weights = w2020official*np.ones(len(zenith_L3_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 zenith_L3')
#ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'lower right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('cos([L3_MonopodFit4_AmptFit].zenith)')
plt.grid()
plt.savefig('zenith_L3.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(0,2*np.pi,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(azimuth_L3_2020combo, weights = w2020combo*np.ones(len(azimuth_L3_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(azimuth_L3_2020official, weights = w2020official*np.ones(len(azimuth_L3_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 azimuth_L3')
#ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'lower right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[L3_MonopodFit4_AmptFit].azimuth')
plt.grid()
plt.savefig('azimuth_L3.png', dpi=400)
plt.clf()



fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.logspace(1,8,31)
n2020combo, bins2020combo, patch2020combo = plt.hist(energy_redo_2020combo, weights = w2020combo*np.ones(len(energy_redo_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(energy_redo_2020official, weights = w2020official*np.ones(len(energy_redo_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 energy_redo')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'upper right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_xscale('log')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[RedoMonopodAmpFit].energy')
plt.grid()
plt.savefig('energy_redo.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(-1,1,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(np.cos(zenith_redo_2020combo), weights = w2020combo*np.ones(len(zenith_redo_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(np.cos(zenith_redo_2020official), weights = w2020official*np.ones(len(zenith_redo_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 zenith_redo')
#ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'lower right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('cos([RedoMonopodAmpFit].zenith)')
plt.grid()
plt.savefig('zenith_redo.png', dpi=400)
plt.clf()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_axes([0.1,0.5,0.85,0.4])
b = np.linspace(0,2*np.pi,30)
n2020combo, bins2020combo, patch2020combo = plt.hist(azimuth_redo_2020combo, weights = w2020combo*np.ones(len(azimuth_redo_2020combo)), label = '2020combo', histtype = 'step', bins = b)
n2020official, bins2020official, patch2020official = plt.hist(azimuth_redo_2020official, weights = w2020official*np.ones(len(azimuth_redo_2020official)), label = '2020official', histtype = 'step', bins = b, linestyle = 'dashed')
err2020combo = np.sqrt(n2020combo*w2020combo)
err2020official = np.sqrt(n2020official*w2020official)
errratio20_20 = get_ratio_error(n2020combo, err2020combo, n2020official, err2020official)
plt.title('2021 24h check level3 azimuth_redo')
#ax1.set_yscale('log')
ax1.set_ylabel('Rate [Hz]')
ax1.set_xlim(bins2020combo[0],bins2020combo[-1])
ax1.set_xticklabels([])
ax1.legend(loc = 'lower right')
plt.grid()
ax3 = fig.add_axes([0.1,0.1,0.85,0.2], xlim=(bins2020combo[0], bins2020combo[-1]), ylim=(0.8,1.2))
plt.errorbar((bins2020combo[0:-1]+bins2020combo[1:])/2,n2020combo/(n2020official+eps), yerr=errratio20_20, fmt='.k')
ax3.set_ylim(0.8,1.2)
ax3.set_ylabel('n2020combo/n2020official')
ax3.set_xlabel('[RedoMonopodAmpFit].azimuth')
plt.grid()
plt.savefig('azimuth_redo.png', dpi=400)
plt.clf()
