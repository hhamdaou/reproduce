# %%

import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
import math as m
import tables
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from optparse import OptionParser
from matplotlib.colors import LogNorm,Normalize
from pylab import *
from scipy.interpolate import interp1d, UnivariateSpline
import glob

# %%
name_nu_1 = []
name_nu_02 = []
name_nu_50 = []

# %%
name_nu_1 = glob.glob('/data/user/hhamdaoui/reproduce/NuTau_CSMS_1_2.h5') # my file full sim
#name_nu_1 = glob.glob('/data/user/hhamdaoui/reproduce/NuTau:NuTauBar_G1.0_E3:7_Z0:180_1_csms_FullnoGR.h5') # my file NuGeN Only

#name_nu_1 = glob.glob('/data/user/joanna/CSecGen-Data/020_v4/nugen_nodetector_nutau_100_50*.h5') #YX file
name_nu_02 = glob.glob('/data/user/joanna/CSecGen-Data/020_v4/nugen_nodetector_nutau_020_50*.h5')
name_nu_50 = glob.glob('/data/user/joanna/CSecGen-Data/500_v4/nugen_nodetector_nutau_500_50*.h5')

# %%
name_nu_1  =name_nu_1[:1]
name_nu_02 =name_nu_1[:1]
name_nu_50 =name_nu_1[:1]

# %%
len(name_nu_1)

# %%
NFiles_nu_1 =  2000.*np.ones(1)
NFiles_nu_02 = 2000.*np.ones(1)
NFiles_nu_50 = 1980.*np.ones(1)

# %%


pdist = lambda p1, p2 : ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def weight_cal(f,fit,NFiles,livetime,selfveto):
    if fit == 0:
        index=-2
        norm_astro=1
        norm_conv=1
        norm_prompt=1
    elif fit == 2:
        index=-2.67
        norm_astro=2.15
        norm_conv=0.84
        norm_prompt=0.47
    elif fit == 1:
        index = -2.73
        norm_astro=2.2
        norm_conv=0.85
        norm_prompt=0.3
    elif fit == 8:
        index = -2.48
        norm_astro = 1.61
        norm_conv = 1.06
        norm_prompt = 1.

    OneWeight=f.root.I3MCWeightDict.col('OneWeight')
    NEvents=f.root.I3MCWeightDict.col('NEvents')
    PrimaryNeutrinoEnergy=f.root.I3MCWeightDict.col('PrimaryNeutrinoEnergy')
    #Conv_Weight=f.root.AtmWeight_Conv.col('value')
    #Prompt_Weight=f.root.AtmWeight_Prompt.col('value')
    #Prompt_PassRate=f.root.AtmWeight_Prompt_PassRate.col('value')
    #Conv_PassRate=f.root.AtmWeight_Conv_PassRate.col('value')
    print('NEvents=',NEvents)
    print('NFiles=',NFiles)

    
    astro_weight = OneWeight/(NEvents*NFiles)*1.0e-18*np.power(PrimaryNeutrinoEnergy/100000,index)*norm_astro*livetime
    #prompt_weight = OneWeight*2/(NEvents*NFiles)*Prompt_Weight*norm_prompt*livetime*Prompt_PassRate
    #conv_weight = OneWeight*2/(NEvents*NFiles)*Conv_Weight*norm_conv*livetime*Conv_PassRate
    #if selfveto == 1:
       # prompt_weight = OneWeight*2/(NEvents*NFiles)*Prompt_Weight*norm_prompt*livetime*Prompt_PassRate
      #  conv_weight = OneWeight*2/(NEvents*NFiles)*Conv_Weight*norm_conv*livetime*Conv_PassRate
     #   print ("SelfVeto")
    #else:
        #print ("No SelfVeto")
       # prompt_weight = OneWeight*2/(NEvents*NFiles)*Prompt_Weight*norm_prompt*livetime
      #  conv_weight = OneWeight*2/(NEvents*NFiles)*Conv_Weight*norm_conv*livetime


    #return zip(astro_weight,conv_weight,prompt_weight)
    #return astro_weight+conv_weight+prompt_weight
    return astro_weight

def timesplit_cal(f,name):
    x0=f.getNode(name+'0').col('x')
    x1=f.getNode(name+'1').col('x')
    y0=f.getNode(name+'0').col('y')
    y1=f.getNode(name+'1').col('y')
    z0=f.getNode(name+'0').col('z')
    z1=f.getNode(name+'1').col('z')
    timesplit=np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
    return timesplit

class variable:
    def __init__(self,f,year,NFiles):
        self.f=f
        self.xsec = np.log10(f.root.I3MCWeightDict.col('TotalXsectionCGS')*1.0e27) #converting xsec unit from cm^2 to mb
        #self.xsec = np.log10(f.root.I3MCWeightDict.col('TotalCrosssectionCGS')*1.0e27) #converting xsec unit from cm^2 to mb #YX files
        #self.xsec = np.log10(f.root.I3MCWeightDict.col('TotalXsectionCGS')*1.0e27)
        fit = 8
        #livetime = 1.0*3600*24*365*5
        #livetime = 1.0
        livetime = 1.0*(31.04*3600*24+3156904+2820269+2909466+2966167)
        #self.weight_wo_sv=weight_cal(self.f,fit,NFiles,livetime,0)
        self.weight=weight_cal(self.f,fit,NFiles,livetime,0)


        self.zenith = np.cos(f.root.I3MCWeightDict.col('PrimaryNeutrinoZenith'))
        #self.inice_primary_energy = np.log10(f.root.cscdSBU_MCInIcePrimary.col('energy')) #YX files
        self.inice_primary_energy = np.log10(f.root.I3MCWeightDict.col('PrimaryNeutrinoEnergy'))
        self.primary_energy = np.log10(f.root.I3MCWeightDict.col('PrimaryNeutrinoEnergy'))
        self.diff = (np.power(10,self.primary_energy)-np.power(10,self.inice_primary_energy))/np.power(10,self.primary_energy)*100.0
        #self.prop=f.root.I3MCWeightDict.col('TotalPropagationProbability')
        #self.prop=f.root.I3MCWeightDict.col('PropagationWeight')
        #self.type= f.root.I3MCWeightDict.col('PrimaryNeutrinoType')
        #self.inter_type = f.root.I3MCWeightDict.col('InteractionType')
        #self.energy = np.log10(f.root.I3MCWeightDict.col('PrimaryNeutrinoEnergy'))
    

def selection(sample):
    #idx = sample.zenith > -100
    idx = np.isfinite(sample.xsec)
    idx = np.logical_and(idx,np.logical_not(np.isnan(sample.zenith)))
    idx = np.logical_and(idx,np.logical_not(np.isnan(sample.primary_energy)))
    idx = np.logical_and(idx,np.logical_not(np.isnan(sample.inice_primary_energy)))
    idx = np.logical_and(idx,np.logical_not(np.isnan(sample.weight)))
    #idx = np.logical_and(idx,sample.diff>0)
    #idx = np.logical_and(idx,np.logical_not(abs(sample.type)==14))
    #idx = np.logical_not(abs(sample.type)==14)
    #idx = abs(sample.type)==12
    #idx = np.logical_not(abs(sample.type)==14)
    #idx = np.logical_or(idx,sample.inter_type==2)
    #idx = np.logical_and(idx,idx_f)
    return idx


seperation = 0.

function = selection

nu_1_inice=[]
nu_02_inice=[]
nu_50_inice=[]
nu_1_zenith=[]
nu_02_zenith=[]
nu_50_zenith=[]
nu_1_weight=[]
nu_02_weight=[]
nu_50_weight=[]
nu_1_diff=[]
nu_02_diff=[]
nu_50_diff=[]
nu_1_prim=[]
nu_02_prim=[]
nu_50_prim=[]
nu_1_xsec=[]
nu_02_xsec=[]
nu_50_xsec=[]


for file, nfiles in zip(name_nu_1,NFiles_nu_1):
    file_nu = tables.open_file(file)
    v_nu = variable(file_nu,2011,nfiles)
    idx_nu = function(v_nu)
    nu_1_inice = np.concatenate((nu_1_inice,v_nu.inice_primary_energy[idx_nu]))
    nu_1_zenith = np.concatenate((nu_1_zenith,v_nu.zenith[idx_nu]))
    nu_1_weight = np.concatenate((nu_1_weight,v_nu.weight[idx_nu]))
    nu_1_diff = np.concatenate((nu_1_diff,v_nu.diff[idx_nu]))
    nu_1_prim = np.concatenate((nu_1_prim,v_nu.primary_energy[idx_nu]))
    nu_1_xsec = np.concatenate((nu_1_xsec,v_nu.xsec[idx_nu]))
    file_nu.close()

# %%
print('number of raw events:',len(nu_1_zenith))
idx_nu_1_up = nu_1_zenith<seperation
idx_nu_1_down = nu_1_zenith>=seperation


inice_1_up = nu_1_inice[idx_nu_1_up]
inice_1_down = nu_1_inice[idx_nu_1_down]



weight_1_up = nu_1_weight[idx_nu_1_up]
weight_1_down = nu_1_weight[idx_nu_1_down]



xsec_1_up = nu_1_xsec[idx_nu_1_up]
xsec_1_down = nu_1_xsec[idx_nu_1_down]

diff_1_up = nu_1_diff[idx_nu_1_up]
diff_1_down = nu_1_diff[idx_nu_1_down]


prim_1_up = nu_1_prim[idx_nu_1_up]
prim_1_down = nu_1_prim[idx_nu_1_down]

plt.hist(prim_1_up,weights=weight_1_up,histtype='step',label='up')
plt.hist(prim_1_down,weights=weight_1_down,histtype='step',label='down')
plt.legend()
plt.yscale('log')
plt.xlabel('MCPrimaryEnergy')
plt.ylabel(r'Event[$livetime^{-1}$]')

plt.show()
plt.savefig('Energy_Full.png')
plt.close()
plt.hist(nu_1_zenith,weights=nu_1_weight,histtype='step',label='zenith')
plt.legend()
plt.xlabel('MCPrimaryZenith')
plt.ylabel(r'Event[$livetime^{-1}$]')
plt.show()
plt.savefig('Zenith_Full.png')