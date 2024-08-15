import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import (namedtuple, Counter)
from pysurf.database import PySurfDB

class PlotComb:

    def __init__(self, t_0, t_max):
        self.ev = 27.211324570273 
        self.fs = 0.02418884254
        self.aa = 0.5291772105638411 
        self.fs_rcParams = '20'
        self.t_0 = t_0
        self.t_max = t_max

    def read_prop(self, fssh):
        sampling = open(os.path.join(fssh,"sampling.inp"), 'r+')    
        for line in sampling:
            if "n_conditions" in line:
                trajs = int(line.split()[2])
        #LVC
        spp = open(os.path.join(fssh,"spp.inp"), 'r+')
        self.model = None
        for line in spp:
            if "model =" in line:
                self.model = str(line.split()[2])
        #LVC
        prop = open(os.path.join(fssh,"prop.inp"), 'r+')    
        tully = None
        for line in prop:
            if "dt" in line:
                tully = "yes"
                dt = int(float(line.split()[2]))
            elif "mdsteps" in line:
                mdsteps = int(line.split()[2])
            elif "nstates" in line:
                nstates = int(line.split()[2])
            elif "prob =" in line:
                prob = str(line.split()[2])
            #LZ
            elif "time_final" in line:
                time_final = int(line.split()[3])
            elif "timestep" in line:
                timestep = float(line.split()[3])
            elif "n_states" in line:
                nstates = int(line.split()[2])
            #LZ
            if "states" in line and not ("nstates" or "nghost_states") in line and tully is not None:
                states = [int(line.split()[2+i]) for i in range(nstates)]
            elif "method =" in line:
                method = str(line.split()[2])
                if method == "Surface_Hopping":
                    self.results = "results.db"
                    properties = namedtuple("properties", "dt mdsteps nstates states trajs prob")
                    return properties(dt, mdsteps, nstates, states, trajs, prob)
                elif method == "LandauZener":
                    self.results = "prop.db"
                    properties = namedtuple("properties", "dt mdsteps nstates states trajs")
                    return properties(timestep/self.fs, int(time_final/timestep), nstates, [i for i in range(nstates)], trajs)

    def get_popu_adi(self, fssh, filename):
        prop = self.read_prop(fssh)
        states = prop.states
        trajs = prop.trajs
        nstates = prop.nstates
        ave_time = []
        ave_popu = []
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                ref = np.zeros(nstates)
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val == 'nan':
                        nans += 1
                    else:
                        state = np.equal(float(val), states)
                        for i in range(len(states)):
                            if state[i]:
                                ref[i] += 1  
                if int(trajs-nans) == 0:
                    break
                else:
                    ave_popu.append(ref/int(trajs-nans))
        return ave_time, ave_popu
    
    def plot_population_adi(self,index,fs,lz_p,lz_nacs,lz_p_rk):
        prop = self.read_prop(fs)
        time_0, population_0 = self.get_popu_adi(fs,os.path.join(fs,"pop.dat"))
        time_1, population_1 = self.get_popu_adi(fs,os.path.join(lz_p,"pop.dat"))
        time_2, population_2 = self.get_popu_adi(fs,os.path.join(lz_nacs,"pop.dat"))
        time_3, population_3 = self.get_popu_adi(fs,os.path.join(lz_p_rk,"pop.dat"))
        fig, ax = plt.subplots()
        plt.plot(time_0,np.array(population_0)[:,index], label = 'FSSH')
        plt.plot(time_1,np.array(population_1)[:,index], label = 'LZSH_P')
        plt.plot(time_2,np.array(population_2)[:,index], label = 'LZSH_NACs')
        plt.plot(time_3,np.array(population_3)[:,index], label = 'LZSH_P_REDKIN')
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{S_%i\ Population}$' %index, fontsize = 16)
        ax.spines['right'].set_visible(True)
        if index == 0:
            plt.ylim([-0.05, 1.05])
            plt.legend(loc='lower right',fontsize=13, frameon=False)
            ax1 = ax.twinx()
            ax1.set_ylim([-0.05, 1.05])
        elif index ==1:
            plt.ylim([-0.05, 1.05])
            plt.legend(loc='upper right',fontsize=13, frameon=False)
            ax1 = ax.twinx()
            ax1.set_ylim([-0.05, 1.05])
        elif index ==2:
            plt.ylim([-0.005, 0.12])
            plt.legend(loc='upper right',fontsize=13, frameon=False)
            ax1 = ax.twinx()
            ax1.set_ylim([-0.005, 0.12])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_adi_comb_S%i.pdf" %index, bbox_inches='tight')
        plt.savefig("population_adi_comb_S%i.png" %index, bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    #state
    #index = 0
    #paths
    fs = "fssh"
    lz_p = "lz_p"
    lz_nacs = "lz_nac"
    lz_p_rk = "lz_p_rk"
    #time in fs
    t_0 = 0
    t_max = 400
    out = PlotComb(t_0, t_max)
    for i in range(3):
        out.plot_population_adi(i,fs,lz_p,lz_nacs,lz_p_rk)
    

