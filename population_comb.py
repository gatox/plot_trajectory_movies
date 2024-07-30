import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
import csv

from time import sleep
from tqdm.auto import tqdm
from pandas import (read_csv, DataFrame)
from scipy.optimize import curve_fit
from scipy.stats.distributions import t

from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
from collections import (namedtuple, Counter)
from pysurf.database import PySurfDB

class PlotComb:

    def __init__(self):
        self.ev = 27.211324570273 
        self.fs = 0.02418884254
        self.aa = 0.5291772105638411 
        self.fs_rcParams = '20'
        self.t_0 = 0
        self.t_max = 200 #fs
        self.fs_ylabel = 18
        self.fs_xlabel = 18
        self.fs_yticks = 18
        self.fs_xticks = 18
        self.legend = "yes"
        self.label = "fssh"

    def read_prop(self):
        sampling = open("../test_fssh/sampling.inp", 'r+')    
        for line in sampling:
            if "n_conditions" in line:
                trajs = int(line.split()[2])
        #LVC
        spp = open("../test_fssh/spp.inp", 'r+')
        self.model = None
        for line in spp:
            if "model =" in line:
                self.model = str(line.split()[2])
        #LVC
        prop = open("../test_fssh/prop.inp", 'r+')    
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

    def get_popu_adi(self, filename):
        prop = self.read_prop()
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
    
    def plot_population_adi_S1(self):
        prop = self.read_prop()
        time, population_0 = self.get_popu_adi("../test_fssh/pop.dat")
        time, population_1 = self.get_popu_adi("../test_lz/pop.dat")
        time, population_2 = self.get_popu_adi("../test_lz_nac/pop.dat")
        time, population_3 = self.get_popu_adi("../vdf_lz_momentum/pop.dat")
        fig, ax = plt.subplots()
        plt.plot(time,np.array(population_0)[:,1], label = 'FSSH')
        plt.plot(time,np.array(population_1)[:,1], label = 'LZSH')
        plt.plot(time,np.array(population_2)[:,1], label = 'LZSH_NACs')
        plt.plot(time,np.array(population_3)[:,1], label = 'LZSH_P_REDKIN')
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{S_1\ Population}$', fontsize = 16)
        plt.legend(loc='upper right',fontsize=13, frameon=False)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), prop={'size': 15}, ncol=2)
        ax.spines['right'].set_visible(True)
        ax1 = ax.twinx()
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_adi_comb.pdf", bbox_inches='tight')
        plt.savefig("population_adi_comb.png", bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    out = PlotComb()
    out.plot_population_adi_S1()

