import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sys
import csv

from pandas import (read_csv, DataFrame)
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
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3] 
        self.labels = ["XMS-CASPT2","SA-CASSCF","SA-OO-VQE"]
        self.t_max = t_max

    def read_prop(self, fssh):
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
                    properties = namedtuple("properties", "dt mdsteps nstates states prob")
                    return properties(dt, mdsteps, nstates, states, prob)
                elif method == "LandauZener":
                    self.results = "prop.db"
                    properties = namedtuple("properties", "dt mdsteps nstates states")
                    return properties(timestep/self.fs, int(time_final/timestep), nstates, [i for i in range(nstates)])

    def get_popu_adi(self, fssh, filename):
        prop = self.read_prop(fssh)
        states = prop.states
        nstates = prop.nstates
        ave_time = []
        ave_popu = []
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                trajs = 0
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
                    trajs +=1
                if int(trajs-nans) == 0:
                    break
                else:
                    ave_popu.append(ref/int(trajs-nans))
        return ave_time, ave_popu

    def plot_1d_histogram_2_plots_samen_energy(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=8):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        hops_l = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(8,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # the first subplot
        hop_10 = [hop_0_10,hop_1_10,hop_2_10]
        ax0 = plt.subplot(gs[0])
        ax0.hist(hop_10, bins = bins, color=self.colors, label=self.labels)
            
        # the second subplot
        # shared axis X
        hop_01 = [hop_0_01,hop_1_01,hop_2_01]
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.hist(hop_01, bins = bins, color=self.colors, label=self.labels)

        # Set a single y-axis label for both histograms
        fig.supylabel('Number of Hops', fontweight='bold', fontsize=16)
        
        # Set labels and legends
        ax0.text(0.95, 0.9, f'(a) {hops_l[0]}', transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.9, f'(b) {hops_l[1]}', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

        plt.setp(ax0.get_xticklabels(), visible=False)

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.savefig("number_of_hops_2_samen_time.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_2_samen_time.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram_2_plots_samen(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=8):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        hops_l = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(8,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # the first subplot
        hop_10 = [hop_0_10,hop_1_10,hop_2_10]
        ax0 = plt.subplot(gs[0])
        ax0.hist(hop_10, bins = bins, color=self.colors, label=self.labels)
            
        # the second subplot
        # shared axis X
        hop_01 = [hop_0_01,hop_1_01,hop_2_01]
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.hist(hop_01, bins = bins, color=self.colors, label=self.labels)

        # Set a single y-axis label for both histograms
        fig.supylabel('Number of Hops', fontweight='bold', fontsize=16)
        
        # Set labels and legends
        ax0.text(0.95, 0.9, f'(a) {hops_l[0]}', transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.9, f'(b) {hops_l[1]}', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

        plt.setp(ax0.get_xticklabels(), visible=False)

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.savefig("number_of_hops_2_samen_time.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_2_samen_time.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram_2_plots(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=8):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        hops_l = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(8,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # the first subplot
        ax0 = plt.subplot(gs[0])
        ax0.hist(hop_0_10, bins = bins, ec = self.colors[0], label=self.labels[0] ,fc='none', lw=2)
        ax0.hist(hop_1_10, bins = bins, ec = self.colors[1], label=self.labels[1] ,fc='none', lw=2)
        ax0.hist(hop_2_10, bins = bins, ec = self.colors[2], label=self.labels[2] ,fc='none', lw=2)
            
        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.hist(hop_0_01, bins = bins, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax1.hist(hop_1_01, bins = bins, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax1.hist(hop_2_01, bins = bins, ec = self.colors[2], label="" ,fc='none', lw=2)

        # Set a single y-axis label for both histograms
        fig.supylabel('Number of Hops', fontweight='bold', fontsize=16)
        
        # Set labels and legends
        ax0.text(0.95, 0.9, f'(a) {hops_l[0]}', transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.9, f'(b) {hops_l[1]}', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

        plt.setp(ax0.get_xticklabels(), visible=False)

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        plt.xlim([0, 200])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.savefig("number_of_hops_2_time.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_2_time.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram(self,xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=8):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        plt.rcParams['font.size'] = self.fs_rcParams
        plt.xlim([self.t_0, self.t_max])
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        hops_l = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.hist(hop_0_10, bins = bins, ec = self.colors[0], label=self.labels[0] ,fc='none', lw=2)
        plt.hist(hop_1_10, bins = bins, ec = self.colors[1], label=self.labels[1] ,fc='none', lw=2)
        plt.hist(hop_2_10, bins = bins, ec = self.colors[2], label=self.labels[2] ,fc='none', lw=2)
        plt.ylabel(f'Number of Hops, {hops_l[0]}', fontweight = 'bold', fontsize = 16) 
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16) 
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=3)
        plt.savefig("number_of_hops_time.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_time.png", bbox_inches='tight')
        plt.close()

    def get_histogram_hops(self, fssh, pop_name):
        prop = self.read_prop(fssh)
        nstates = prop.nstates
        pop = read_csv(pop_name)
        time = pop['time']
        time = time.to_numpy()
        hop = pop.to_numpy()[:,1:] # removing time column
        mdsteps,trajs = hop.shape 
        hop_10 = []
        hop_01 = []
        for j in range(1,mdsteps):   #time_steps 
            for i in range(trajs):          #trajectories
                x = time[j]
                if hop[j-1,i]==1 and hop[j,i]==0:
                    hop_10.append(x)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    hop_01.append(x)
        return hop_10, hop_01

    
    def plot_population_adi(self,index,xms_caspt2,sa_casscf,sa_oo_vqe):
        time_0, population_0 = self.get_popu_adi(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        time_1, population_1 = self.get_popu_adi(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        time_2, population_2 = self.get_popu_adi(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        fig, ax = plt.subplots()
        plt.plot(time_0,np.array(population_0)[:,index], label = self.labels[0], lw=2)
        plt.plot(time_1,np.array(population_1)[:,index], label = self.labels[1], lw=2)
        plt.plot(time_2,np.array(population_2)[:,index], label = self.labels[2], lw=2)
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
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("population_adi_comb_S%i.pdf" %index, bbox_inches='tight')
        plt.savefig("population_adi_comb_S%i.png" %index, bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    #state
    index = 1 
    #paths
    xms_caspt2 = "../xms_caspt2"
    sa_oo_vqe = "../sa_oo_vqe"
    sa_casscf = "../sa_casscf"
    #time in fs
    t_0 = 0
    t_max = 200 
    out = PlotComb(t_0, t_max)
    #out.plot_population_adi(index,xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.plot_1d_histogram(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    #out.plot_1d_histogram_2_plots(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    out.plot_1d_histogram_2_plots_samen(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    

