import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from scipy import stats
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
        self.fs_rcParams = '10'
        self.f_size = '11'
        self.t_0 = t_0
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3] 
        self.n_colors = ["purple","gold","olive"] 
        self.labels = ["XMS-CASPT2","SA-CASSCF","SA-OO-VQE"]
        self.t_max = t_max
        # Define bins
        bins_ene = np.linspace(0, 3, 16)
        #bins_hnch = np.linspace(-180, 180, 19)
        bins_hnch = np.linspace(0, 180, 19)
        bins_hnc = np.linspace(0, 180, 19)
        #bins_pyr = np.linspace(-180, 180, 19)
        bins_pyr = np.linspace(0, 180, 19)
        bins = namedtuple("bins","ene hnch hnc pyr")
        self.bins = bins(bins_ene,bins_hnch,bins_hnc,bins_pyr)

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

    def get_torsion_qy_ave_noise(self, folder):
        time_0, lower_0, upper_0, else_0 = self.get_torsion_qy_ave(os.path.join(folder,"variance_10"))
        time_1, lower_1, upper_1, else_1 = self.get_torsion_qy_ave(os.path.join(folder,"variance_08"))
        #time_2, lower_2, upper_2, else_2 = self.get_torsion_qy_ave(os.path.join(folder,"variance_06"))
        time_3, lower_3, upper_3, else_3 = self.get_torsion_qy_ave(os.path.join(folder,"variance_00"))

    def get_torsion_qy_ave_2(self, folder):
        filename = os.path.join(folder,"dihe_2014.dat")
        popu = os.path.join(folder,"pop.dat")
        ave_torsion = []
        ave_time = []
        ave_lower = []
        ave_upper = []
        ave_else = []
        with open(filename, 'r') as f1, open(popu, 'r') as f2:
            r_torsion = csv.DictReader(f1)
            r_popu = csv.DictReader(f2)
            for row_1, row_2 in zip(r_torsion, r_popu):
                ave_time.append(float(row_1['time']))
                nans = 0
                trajs = 0
                ref = 0 
                ref_non_r = 0 
                ref_rac = 0 
                ref_rest = 0 
                ref_S1 = 0
                lower_50 = 0
                upper_125 = 0
                else_ang = 0
                for k_1, val_1 in row_1.items():
                    if k_1 == 'time':
                        continue
                    if k_1 == '0':
                        tor_i = float(val_1)
                    if val_1 == 'nan':
                        nans += 1
                    else:
                        val_2 = float(row_2.get(k_1))
                        tor_f = float(val_1)
                        dis = abs(tor_f-tor_i)
                        if val_2 == 0:
                            if dis <= 30:
                                ref_non_r += abs(tor_f) 
                                lower_50 += 1
                            elif dis >= 150:
                                ref_rac += abs(tor_f) 
                                upper_125 += 1
                            else:
                                ref_rest += abs(tor_f)
                                else_ang += 1
                        else:
                            ref += abs(tor_f) 
                            ref_S1 += 1
                    trajs +=1
                if int(trajs-nans) == 0:
                    break
                else:
                    if upper_125 != 0:
                        ave_upper.append(ref_rac/int(upper_125-nans))
                    else:
                        ave_upper.append(ref/int(trajs-nans))
                    if lower_50 != 0:
                        print(lower_50, nans)
                        ave_lower.append(ref_non_r/int(lower_50-nans))
                    else:
                        ave_lower.append(ref/int(trajs-nans))
                    if else_ang != 0:
                        ave_else.append(ref_rest/else_ang)
                    else:
                        ave_else.append(ref/int(trajs-nans))
        if "noise_sa_oo_vqe" in folder:
            title = folder.replace('../noise_sa_oo_vqe/', '')  
        else:
            title = folder.replace('../', '')
        with open(f'QY_information_{title}_2.out', 'w') as f3:
            f3.write('--------------------------------------------------------------\n')
            f3.write(f'Folder: {title}\n')
            f3.write(f'lower_50/{trajs-nans}: {lower_50/int(trajs-nans)}\n')
            f3.write(f'lower_50/(lower_50+upper_125): {lower_50/(lower_50+upper_125)}\n')
            f3.write(f'upper_125/{trajs-nans}: {upper_125/int(trajs-nans)}\n')
            f3.write(f'upper_125/(lower_50+upper_125): {upper_125/(lower_50+upper_125)}\n')
            f3.write(f'lower_S0_50 = {lower_50}, upper_S0_125 = {upper_125}, rest_S0 = {else_ang}, S1 = {ref_S1}\n')
            f3.write(f'Total:  {lower_50 + upper_125 + else_ang + ref_S1}\n')
            f3.write(f'Trajs - Nans: {int(trajs-nans)}\n')
            f3.write('--------------------------------------------------------------')
            f3.close()
        return ave_time, ave_lower, ave_upper, ave_else

    def get_torsion_qy_ave(self, folder):
        filename = os.path.join(folder,"dihe_2014.dat")
        popu = os.path.join(folder,"pop.dat")
        ave_torsion = []
        ave_time = []
        ave_lower = []
        ave_upper = []
        ave_else = []
        with open(filename, 'r') as f1, open(popu, 'r') as f2:
            r_torsion = csv.DictReader(f1)
            r_popu = csv.DictReader(f2)
            for row_1, row_2 in zip(r_torsion, r_popu):
                ave_time.append(float(row_1['time']))
                nans = 0
                trajs = 0
                ref = 0 
                ref_non_r = 0 
                ref_rac = 0 
                ref_rest = 0 
                ref_S1 = 0
                lower_30 = 0
                upper_150 = 0
                else_ang = 0
                for k_1, val_1 in row_1.items():
                    if k_1 == 'time':
                        continue
                    if val_1 == 'nan':
                        nans += 1
                    else:
                        val_2 = float(row_2.get(k_1))
                        ref_abs = abs(float(val_1))
                        if val_2 == 0:
                            if ref_abs <= 30:
                                ref_non_r += ref_abs 
                                lower_30 += 1
                            elif ref_abs >= 150:
                                ref_rac += ref_abs 
                                upper_150 += 1
                            else:
                                ref_rest += ref_abs
                                else_ang += 1
                        else:
                            ref += ref_abs 
                            ref_S1 += 1
                    trajs +=1
                if int(trajs-nans) == 0:
                    break
                else:
                    if upper_150 != 0:
                        ave_upper.append(ref_rac/int(upper_150-nans))
                    else:
                        ave_upper.append(ref/int(trajs-nans))
                    if lower_30 != 0:
                        print(lower_30, nans)
                        ave_lower.append(ref_non_r/int(lower_30-nans))
                    else:
                        ave_lower.append(ref/int(trajs-nans))
                    if else_ang != 0:
                        ave_else.append(ref_rest/else_ang)
                    else:
                        ave_else.append(ref/int(trajs-nans))
        if "noise_sa_oo_vqe" in folder:
            title = folder.replace('../noise_sa_oo_vqe/', '')  
        else:
            title = folder.replace('../', '')
        with open(f'QY_information_30_150_{title}.out', 'w') as f3:
            f3.write('--------------------------------------------------------------\n')
            f3.write(f'Folder: {title}\n')
            f3.write(f'lower_30/{trajs-nans}: {lower_30/int(trajs-nans)}\n')
            f3.write(f'lower_30/(lower_30+upper_150): {lower_30/(lower_30+upper_150)}\n')
            f3.write(f'upper_150/{trajs-nans}: {upper_150/int(trajs-nans)}\n')
            f3.write(f'upper_150/(lower_30+upper_150): {upper_150/(lower_30+upper_150)}\n')
            f3.write(f'lower_S0_50 = {lower_30}, upper_S0_150 = {upper_150}, rest_S0 = {else_ang}, S1 = {ref_S1}\n')
            f3.write(f'Total:  {lower_30 + upper_150 + else_ang + ref_S1}\n')
            f3.write(f'Trajs - Nans: {int(trajs-nans)}\n')
            f3.write('--------------------------------------------------------------')
            f3.close()
        return ave_time, ave_lower, ave_upper, ave_else

    def get_bend_ave(self, folder):
        filename = os.path.join(folder,"dihe_2014.dat")
        ave_time = []
        ave_torsion = []
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                trajs = 0
                ref = 0 
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val == 'nan':
                        nans += 1
                    else:
                        ref += abs(float(val))
                    trajs +=1
                if int(trajs-nans) == 0:
                    break
                else:
                    ave_torsion.append(ref/int(trajs-nans))
        return ave_time, ave_torsion

    #def get_torsion_ave(self, folder):
    #    filename = os.path.join(folder,"dihe_2014.dat")
    #    ave_time = []
    #    ave_torsion = []
    #    torsion_data = []
    #    with open(filename, 'r') as fh:
    #        reader = csv.DictReader(fh)
    #        for row in reader:
    #            ave_time.append(float(row['time']))
    #            nans = 0
    #            trajs = 0
    #            ref = 0 
    #            for k, val in row.items():
    #                if k == 'time':
    #                    continue
    #                if val == 'nan':
    #                    nans += 1
    #                else:
    #                    ref += abs(float(val))
    #                trajs +=1
    #            if int(trajs-nans) == 0:
    #                break
    #            else:
    #                ave_torsion.append(np.mean(ref))
    #                torsion_data.append(ref)          # Store the torsion data for this timestep
    #                
    #    torsion_data = np.array(torsion_data)       # Convert to numpy array for easy manipulation
    #    torsion_std = np.std(torsion_data, axis=1)  # Calculate standard deviation across trajectories
    #    
    #    return ave_time, ave_torsion, torsion_std

    def get_parameter_ave(self, folder, data):
        filename = os.path.join(folder, data)
        ave_time = []
        ave_para = []
        para_data = []  # Store all parameter values across time steps for each trajectory
        
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                para_vals = []
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val != 'nan':  # Only consider valid (non-'nan') values
                        para_vals.append(abs(float(val)))
                
                if len(para_vals) > 0:  # If we have valid para values
                    ave_para.append(np.mean(para_vals))  # Compute average
                    para_data.append(para_vals)  # Store the para values for std calculation
        
        # Convert para_data into a numpy array (2D array: rows -> time steps, columns -> trajectories)
        para_data = np.array([np.pad(t, (0, max(len(x) for x in para_data) - len(t)), constant_values=np.nan) for t in para_data])
        
        # Compute standard deviation along axis=1 (time axis)
        para_std = np.nanstd(para_data, axis=1)  # Use np.nanstd to ignore NaN values
        
        return ave_time, ave_para, para_std

    def get_noise_ave(self, folder, noise):
        filename = os.path.join(folder, noise)
        ave_time = []
        ave_noise = []
        noise_data = []  # Store all noise values across time steps for each trajectory
        
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                noise_vals = []
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val != 'nan':  # Only consider valid (non-'nan') values
                        noise_vals.append(abs(float(val)))
                
                if len(noise_vals) > 0:  # If we have valid noise values
                    ave_noise.append(np.mean(noise_vals))  # Compute average
                    noise_data.append(noise_vals)  # Store the noise values for std calculation
        
        # Convert noise_data into a numpy array (2D array: rows -> time steps, columns -> trajectories)
        noise_data = np.array([np.pad(t, (0, max(len(x) for x in noise_data) - len(t)), constant_values=np.nan) for t in noise_data])
        
        # Compute standard deviation along axis=1 (time axis)
        noise_std = np.nanstd(noise_data, axis=1)  # Use np.nanstd to ignore NaN values
        
        return ave_time, ave_noise, noise_std

    def get_torsion_ave(self, folder):
        filename = os.path.join(folder, "dihe_2014.dat")
        ave_time = []
        ave_torsion = []
        torsion_data = []  # Store all torsion values across time steps for each trajectory
        
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                torsion_vals = []
                for k, val in row.items():
                    if k == 'time':
                        continue
                    if val != 'nan':  # Only consider valid (non-'nan') values
                        torsion_vals.append(abs(float(val)))
                
                if len(torsion_vals) > 0:  # If we have valid torsion values
                    ave_torsion.append(np.mean(torsion_vals))  # Compute average
                    torsion_data.append(torsion_vals)  # Store the torsion values for std calculation
        
        # Convert torsion_data into a numpy array (2D array: rows -> time steps, columns -> trajectories)
        torsion_data = np.array([np.pad(t, (0, max(len(x) for x in torsion_data) - len(t)), constant_values=np.nan) for t in torsion_data])
        
        # Compute standard deviation along axis=1 (time axis)
        torsion_std = np.nanstd(torsion_data, axis=1)  # Use np.nanstd to ignore NaN values
        
        return ave_time, ave_torsion, torsion_std

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

    def plot_1d_histogram_2_plots_samen_energy(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=20):
        hop_0_10, hop_0_01 = self.get_histogram_hops_energy(xms_caspt2)
        hop_1_10, hop_1_01 = self.get_histogram_hops_energy(sa_casscf)
        hop_2_10,hop_2_01 = self.get_histogram_hops_energy(sa_oo_vqe)
        bins = [x for x in np.linspace(0, 3, 21)]
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
        plt.xlim([0, 3])
        plt.xlabel('Energy Gap (eV)', fontweight = 'bold', fontsize = 16)
        plt.savefig("number_of_hops_2_samen_energy.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_2_samen_energy.png", bbox_inches='tight')
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

    def plot_1d_histogram_2_plots_energy(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=31):
        hop_0_10, hop_0_01 = self.get_histogram_hops_energy(xms_caspt2)
        hop_1_10, hop_1_01 = self.get_histogram_hops_energy(sa_casscf)
        hop_2_10,hop_2_01 = self.get_histogram_hops_energy(sa_oo_vqe)
        bins = [x for x in np.linspace(0, 3, n_bins)]
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
        plt.xlim([0, 3])
        plt.xlabel('Energy Gap (eV)', fontweight = 'bold', fontsize = 16)
        plt.savefig("number_of_hops_2_energy.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_2_energy.png", bbox_inches='tight')
        plt.close()

    def plot_2d_histogram_QY_time(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=16):
        #Grond state
        tor_0_0 = self.get_histogram_qy(xms_caspt2,0)
        tor_1_0 = self.get_histogram_qy(sa_casscf,0)
        tor_2_0 = self.get_histogram_qy(sa_oo_vqe,0)
        #First state
        tor_0_1 = self.get_histogram_qy(xms_caspt2,1)
        tor_1_1 = self.get_histogram_qy(sa_casscf,1)
        tor_2_1 = self.get_histogram_qy(sa_oo_vqe,1)
        bins = np.linspace(0, 180, n_bins) 
        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # the first subplot
        ax0 = plt.subplot(gs[0])
        ax0.hist(tor_0_1, bins=bins, ec=self.colors[0], label=self.labels[0], fc='none', lw=2)
        ax0.hist(tor_1_1, bins=bins, ec=self.colors[1], label=self.labels[1], fc='none', lw=2)
        ax0.hist(tor_2_1, bins=bins, ec=self.colors[2], label=self.labels[2], fc='none', lw=2)
        
        # Set limits and labels
        ax0.set_xlim([0, 180])
        ax0.set_ylim([0, 200])
        ax0.set_ylabel('Number of Initial Torsion Angles', fontweight='bold', fontsize=self.f_size)
        
        # Set major locator for x-axis
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(30))

        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.hist(tor_0_0, bins=bins, ec=self.colors[0], label=self.labels[0], fc='none', lw=2)
        ax1.hist(tor_1_0, bins=bins, ec=self.colors[1], label=self.labels[1], fc='none', lw=2)
        ax1.hist(tor_2_0, bins=bins, ec=self.colors[2], label=self.labels[2], fc='none', lw=2)

        # Set limits and labels
        ax1.set_xlim([0, 180])
        ax1.set_ylim([0, 58])
        ax1.set_ylabel('Number of Final Torsion Angles', fontweight='bold', fontsize=self.f_size)
        ax1.set_xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        
        # Set major locator for x-axis
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(30))

        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

        plt.setp(ax0.get_xticklabels(), visible=False)

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.48, 1.2), prop={'size': 12}, ncol=3)
        
        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)
        plt.savefig("number_of_dihe_qy.pdf", bbox_inches='tight')
        plt.savefig("number_of_dihe_qy.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram_QY_time(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=16):
        tor_0_0 = self.get_histogram_qy(xms_caspt2,0)
        tor_1_0 = self.get_histogram_qy(sa_casscf,0)
        tor_2_0 = self.get_histogram_qy(sa_oo_vqe,0)
        bins = np.linspace(0, 180, n_bins) 

        # Create figure and axis
        fig, ax = plt.subplots()
        plt.rcParams['font.size'] = self.fs_rcParams
        
        # Plot histograms on the axis
        ax.hist(tor_0_0, bins=bins, ec=self.colors[0], label=self.labels[0], fc='none', lw=2)
        ax.hist(tor_1_0, bins=bins, ec=self.colors[1], label=self.labels[1], fc='none', lw=2)
        ax.hist(tor_2_0, bins=bins, ec=self.colors[2], label=self.labels[2], fc='none', lw=2)
        
        # Set limits and labels
        ax.set_xlim([0, 180])
        ax.set_xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        ax.set_ylabel('Number of Final Torsion Angles', fontweight='bold', fontsize=self.f_size)
        
        # Set major locator for x-axis
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

        # put legend on first subplot
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=3)
        
        # Save and close the plot
        plt.savefig("number_of_dihe_qy.pdf", bbox_inches='tight')
        plt.savefig("number_of_dihe_qy.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram_2_plots(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=16):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2)
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf)
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe)
        #bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        bins = np.linspace(0, 200, n_bins) 
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

    def _index(self, val, array):
        index = -1
        for i in range(array.size):
            if array[i] == val:
                index = i
                break
        return i
        

    def _calculate_stats(self, data, bins):
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        max_val = bin_centers[np.argmax(hist)]  # Maximum (Mode)
        return {
            'max': max_val,
            'his': hist, 
            'bin_centers':bin_centers,
            'bin_edges':bin_edges,
        }

    
    def print_stat(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        energy = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"e_gap.dat") 
        torsion = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"dihe_2014.dat")
        bending = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"angle_014.dat")
        pyramidal = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"pyr_3210.dat")
        pars = namedtuple("pars", "ene hnch hnc pyr")
        pars = pars(energy,torsion,bending,pyramidal)
        for par in pars._fields:
            par_method = getattr(pars, par)
            for method in par_method._fields:  
                res = self._calculate_stats(getattr(par_method,method), getattr(self.bins,par))
                with open(f'Analysis_{method}_{par}', 'w') as f:
                    for key, value in res.items():
                        # Write key and value to the file
                        f.write(f'{key}: {value}\n')
        f.close()

    def _hop_10(self, xms_caspt2, sa_casscf, sa_oo_vqe, data):
        hop_10_xms, hop_01_xms = self.get_histogram_hops_energy(xms_caspt2, data)
        hop_10_cas, hop_01_cas = self.get_histogram_hops_energy(sa_casscf, data)
        hop_10_vqe, hop_01_vqe = self.get_histogram_hops_energy(sa_oo_vqe, data)
        hops = namedtuple("hops","xms cas vqe") 
        return hops(hop_10_xms,hop_10_cas,hop_10_vqe)

    def _one_para(self, method, data):
        time_met, ave_met, std_met = self.get_parameter_ave(method, data)
        para = namedtuple("para","t_met av_met std_met") 
        return para(time_met,ave_met,std_met)

    def _para(self, xms_caspt2, sa_casscf, sa_oo_vqe, data):
        time_xms, ave_xms, std_xms = self.get_parameter_ave(xms_caspt2, data)
        time_cas, ave_cas, std_cas = self.get_parameter_ave(sa_casscf, data)
        time_vqe, ave_vqe, std_vqe = self.get_parameter_ave(sa_oo_vqe, data)
        para = namedtuple("para","t_xms av_xms std_xms t_cas av_cas std_cas t_vqe av_vqe std_vqe") 
        return para(time_xms,ave_xms,std_xms,time_cas,ave_cas,std_cas,time_vqe,ave_vqe,std_vqe)

    def plot_av_popu_torsion_bend(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        #popu
        time_0, population_0 = self.get_popu_adi(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        time_1, population_1 = self.get_popu_adi(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        time_2, population_2 = self.get_popu_adi(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        #dihe_2014
        dihe = self._para(xms_caspt2,sa_casscf,sa_oo_vqe,"dihe_2014.dat")
        #angle_014
        bend = self._para(xms_caspt2,sa_casscf,sa_oo_vqe,"angle_014.dat")
        #pyr_3210
        pyr = self._para(xms_caspt2,sa_casscf,sa_oo_vqe,"pyr_3210.dat")

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_0,np.array(population_0)[:,1], label = self.labels[0], lw=2)
        ax0.plot(time_1,np.array(population_1)[:,1], label = self.labels[1], lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], label = self.labels[2], lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))
            
        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(dihe.t_xms,dihe.av_xms, lw=2)
        ax1.plot(dihe.t_cas,dihe.av_cas, lw=2)
        ax1.plot(dihe.t_vqe,dihe.av_vqe, lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-8, 185])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(dihe.t_xms, np.array(dihe.av_xms) - np.array(dihe.std_xms),
                         np.array(dihe.av_xms) + np.array(dihe.std_xms), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        ax1.fill_between(dihe.t_cas, np.array(dihe.av_cas) - np.array(dihe.std_cas),
                         np.array(dihe.av_cas) + np.array(dihe.std_cas), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        ax1.fill_between(dihe.t_vqe, np.array(dihe.av_vqe) - np.array(dihe.std_vqe),
                         np.array(dihe.av_vqe) + np.array(dihe.std_vqe), alpha=0.3, linestyle=':', edgecolor=self.colors[2])
        ax1.set_ylim([-8,185])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # the 3rd subplot
        ax2 = plt.subplot(gs[2], sharex = ax0)
        ax2.plot(bend.t_xms,bend.av_xms, lw=2)
        ax2.plot(bend.t_cas,bend.av_cas, lw=2)
        ax2.plot(bend.t_vqe,bend.av_vqe, lw=2)
        ax2r = ax2.twinx()
        ax2r.set_ylim([53, 185])
        ax2r.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax2.fill_between(bend.t_xms, np.array(bend.av_xms) - np.array(bend.std_xms),
                         np.array(bend.av_xms) + np.array(bend.std_xms), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        ax2.fill_between(bend.t_cas, np.array(bend.av_cas) - np.array(bend.std_cas),
                         np.array(bend.av_cas) + np.array(bend.std_cas), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        ax2.fill_between(bend.t_vqe, np.array(bend.av_vqe) - np.array(bend.std_vqe),
                         np.array(bend.av_vqe) + np.array(bend.std_vqe), alpha=0.3, linestyle=':', edgecolor=self.colors[2])
        ax2.set_ylim([53,185])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.set_ylabel('$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # the 4th subplot
        ax3 = plt.subplot(gs[3], sharex = ax0)
        ax3.plot(pyr.t_xms,pyr.av_xms, lw=2)
        ax3.plot(pyr.t_cas,pyr.av_cas, lw=2)
        ax3.plot(pyr.t_vqe,pyr.av_vqe, lw=2)
        ax3r = ax3.twinx()
        ax3r.set_ylim([-8, 95])
        ax3r.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax3r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax3.fill_between(pyr.t_xms, np.array(pyr.av_xms) - np.array(pyr.std_xms),
                         np.array(pyr.av_xms) + np.array(pyr.std_xms), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        ax3.fill_between(pyr.t_cas, np.array(pyr.av_cas) - np.array(pyr.std_cas),
                         np.array(pyr.av_cas) + np.array(pyr.std_cas), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        ax3.fill_between(pyr.t_vqe, np.array(pyr.av_vqe) - np.array(pyr.std_vqe),
                         np.array(pyr.av_vqe) + np.array(pyr.std_vqe), alpha=0.3, linestyle=':', edgecolor=self.colors[2])
        ax3.set_ylim([-8,95])
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax3.set_ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize=self.f_size)
        ax3.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax2.text(0.95, 0.95, f'(c)', transform=ax2.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax3.text(0.95, 0.95, f'(d)', transform=ax3.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        plt.savefig("avg_popu_dihe_bend_pyr.pdf", bbox_inches='tight')
        plt.savefig("avg_popu_dihe_bend_pyr.png", bbox_inches='tight')
        plt.close()

    def plot_one_method_av_popu_diff_ene(self,method):
        #popu
        time_0, population_0 = self.get_popu_adi(method,os.path.join(method,"pop.dat"))
        #diff_ene
        d_ene = self._one_para(method,"etot.dat")

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_0,np.array(population_0)[:,1], label = self.labels[2], lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))
            
        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(d_ene.t_met,d_ene.av_met, lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-0.05, 1])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(d_ene.t_met, np.array(d_ene.av_met) - np.array(d_ene.std_met),
                         np.array(d_ene.av_met) + np.array(d_ene.std_met), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        ax1.set_ylim([-0.05, 1])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1.set_ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize=self.f_size)
        ax1.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        plt.savefig("av_one_met_popu_diff_ene.pdf", bbox_inches='tight')
        plt.savefig("av_one_met_popu_diff_ene.png", bbox_inches='tight')
        plt.close()

    def plot_av_popu_diff_ene(self,xms_caspt2,sa_casscf,sa_oo_vqe):
        #popu
        time_0, population_0 = self.get_popu_adi(xms_caspt2,os.path.join(xms_caspt2,"pop.dat"))
        time_1, population_1 = self.get_popu_adi(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        time_2, population_2 = self.get_popu_adi(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        #diff_ene
        d_ene = self._para(xms_caspt2,sa_casscf,sa_oo_vqe,"etot.dat")

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_0,np.array(population_0)[:,1], label = self.labels[0], lw=2)
        ax0.plot(time_1,np.array(population_1)[:,1], label = self.labels[1], lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], label = self.labels[2], lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))
            
        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(d_ene.t_xms,d_ene.av_xms, lw=2)
        ax1.plot(d_ene.t_cas,d_ene.av_cas, lw=2)
        ax1.plot(d_ene.t_vqe,d_ene.av_vqe, lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-0.05, 1])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(d_ene.t_xms, np.array(d_ene.av_xms) - np.array(d_ene.std_xms),
                         np.array(d_ene.av_xms) + np.array(d_ene.std_xms), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        ax1.fill_between(d_ene.t_cas, np.array(d_ene.av_cas) - np.array(d_ene.std_cas),
                         np.array(d_ene.av_cas) + np.array(d_ene.std_cas), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        ax1.fill_between(d_ene.t_vqe, np.array(d_ene.av_vqe) - np.array(d_ene.std_vqe),
                         np.array(d_ene.av_vqe) + np.array(d_ene.std_vqe), alpha=0.3, linestyle=':', edgecolor=self.colors[2])
        ax1.set_ylim([-0.05, 1])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1.set_ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize=self.f_size)
        ax1.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=3)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        plt.savefig("av_popu_diff_ene.pdf", bbox_inches='tight')
        plt.savefig("av_popu_diff_ene.png", bbox_inches='tight')
        plt.close()

    def plot_av_popu_torsion_noise(self, folder):
        #popu
        time_0, population_0 = self.get_popu_adi(folder,os.path.join(folder,"variance_10/pop.dat"))
        time_1, population_1 = self.get_popu_adi(folder,os.path.join(folder,"variance_08/pop.dat"))
        time_2, population_2 = self.get_popu_adi(folder,os.path.join(folder,"variance_06/pop.dat"))
        time_3, population_3 = self.get_popu_adi(folder,os.path.join(folder,"variance_00/pop.dat"))
        #torsion
        time_0, t_noise_0, t_std_0 = self.get_noise_ave(folder,'variance_10/dihe_2014.dat')
        time_1, t_noise_1, t_std_1 = self.get_noise_ave(folder,'variance_08/dihe_2014.dat')
        time_2, t_noise_2, t_std_2 = self.get_noise_ave(folder,'variance_06/dihe_2014.dat')
        time_3, t_noise_3, t_std_3 = self.get_noise_ave(folder,'variance_00/dihe_2014.dat')
        #noise
        time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_10/etot.dat')
        time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_06/etot.dat')
        time_3, noise_3, std_3 = self.get_noise_ave(folder,'variance_00/etot.dat')

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_3,np.array(population_3)[:,1], color = "blue", label = "no noise", lw=2, alpha=0.8)
        ax0.plot(time_0,np.array(population_0)[:,1], color = self.n_colors[0], label = r"$\sigma^2$=1.0e-10", lw=2)
        ax0.plot(time_1,np.array(population_1)[:,1], color = self.n_colors[1], label = r"$\sigma^2$=1.0e-08", lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], color = self.n_colors[2], label = r"$\sigma^2$=1.0e-06", lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))

        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(time_3, t_noise_3, color = "blue", lw=2, alpha=0.8)
        ax1.plot(time_0, t_noise_0, color = self.n_colors[0], lw=2)
        ax1.plot(time_1, t_noise_1, color = self.n_colors[1], lw=2)
        ax1.plot(time_2, t_noise_2, color = self.n_colors[2], lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-8, 185])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(time_3, np.array(t_noise_3) - np.array(t_std_3),
                         np.array(t_noise_3) + np.array(t_std_3), alpha=0.3, color = "blue", linestyle=':', edgecolor="blue")
        ax1.fill_between(time_0, np.array(t_noise_0) - np.array(t_std_0),
                         np.array(t_noise_0) + np.array(t_std_0), alpha=0.3, color = self.n_colors[0], linestyle='-.', edgecolor=self.n_colors[0])
        ax1.fill_between(time_1, np.array(t_noise_1) - np.array(t_std_1),
                         np.array(t_noise_1) + np.array(t_std_1), alpha=0.3, color = self.n_colors[1], linestyle='--', edgecolor=self.n_colors[1])
        ax1.fill_between(time_2, np.array(t_noise_2) - np.array(t_std_2),
                         np.array(t_noise_2) + np.array(t_std_2), alpha=0.3, color = self.n_colors[2], linestyle=':', edgecolor=self.n_colors[2])
        ax1.set_ylim([-8,185])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)
            
        # the 3rd subplot
        ax2 = plt.subplot(gs[2], sharex = ax0)
        ax2.plot(time_3, noise_3, color = "blue", lw=2, alpha=0.8)
        ax2.plot(time_0, noise_0, color = self.n_colors[0], lw=2)
        ax2.plot(time_1, noise_1, color = self.n_colors[1], lw=2)
        ax2.plot(time_2, noise_2, color = self.n_colors[2], lw=2)
        ax2r = ax2.twinx()
        ax2r.set_ylim([-0.05, 2.37])
        ax2r.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax2r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax2.fill_between(time_3, np.array(noise_3) - np.array(std_3),
                         np.array(noise_3) + np.array(std_3), alpha=0.3, color = "blue", linestyle=':', edgecolor="blue")
        ax2.fill_between(time_0, np.array(noise_0) - np.array(std_0),
                         np.array(noise_0) + np.array(std_0), alpha=0.3, color = self.n_colors[0], linestyle='-.', edgecolor=self.n_colors[0])
        ax2.fill_between(time_1, np.array(noise_1) - np.array(std_1),
                         np.array(noise_1) + np.array(std_1), alpha=0.3, color = self.n_colors[1], linestyle='--', edgecolor=self.n_colors[1])
        ax2.fill_between(time_2, np.array(noise_2) - np.array(std_2),
                         np.array(noise_2) + np.array(std_2), alpha=0.3, color = self.n_colors[2], linestyle=':', edgecolor=self.n_colors[2])
        ax2.set_ylim([-0.05, 2.37])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax2.set_ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize=self.f_size)
        ax2.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax2.text(0.95, 0.95, f'(c)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')


        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), prop={'size': 12}, ncol=2)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        plt.savefig("avg_popu_torsion_noise.pdf", bbox_inches='tight')
        plt.savefig("avg_popu_torsion_noise.png", bbox_inches='tight')
        plt.close()

    def plot_av_popu_noise(self, folder):
        #popu
        time_0, population_0 = self.get_popu_adi(folder,os.path.join(folder,"variance_10/pop.dat"))
        time_1, population_1 = self.get_popu_adi(folder,os.path.join(folder,"variance_08/pop.dat"))
        time_2, population_2 = self.get_popu_adi(folder,os.path.join(folder,"variance_06/pop.dat"))
        time_3, population_3 = self.get_popu_adi(folder,os.path.join(folder,"variance_00/pop.dat"))
        #noise
        time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_10/etot.dat')
        time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_06/etot.dat')
        time_3, noise_3, std_3 = self.get_noise_ave(folder,'variance_00/etot.dat')

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_3,np.array(population_3)[:,1], color = "blue", label = "no noise", lw=2, alpha=0.8)
        ax0.plot(time_0,np.array(population_0)[:,1], color = self.n_colors[0], label = r"$\sigma^2$=1.0e-10", lw=2)
        ax0.plot(time_1,np.array(population_1)[:,1], color = self.n_colors[1], label = r"$\sigma^2$=1.0e-08", lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], color = self.n_colors[2], label = r"$\sigma^2$=1.0e-06", lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))
            
        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(time_3, noise_3, color = "blue", lw=2, alpha=0.8)
        ax1.plot(time_0, noise_0, color = self.n_colors[0], lw=2)
        ax1.plot(time_1, noise_1, color = self.n_colors[1], lw=2)
        ax1.plot(time_2, noise_2, color = self.n_colors[2], lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-0.05, 2.37])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(time_3, np.array(noise_3) - np.array(std_3),
                         np.array(noise_3) + np.array(std_3), alpha=0.3, color = "blue", linestyle=':', edgecolor="blue")
        ax1.fill_between(time_0, np.array(noise_0) - np.array(std_0),
                         np.array(noise_0) + np.array(std_0), alpha=0.3, color = self.n_colors[0], linestyle='-.', edgecolor=self.n_colors[0])
        ax1.fill_between(time_1, np.array(noise_1) - np.array(std_1),
                         np.array(noise_1) + np.array(std_1), alpha=0.3, color = self.n_colors[1], linestyle='--', edgecolor=self.n_colors[1])
        ax1.fill_between(time_2, np.array(noise_2) - np.array(std_2),
                         np.array(noise_2) + np.array(std_2), alpha=0.3, color = self.n_colors[2], linestyle=':', edgecolor=self.n_colors[2])
        ax1.set_ylim([-0.05, 2.37])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax1.set_ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize=self.f_size)
        ax1.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')

        # put legend on first subplot
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), prop={'size': 12}, ncol=2)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        plt.savefig("avg_popu_noise_4.pdf", bbox_inches='tight')
        plt.savefig("avg_popu_noise_4.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram_4_plots_S1_S0(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        #e_gap
        hop_e = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"e_gap.dat")
        #dihe_2014
        hop_d = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"dihe_2014.dat")
        #angle_014
        hop_a = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"angle_014.dat")
        #pyr_3210
        hop_p = self._hop_10(xms_caspt2,sa_casscf,sa_oo_vqe,"pyr_3210.dat")

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(10,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        # the 1st subplot
        ax00 = plt.subplot(gs[0,0])
        ax00.hist(hop_e.xms, bins = self.bins.ene, ec = self.colors[0], label=self.labels[0] ,fc='none', lw=2)
        ax00.hist(hop_e.cas, bins = self.bins.ene, ec = self.colors[1], label=self.labels[1] ,fc='none', lw=2)
        ax00.hist(hop_e.vqe, bins = self.bins.ene, ec = self.colors[2], label=self.labels[2] ,fc='none', lw=2)
        ax00.set_xlim([0,3])
        ax00.xaxis.set_major_locator(ticker.MultipleLocator(0.6))
        ax00.set_xlabel('Energy Gap (eV)', fontweight='bold', fontsize= self.f_size)
            
        # the 2nd subplot
        ax01 = plt.subplot(gs[0,1])
        ax01.hist(hop_d.xms, bins = self.bins.hnch, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax01.hist(hop_d.cas, bins = self.bins.hnch, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax01.hist(hop_d.vqe, bins = self.bins.hnch, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax01.set_xlim([0,180])
        ax01.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax01.set_xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)

        # the 3rd subplot
        ax10 = plt.subplot(gs[1,0])
        ax10.hist(hop_a.xms, bins = self.bins.hnc, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax10.hist(hop_a.cas, bins = self.bins.hnc, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax10.hist(hop_a.vqe, bins = self.bins.hnc, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax10.set_xlim([0,180])
        ax10.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax10.set_xlabel('$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', fontsize=self.f_size)

        # the 4th subplot
        ax11 = plt.subplot(gs[1,1])
        ax11.hist(hop_p.xms, bins = self.bins.pyr, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax11.hist(hop_p.cas, bins = self.bins.pyr, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax11.hist(hop_p.vqe, bins = self.bins.pyr, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax11.set_xlim([0,180])
        ax11.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax11.set_xlabel('$\mathbf{Pyramidalization (degrees)}$', fontsize=self.f_size)

        # Set a single y-axis label for both histograms
        fig.supylabel('Number of Hops', fontweight='bold', fontsize=18)
        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax00.text(0.95, 0.95, f'(a)', transform=ax00.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax01.text(0.95, 0.95, f'(b)', transform=ax01.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax10.text(0.95, 0.95, f'(c)', transform=ax10.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax11.text(0.95, 0.95, f'(d)', transform=ax11.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')

        # put legend on first subplot
        ax00.legend(loc='upper center', bbox_to_anchor=(1, 1.2), prop={'size': 14}, ncol=3)

        plt.savefig("number_of_hops_4.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_4.png", bbox_inches='tight')
        plt.close()

    def plot_1d_histogram(self,xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=8):
        hop_0_10, hop_0_01 = self.get_histogram_hops(xms_caspt2)
        hop_1_10, hop_1_01 = self.get_histogram_hops(sa_casscf)
        hop_2_10,hop_2_01 = self.get_histogram_hops(sa_oo_vqe)
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

    def get_histogram_hops(self, folder):
        pop_name = os.path.join(folder,"pop.dat")
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

    def get_histogram_qy(self, folder, state):
        pop_name = os.path.join(folder,"pop.dat")
        torsion_name = os.path.join(folder,"dihe_2014.dat")
        pop = read_csv(pop_name)
        torsion = read_csv(torsion_name)
        cur = pop.to_numpy()[:,1:] # removing time column
        tor = torsion.to_numpy()[:,1:] # removing time column
        mdsteps,trajs = cur.shape 
        torsion_0 = []
        for i in range(trajs):          #trajectories
            if state == 0:
                dihe = tor[mdsteps-1,i] 
                if cur[mdsteps-1,i]==0:
                    torsion_0.append(abs(dihe))
            elif state == 1:
                dihe = tor[0,i] 
                if cur[0,i]==1:
                    torsion_0.append(abs(dihe))
        return torsion_0 

    def get_histogram_hops_energy(self, folder, parameter):
        if "xms_caspt2" in folder:
            print("Average values for xms_caspt2")
        if "sa_casscf" in folder:
            print("Average values for sa_casscf")
        if "sa_oo_vqe" in folder:
            print("Average values for sa_oo_vqe")
        e_gap_name = os.path.join(folder,parameter)
        pop_name = os.path.join(folder,"pop.dat")
        e_gap = read_csv(e_gap_name)
        pop = read_csv(pop_name)
        hop = pop.to_numpy()[:,1:] # removing time column
        ene_d = e_gap.to_numpy()[:,1:] # removing time column
        mdsteps,trajs = hop.shape 
        hop_10 = []
        hop_01 = []
        hop_10_hnch_lower = []
        hop_10_hnch_upper = []
        hop_10_hnc = []
        hop_10_pyr = []
        for j in range(1,mdsteps):   #time_steps 
            for i in range(trajs):          #trajectories
                ene = ene_d[j,i] 
                if hop[j-1,i]==1 and hop[j,i]==0:
                    hop_10.append(abs(ene))
                    if parameter == "dihe_2014.dat" and ene < 0:
                        hop_10_hnch_lower.append(ene)
                    elif parameter == "dihe_2014.dat" and ene > 0:
                        hop_10_hnch_upper.append(ene)
                    elif parameter == "angle_014.dat":
                        hop_10_hnc.append(ene)
                    elif parameter == "pyr_3210.dat":
                        hop_10_pyr.append(ene)
                elif hop[j-1,i]==0 and hop[j,i]==1:
                    hop_01.append(ene)
                    #if parameter == "angle_014.dat":
                    #    hop_10_hnc.append(ene)
                    #elif parameter == "pyr_3210.dat":
                    #    hop_10_pyr.append(abs(ene))
        if parameter in ["dihe_2014.dat"]:
            print(f"Average hnch < 0:", sum(hop_10_hnch_lower)/len(hop_10_hnch_lower))
            print(f"Average hnch > 0:", sum(hop_10_hnch_upper)/len(hop_10_hnch_upper))
            print(f"Average abs(hnch):", sum(hop_10)/len(hop_10))
        elif parameter in ["angle_014.dat"]:
            print(f"Average hnc:", sum(hop_10_hnc)/len(hop_10_hnc))
            print(f"Average abc(hnc):", sum(hop_10)/len(hop_10))
        elif parameter in ["pyr_3210.dat"]:
            print(f"Average pyr:", sum(hop_10_pyr)/len(hop_10_pyr))
            print(f"Average abs(pyr):", sum(hop_10)/len(hop_10))
        elif parameter in ["e_gap.dat"]:
            print(f"Average e_gap:", sum(hop_10)/len(hop_10))
        return hop_10, hop_01

    def plot_torsion_ave_qy(self,xms_caspt2,sa_casscf,sa_oo_vqe):
        time_0, lower_0, upper_0, else_0 = self.get_torsion_qy_ave(xms_caspt2)
        time_1, lower_1, upper_1, else_1 = self.get_torsion_qy_ave(sa_casscf)
        time_2, lower_2, upper_2, else_2 = self.get_torsion_qy_ave(sa_oo_vqe)
        fig, ax = plt.subplots()
        plt.plot(time_0,upper_0, label = self.labels[0], color = self.colors[0],lw=2)
        plt.plot(time_1,upper_1, label = self.labels[1], color = self.colors[1],lw=2)
        plt.plot(time_2,upper_2, label = self.labels[2], color = self.colors[2],lw=2)
        plt.plot(time_0,lower_0, label = self.labels[0], color = self.colors[0],lw=2)
        plt.plot(time_1,lower_1, label = self.labels[1], color = self.colors[1],lw=2)
        plt.plot(time_2,lower_2, label = self.labels[2], color = self.colors[2],lw=2)
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('Torsion (degrees)', fontweight = 'bold', fontsize = 16)
        ax.spines['right'].set_visible(True)
        plt.ylim([-1, 180])
        plt.legend(loc='lower right',fontsize=13, frameon=False)
        ax1 = ax.twinx()
        ax1.set_ylim([-1, 180])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        plt.savefig("torsion_ave_comb_qy.pdf", bbox_inches='tight')
        plt.savefig("torsion_ave_comb_qy.png", bbox_inches='tight')
        plt.close()

    def plot_variance_noise(self, folder):
        time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_10/etot.dat')
        time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_06/etot.dat')

        fig, ax = plt.subplots()

        # Plot the lines
        plt.plot(time_0, noise_0, label=r"$\sigma^2$=1.0e-10", lw=2)
        plt.plot(time_1, noise_1, label=r"$\sigma^2$=1.0e-08", lw=2)
        plt.plot(time_2, noise_2, label=r"$\sigma^2$=1.0e-06", lw=2)

        # Plot the standard deviation (shaded area)
        plt.fill_between(time_0, np.array(noise_0) - np.array(std_0),
                         np.array(noise_0) + np.array(std_0), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        plt.fill_between(time_1, np.array(noise_1) - np.array(std_1),
                         np.array(noise_1) + np.array(std_1), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        plt.fill_between(time_2, np.array(noise_2) - np.array(std_2),
                         np.array(noise_2) + np.array(std_2), alpha=0.3, linestyle=':', edgecolor=self.colors[2])

        # Customize axis limits, labels, and formatting
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel(r'$\Delta$ Total Energy (eV)', fontweight='bold', fontsize=16)

        # Show legend and save the plot
        plt.legend(loc='upper left', fontsize=13, frameon=False)
        plt.ylim([-0.05, 2.4])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        plt.xlim([0, 200])

        plt.savefig("total_ene_ave_comb_std.pdf", bbox_inches='tight')
        plt.savefig("total_ene_ave_comb_std.png", bbox_inches='tight')
        plt.close()

    def plot_torsion_ave(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        time_0, torsion_0, std_0 = self.get_torsion_ave(xms_caspt2)
        time_1, torsion_1, std_1 = self.get_torsion_ave(sa_casscf)
        time_2, torsion_2, std_2 = self.get_torsion_ave(sa_oo_vqe)

        fig, ax = plt.subplots()

        # Plot the lines
        plt.plot(time_0, torsion_0, label=self.labels[0], lw=2)
        plt.plot(time_1, torsion_1, label=self.labels[1], lw=2)
        plt.plot(time_2, torsion_2, label=self.labels[2], lw=2)

        # Plot the standard deviation (shaded area)
        plt.fill_between(time_0, np.array(torsion_0) - np.array(std_0),
                         np.array(torsion_0) + np.array(std_0), alpha=0.3, linestyle='-.', edgecolor=self.colors[0])
        plt.fill_between(time_1, np.array(torsion_1) - np.array(std_1),
                         np.array(torsion_1) + np.array(std_1), alpha=0.3, linestyle='--', edgecolor=self.colors[1])
        plt.fill_between(time_2, np.array(torsion_2) - np.array(std_2),
                         np.array(torsion_2) + np.array(std_2), alpha=0.3, linestyle=':', edgecolor=self.colors[2])

        # Customize axis limits, labels, and formatting
        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel('Torsion (degrees)', fontweight='bold', fontsize=16)

        # Show legend and save the plot
        plt.legend(loc='lower right', fontsize=13, frameon=False)
        plt.ylim([-1, 180])

        plt.savefig("torsion_ave_comb_std.pdf", bbox_inches='tight')
        plt.savefig("torsion_ave_comb_std.png", bbox_inches='tight')
        plt.close()


    #def plot_torsion_ave(self,xms_caspt2,sa_casscf,sa_oo_vqe):
    #    time_0, torsion_0 = self.get_torsion_ave(xms_caspt2)
    #    time_1, torsion_1 = self.get_torsion_ave(sa_casscf)
    #    time_2, torsion_2 = self.get_torsion_ave(sa_oo_vqe)
    #    fig, ax = plt.subplots()
    #    plt.plot(time_0,torsion_0, label = self.labels[0], lw=2)
    #    plt.plot(time_1,torsion_1, label = self.labels[1], lw=2)
    #    plt.plot(time_2,torsion_2, label = self.labels[2], lw=2)
    #    plt.xlim([self.t_0, self.t_max])
    #    plt.xticks(fontsize=15)
    #    plt.yticks(fontsize=15)
    #    plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
    #    plt.ylabel('Torsion (degrees)', fontweight = 'bold', fontsize = 16)
    #    ax.spines['right'].set_visible(True)
    #    plt.ylim([-1, 180])
    #    plt.legend(loc='lower right',fontsize=13, frameon=False)
    #    ax1 = ax.twinx()
    #    ax1.set_ylim([-1, 180])
    #    ax1.tick_params(labelsize=15)
    #    ax1.set_ylabel(" ")
    #    plt.savefig("torsion_ave_comb.pdf", bbox_inches='tight')
    #    plt.savefig("torsion_ave_comb.png", bbox_inches='tight')
    #    plt.close()
    
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
    noise_sa_oo_vqe = "../noise_sa_oo_vqe"
    method = os.getcwd()
    #time in fs
    t_0 = 0
    t_max = 200 
    out = PlotComb(t_0, t_max)
    #out.plot_population_adi(index,xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.plot_1d_histogram(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    #out.plot_1d_histogram_2_plots(xms_caspt2,sa_casscf,sa_oo_vqe, 17)
    #out.plot_1d_histogram_2_plots_samen(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    #out.plot_1d_histogram_2_plots_samen_energy(xms_caspt2,sa_casscf,sa_oo_vqe, 20)
    #out.plot_1d_histogram_2_plots_energy(xms_caspt2,sa_casscf,sa_oo_vqe, 31)
    #out.plot_1d_histogram_4_plots_S1_S0(xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.print_stat(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_torsion_ave(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_torsion_ave_qy(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_av_popu_torsion_bend(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_variance_noise(noise_sa_oo_vqe)
    #out.plot_av_popu_noise(noise_sa_oo_vqe)
    #out.plot_av_popu_torsion_noise(noise_sa_oo_vqe)
    #out.plot_av_popu_diff_ene(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_one_method_av_popu_diff_ene(method)
    #out.get_torsion_qy_ave(xms_caspt2)
    #out.get_torsion_qy_ave(sa_oo_vqe)
    #out.get_torsion_qy_ave(sa_casscf)
    #out.get_torsion_qy_ave_2(xms_caspt2)
    #out.get_torsion_qy_ave_2(sa_oo_vqe)
    #out.get_torsion_qy_ave_2(sa_casscf)
    #out.get_torsion_qy_ave_noise(noise_sa_oo_vqe)
    #out.plot_1d_histogram_QY_time(xms_caspt2,sa_casscf,sa_oo_vqe, 7)
    out.plot_2d_histogram_QY_time(xms_caspt2,sa_casscf,sa_oo_vqe, 7)
