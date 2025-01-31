import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import sys
import csv

from scipy.interpolate import make_interp_spline
from pandas import (read_csv, DataFrame)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import (namedtuple, Counter)
from pysurf.database import PySurfDB

class PlotComb:

    def __init__(self, t_0, t_max, lower_2_a):
        self.lower = lower_2_a
        self.ev = 27.211324570273 
        self.fs = 0.02418884254
        self.aa = 0.5291772105638411 
        self.fs_rcParams = '10'
        self.f_size = '11'
        self.t_0 = t_0
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3] 
        self.n_colors = [self.colors[2],"purple","gold","olive","blue"] 
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

    def filter_files(self, folder):
        file_1 = self._filter_cv_files(os.path.join(folder,"dis_r25.dat")) #NH
        file_2 = self._filter_cv_files(os.path.join(folder,"dis_r13.dat")) #CH
        file_3 = self._filter_cv_files(os.path.join(folder,"dis_r14.dat")) #CH
        result = []
        for elem in file_1:
            if elem in file_2 and elem in file_3:
                result.append(elem)    
        title = folder.replace("../", "").replace("/", "_")
        with open(f'filter_trajectories_{title}.out', 'w') as f1:
            f1.write('--------------------------------------------------------------\n')
            f1.write(f'Folder: {title}\n')
            f1.write(f'Number of trajt. after filter: {len(result)}\n')
            f1.write(f'Array final: {result}\n')
            f1.write('--------------------------------------------------------------')
            f1.close() 
        return result

    def _filter_cv_files(self, files):
        traj_2_l = []
        with open(files, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pass
        for k, val in row.items():
            if k == 'time':
                continue
            if float(val) <= 2:
                traj_2_l.append(k)
        #print("Len before filter:",len(list(row.items())[1:]))
        #print("Array before:",list(row.items())[1:])
        #print("Len after filter:",len(traj_2_l))
        #print("Array after:",traj_2_l)
        return traj_2_l
        
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
        time_2, lower_2, upper_2, else_2 = self.get_torsion_qy_ave(os.path.join(folder,"variance_06"))
        time_3, lower_3, upper_3, else_3 = self.get_torsion_qy_ave(os.path.join(folder,"variance_00"))

    def get_torsion_qy_ave_2(self, folder):
        filename = os.path.join(folder,"dihe_2014.dat")
        popu = os.path.join(folder,"pop.dat")
        filter_2 = self.filter_files(folder)
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
                    if k_1 == 'time'or k_1 not in filter_2:
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
                        #print(lower_50, nans)
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

    def margin_95_confidence(self, n_traj, qy):
        return 1.96*np.sqrt((qy*(1-qy))/n_traj)

    def get_torsion_qy_ave(self, folder):
        filename = os.path.join(folder,"dihe_2014.dat")
        popu = os.path.join(folder,"pop.dat")
        filter_2 = self.filter_files(folder)
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
                    if k_1 == 'time' or k_1 not in filter_2:
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
                        ave_lower.append(ref_non_r/int(lower_30-nans))
                    else:
                        ave_lower.append(ref/int(trajs-nans))
                    if else_ang != 0:
                        ave_else.append(ref_rest/else_ang)
                    else:
                        ave_else.append(ref/int(trajs-nans))
        title = folder.replace("../", "").replace("/", "_")
        l_30_u_150 = lower_30+upper_150
        traj_nans = int(trajs-nans)
        err_95_l_30 = self.margin_95_confidence(l_30_u_150,lower_30/l_30_u_150)
        err_95_u_150 = self.margin_95_confidence(l_30_u_150,upper_150/l_30_u_150)
        with open(f'QY_information_30_150_{title}.out', 'w') as f3:
            f3.write('--------------------------------------------------------------\n')
            f3.write(f'Folder: {title}\n')
            #f3.write(f'lower_30/{traj_nans}: {round(lower_30/traj_nans,2)}\n')
            f3.write(f'lower_30/(lower_30+upper_150): {round(100*(lower_30/l_30_u_150),0)}\n')
            #f3.write(f'upper_150/{traj_nans}: {round(upper_150/traj_nans,2)}\n')
            f3.write(f'upper_150/(lower_30+upper_150): {round(100*(upper_150/l_30_u_150),0)}\n')
            f3.write(f'lower_S0_30 = {lower_30}, upper_S0_150 = {upper_150}, rest_S0 = {else_ang}, S1 = {ref_S1}\n')
            #f3.write(f'Total:  {round(l_30_u_150 + else_ang + ref_S1,2)}\n')
            f3.write(f'error 95% confident interval lower_30: {round(100*(err_95_l_30),0)}\n')
            f3.write(f'error 95% confident interval upper_150: {round(100*(err_95_u_150),0)}\n')
            f3.write(f'Trajs - Nans: {int(trajs-nans)}\n')
            f3.write('--------------------------------------------------------------\n')
            f3.write(f'Length lower 2 angstrom: {len(filter_2)}\n')
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
        filter_2 = self.filter_files(folder)
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                para_vals = []
                for k, val in row.items():
                    if k == 'time' or k not in filter_2:
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
        filter_name = noise.split('/')[0]
        filter_2 = self.filter_files(os.path.join(folder, filter_name))
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                noise_vals = []
                for k, val in row.items():
                    if k == 'time' or k not in filter_2:
                        continue
                    if val != 'nan' and not "etot" in noise:  # Only consider valid (non-'nan') values
                        noise_vals.append(abs(float(val)))
                    else:
                        noise_vals.append(float(val))

                if len(noise_vals) > 0:  # If we have valid noise values
                    ave_noise.append(np.mean(noise_vals))  # Compute average
                    noise_data.append(noise_vals)  # Store the noise values for std calculation
        
        # Convert noise_data into a numpy array (2D array: rows -> time steps, columns -> trajectories)
        noise_data = np.array([np.pad(t, (0, max(len(x) for x in noise_data) - len(t)), constant_values=np.nan) for t in noise_data])
        
        # Compute standard deviation along axis=1 (time axis)
        noise_std = np.nanstd(noise_data, axis=1)  # Use np.nanstd to ignore NaN values
        
        #title = filename.replace('../noise_sa_oo_vqe/', '')
        #title = title.replace('/etot.dat', '')
        #df = DataFrame({"ave_time" : np.array(ave_time), "ave_noise" : np.array(ave_noise)})
        #df.to_csv(f"data_ave_time_noise_{title}.csv", index=False)
        return np.array(ave_time), np.array(ave_noise), noise_std

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
        filter_name = filename.replace("/pop.dat","")
        prop = self.read_prop(filter_name)
        states = prop.states
        nstates = prop.nstates
        ave_time = []
        ave_popu = []
        filter_2 = self.filter_files(filter_name)
        with open(filename, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ave_time.append(float(row['time']))
                nans = 0
                trajs = 0
                ref = np.zeros(nstates)
                for k, val in row.items():
                    if k == 'time' or k not in filter_2:
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
        tor_1_0 = self.get_histogram_qy(sa_casscf,0)
        tor_2_0 = self.get_histogram_qy(sa_oo_vqe,0)
        #First state
        tor_1_1 = self.get_histogram_qy(sa_casscf,1)
        tor_2_1 = self.get_histogram_qy(sa_oo_vqe,1)
        bins = np.linspace(0, 180, n_bins) 
        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,8))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        # the first subplot
        ax0 = plt.subplot(gs[0])
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
        plt.savefig("number_of_dihe_qy_2.pdf", bbox_inches='tight')
        plt.savefig("number_of_dihe_qy_2.png", bbox_inches='tight')
        plt.close()

    def _2d_histogram(self,hop_10_x,hop_10_y,n_bins, x_type="e_gap", y_type="hcnh"):
        plt.rcParams['font.size'] = '14'
        #plt.rcParams['axes.labelpad'] = 9
        fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
        ax = fig.add_subplot(111, projection='3d')
        plt.xlim([self.t_0, self.t_max])
        bins = [x for x in range(self.t_0, self.t_max+1,int(self.t_max/n_bins))]
        hist, xedges, yedges = np.histogram2d(hop_10_x,hop_10_y, bins=[bins,bins_1])

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        dx = (xedges [1] - xedges [0])/3
        dy = (yedges [1] - yedges [0])/3
        dz = hist.flatten()

        xpos_1, ypos_1 = np.meshgrid(xedges_1[:-1] + 0.25 + dx, yedges_1[:-1] - 0.25 - dy, indexing="ij")
        xpos_1 = xpos_1.ravel()
        ypos_1 = ypos_1.ravel()
        zpos_1 = 0

        dx_1 = (xedges_1 [1] - xedges_1 [0])/3
        dy_1 = (yedges_1 [1] - yedges_1 [0])/3
        dz_1 = hist_1.flatten()

        #print("10: ",hist, xedges, yedges)
        #print("10: ",xpos, ypos, zpos, dx, dy, dz)
        #print("01: ",hist_1, xedges_1, yedges_1)
        #print("01: ",xpos_1, ypos_1, zpos_1, dx_1, dy_1, dz_1)
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color="green", zsort='max', alpha=0.4, edgecolor='black')
        green_proxy = plt.Rectangle((0, 0), 1, 1, fc="green")
        ax.bar3d(xpos_1, ypos_1, zpos_1, dx_1, dy_1, dz_1, color="red", zsort='max', alpha=0.5, edgecolor='black')
        labels = [r"$S_1$ $\rightarrow$ $S_0$"]
        ax.legend([green_proxy],labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), prop={'size': 12}, ncol=2)
        ax.set_xlabel('Energy Gap (eV)', fontsize=12,fontweight = 'bold',rotation=150)
        ax.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=12,fontweight = 'bold')
        # change fontsize
        ax.zaxis.set_tick_params(labelsize=12, pad=0)
        ax.xaxis.set_tick_params(labelsize=12, pad=0)
        ax.yaxis.set_tick_params(labelsize=12, pad=0)
        # disable auto rotation
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('Number of Hops', fontsize=12,fontweight = 'bold', rotation=90, labelpad=0)
        ax.view_init(30, 230)
        plt.savefig("number_of_hops_ene_dihe.pdf", bbox_inches='tight', pad_inches = 0.4)
        plt.savefig("number_of_hops_ene_dihe.png", bbox_inches='tight', pad_inches = 0.4)
        plt.close()

    def plot_1d_histogram_QY_time(self, xms_caspt2,sa_casscf,sa_oo_vqe,n_bins=16):
        tor_1_0 = self.get_histogram_qy(sa_casscf,0)
        tor_2_0 = self.get_histogram_qy(sa_oo_vqe,0)
        bins = np.linspace(0, 180, n_bins) 

        # Create figure and axis
        fig, ax = plt.subplots()
        plt.rcParams['font.size'] = self.fs_rcParams
        
        # Plot histograms on the axis
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
        plt.savefig("number_of_dihe_qy_1.pdf", bbox_inches='tight')
        plt.savefig("number_of_dihe_qy_1.png", bbox_inches='tight')
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

    def _para(self, sa_casscf, sa_oo_vqe, data):
        time_cas, ave_cas, std_cas = self.get_parameter_ave(sa_casscf, data)
        time_vqe, ave_vqe, std_vqe = self.get_parameter_ave(sa_oo_vqe, data)
        para = namedtuple("para","t_cas av_cas std_cas t_vqe av_vqe std_vqe") 
        return para(time_cas,ave_cas,std_cas,time_vqe,ave_vqe,std_vqe)

    def plot_population_adi_fitted(self, folder):
        title = folder.replace('../','')
        #popu
        time, population = self.get_popu_adi(folder,os.path.join(folder,"pop.dat"))
        params_S1, cv_S1 = curve_fit(self.monoexponetial_S1, time, np.array(population)[:,1])
        S1_t_d = params_S1[0]
        S1_t_e = params_S1[1]
        plt.plot(time,np.array(population)[:,1], label = '$S_1$')
        plt.plot(time, self.monoexponetial_S1(time, S1_t_d, S1_t_e), '--', label="fitted S1")
        plt.xlim([0, 200])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{Population}$', fontsize = 16)
        plt.legend(loc='center right',fontsize=15)
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"population_adi_fitted_{title}.pdf", bbox_inches='tight')
        plt.savefig(f"population_adi_fitted_{title}.png", bbox_inches='tight')
        plt.close()

        conf_S1, error_S1 = self.confidence_interval_95_bootstrap(self.monoexponetial_S1,np.array(time),np.array(population)[:,1]) 
            
        with open(f'time_d_e_{title}.out','w') as f2:
            f2.write(f'--------------------------------------------------------------\n')
            f2.write(f't_d_S1_mean: {conf_S1[0]:>0.3f}\n')
            f2.write(f't_e_S1_mean: {conf_S1[1]:>0.3f}\n')
            f2.write(f't_e_S1_error: {error_S1[0]:>0.3f}\n')
            f2.write(f't_e_S1_error: {error_S1[1]:>0.3f}\n')
            f2.write('--------------------------------------------------------------')
            f2.close() 

    def plot_av_popu_torsion_bend(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        #popu
        time_1, population_1 = self.get_popu_adi(sa_casscf,os.path.join(sa_casscf,"pop.dat"))
        time_2, population_2 = self.get_popu_adi(sa_oo_vqe,os.path.join(sa_oo_vqe,"pop.dat"))
        #dihe_2014
        dihe = self._para(sa_casscf,sa_oo_vqe,"dihe_2014.dat")
        #angle_014
        bend = self._para(sa_casscf,sa_oo_vqe,"angle_014.dat")
        #pyr_3210
        pyr = self._para(sa_casscf,sa_oo_vqe,"pyr_3210.dat")

        plt.rcParams['font.size'] = self.fs_rcParams
        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_1,np.array(population_1)[:,1], label = self.labels[1], color = self.colors[1], lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], label = self.labels[2], color = self.colors[2], lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,200])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))
            
        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(dihe.t_cas,dihe.av_cas, color = self.colors[1], lw=2)
        ax1.plot(dihe.t_vqe,dihe.av_vqe, color = self.colors[2], lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-8, 185])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(dihe.t_cas, np.array(dihe.av_cas) - np.array(dihe.std_cas),
                         np.array(dihe.av_cas) + np.array(dihe.std_cas), alpha=0.3, linestyle='--', color = self.colors[1], edgecolor=self.colors[1])
        ax1.fill_between(dihe.t_vqe, np.array(dihe.av_vqe) - np.array(dihe.std_vqe),
                         np.array(dihe.av_vqe) + np.array(dihe.std_vqe), alpha=0.3, linestyle=':', color = self.colors[2], edgecolor=self.colors[2])
        ax1.set_ylim([-8,185])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)

        # the 3rd subplot
        ax2 = plt.subplot(gs[2], sharex = ax0)
        ax2.plot(bend.t_cas,bend.av_cas, color = self.colors[1], lw=2)
        ax2.plot(bend.t_vqe,bend.av_vqe, color = self.colors[2], lw=2)
        ax2r = ax2.twinx()
        ax2r.set_ylim([53, 185])
        ax2r.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax2.fill_between(bend.t_cas, np.array(bend.av_cas) - np.array(bend.std_cas),
                         np.array(bend.av_cas) + np.array(bend.std_cas), alpha=0.3, linestyle='--', color = self.colors[1], edgecolor=self.colors[1])
        ax2.fill_between(bend.t_vqe, np.array(bend.av_vqe) - np.array(bend.std_vqe),
                         np.array(bend.av_vqe) + np.array(bend.std_vqe), alpha=0.3, linestyle=':', color = self.colors[2], edgecolor=self.colors[2])
        ax2.set_ylim([53,185])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.set_ylabel('$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # the 4th subplot
        ax3 = plt.subplot(gs[3], sharex = ax0)
        ax3.plot(pyr.t_cas,pyr.av_cas, color = self.colors[1], lw=2)
        ax3.plot(pyr.t_vqe,pyr.av_vqe, color = self.colors[2], lw=2)
        ax3r = ax3.twinx()
        ax3r.set_ylim([-8, 95])
        ax3r.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax3r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax3.fill_between(pyr.t_cas, np.array(pyr.av_cas) - np.array(pyr.std_cas),
                         np.array(pyr.av_cas) + np.array(pyr.std_cas), alpha=0.3, linestyle='--', color = self.colors[1], edgecolor=self.colors[1])
        ax3.fill_between(pyr.t_vqe, np.array(pyr.av_vqe) - np.array(pyr.std_vqe),
                         np.array(pyr.av_vqe) + np.array(pyr.std_vqe), alpha=0.3, linestyle=':', color = self.colors[2], edgecolor=self.colors[2])
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
        time_0, population_0 = self.get_popu_adi(folder,os.path.join(folder,"variance_00/pop.dat"))
        time_1, population_1 = self.get_popu_adi(folder,os.path.join(folder,"variance_08/pop.dat"))
        time_2, population_2 = self.get_popu_adi(folder,os.path.join(folder,"variance_07/pop.dat"))
        time_3, population_3 = self.get_popu_adi(folder,os.path.join(folder,"variance_06/pop.dat"))
        time_4, population_4 = self.get_popu_adi(folder,os.path.join(folder,"variance_05/pop.dat"))
        #torsion
        time_0, t_noise_0, t_std_0 = self.get_noise_ave(folder,'variance_00/dihe_2014.dat')
        time_1, t_noise_1, t_std_1 = self.get_noise_ave(folder,'variance_08/dihe_2014.dat')
        time_2, t_noise_2, t_std_2 = self.get_noise_ave(folder,'variance_07/dihe_2014.dat')
        time_3, t_noise_3, t_std_3 = self.get_noise_ave(folder,'variance_06/dihe_2014.dat')
        time_4, t_noise_4, t_std_4 = self.get_noise_ave(folder,'variance_05/dihe_2014.dat')
        #noise
        time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_00/etot.dat')
        time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_07/etot.dat')
        time_3, noise_3, std_3 = self.get_noise_ave(folder,'variance_06/etot.dat')
        time_4, noise_4, std_4 = self.get_noise_ave(folder,'variance_05/etot.dat')
        #fitted
        params_0, bs_error_95_0 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_0, noise_0)
        a_0 = params_0[0]
        b_0 = params_0[1]
        params_1, bs_error_95_1 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_1, noise_1)
        a_1 = params_1[0]
        b_1 = params_1[1]
        params_2, bs_error_95_2 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_2, noise_2)
        a_2 = params_2[0]
        b_2 = params_2[1]
        params_3, bs_error_95_3 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_3, noise_3)
        a_3 = params_3[0]
        b_3 = params_3[1]
        params_4, bs_error_95_4 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_4, noise_4)
        a_4 = params_4[0]
        b_4 = params_4[1]
        plt.rcParams['font.size'] = self.fs_rcParams

        fig = plt.figure(figsize=(6,14))
        # set height ratios for subplots
        gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])
        # the 1st subplot
        ax0 = plt.subplot(gs[0])
        ax0.plot(time_0,np.array(population_0)[:,1], color = self.n_colors[0], label = "no noise", lw=2, alpha=0.8)
        ax0.plot(time_1,np.array(population_1)[:,1], color = self.n_colors[1], label = r"$\sigma^2$=1.0e-08", lw=2)
        ax0.plot(time_2,np.array(population_2)[:,1], color = self.n_colors[2], label = r"$\sigma^2$=1.0e-07", lw=2)
        ax0.plot(time_3,np.array(population_3)[:,1], color = self.n_colors[3], label = r"$\sigma^2$=1.0e-06", lw=2)
        ax0.plot(time_4,np.array(population_4)[:,1], color = self.n_colors[4], label = r"$\sigma^2$=1.0e-05", lw=2)
        ax0r = ax0.twinx()
        ax0r.set_ylim([-0.05, 1.05])
        ax0r.tick_params(labelsize=self.fs_rcParams)
        ax0.set_ylabel('$\mathbf{S_1\ Population}$', fontsize =self.f_size)
        ax0.set_xlim([0,100])
        ax0.set_ylim([-0.05,1.05])
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(25))

        # the 2nd subplot
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax1.plot(time_0, t_noise_0, color = self.n_colors[0], lw=2, alpha=0.8)
        ax1.plot(time_1, t_noise_1, color = self.n_colors[1], lw=2)
        ax1.plot(time_2, t_noise_2, color = self.n_colors[2], lw=2)
        ax1.plot(time_3, t_noise_3, color = self.n_colors[3], lw=2)
        ax1.plot(time_4, t_noise_4, color = self.n_colors[4], lw=2)
        ax1r = ax1.twinx()
        ax1r.set_ylim([-8, 185])
        ax1r.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1r.tick_params(labelsize=self.fs_rcParams)
        # Plot the standard deviation (shaded area)
        ax1.fill_between(time_0, np.array(t_noise_0) - np.array(t_std_0),
                         np.array(t_noise_0) + np.array(t_std_0), alpha=0.3, color = self.n_colors[0], linestyle='-', edgecolor=self.n_colors[0])
        ax1.fill_between(time_1, np.array(t_noise_1) - np.array(t_std_1),
                         np.array(t_noise_1) + np.array(t_std_1), alpha=0.3, color = self.n_colors[1], linestyle='--', edgecolor=self.n_colors[1])
        ax1.fill_between(time_2, np.array(t_noise_2) - np.array(t_std_2),
                         np.array(t_noise_2) + np.array(t_std_2), alpha=0.3, color = self.n_colors[2], linestyle=':', edgecolor=self.n_colors[2])
        ax1.fill_between(time_3, np.array(t_noise_3) - np.array(t_std_3),
                         np.array(t_noise_3) + np.array(t_std_3), alpha=0.3, color = self.n_colors[3], linestyle='-.', edgecolor=self.n_colors[1])
        ax1.fill_between(time_4, np.array(t_noise_4) - np.array(t_std_4),
                         np.array(t_noise_4) + np.array(t_std_4), alpha=0.3, color = self.n_colors[4], linestyle=(0, (1, 1)), edgecolor=self.n_colors[2])
        ax1.set_ylim([-8,185])
        ax1.set_xlim([0,100])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(30))
        ax1.set_ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)
        plt.setp(ax0.get_xticklabels(), visible=False)
            
        # the 3rd subplot
        ax2 = plt.subplot(gs[2], sharex = ax0)
        ax2.plot(time_0, noise_0, color = self.n_colors[0], label = "no noise", lw=2, alpha=0.8)
        ax2.plot(time_1, noise_1, color = self.n_colors[1], label = r"$\sigma^2$=1.0e-08", lw=2)
        ax2.plot(time_2, noise_2, color = self.n_colors[2], label = r"$\sigma^2$=1.0e-07", lw=2)
        ax2.plot(time_3, noise_3, color = self.n_colors[3], label = r"$\sigma^2$=1.0e-06", lw=2)
        ax2.plot(time_4, noise_4, color = self.n_colors[4], label = r"$\sigma^2$=1.0e-05", lw=2)
        ax2r = ax2.twinx()
        ax2r.set_ylim([-2.37, 2.37])
        ax2r.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax2r.tick_params(labelsize=self.fs_rcParams)
        ## Plot the standard deviation (shaded area)
        #ax2.fill_between(time_0, np.array(noise_0) - np.array(std_0),
        #                 np.array(noise_0) + np.array(std_0), alpha=0.3, color = self.n_colors[0], linestyle='-', edgecolor=self.n_colors[0])
        #ax2.fill_between(time_1, np.array(noise_1) - np.array(std_1),
        #                 np.array(noise_1) + np.array(std_1), alpha=0.3, color = self.n_colors[1], linestyle='--', edgecolor=self.n_colors[1])
        #ax2.fill_between(time_2, np.array(noise_2) - np.array(std_2),
        #                 np.array(noise_2) + np.array(std_2), alpha=0.3, color = self.n_colors[2], linestyle=':', edgecolor=self.n_colors[2])
        #ax2.fill_between(time_3, np.array(noise_3) - np.array(std_3),
        #                 np.array(noise_3) + np.array(std_3), alpha=0.3, color = self.n_colors[3], linestyle='-.', edgecolor=self.n_colors[1])
        #ax2.fill_between(time_4, np.array(noise_4) - np.array(std_4),
        #                 np.array(noise_4) + np.array(std_4), alpha=0.3, color = self.n_colors[4], linestyle=(0, (1, 1)), edgecolor=self.n_colors[2])
        # Plot linear equation with fitted data
        #sig = "+"
        ax2.plot(time_0, self.linear_total_energy(time_0, a_0, b_0), '--', color = "orange", label=f"$\Delta T.E. = {a_0:.5f}t {'+' if b_0 >= 0 else ''}{b_0:.5f}$")
        ax2.plot(time_1, self.linear_total_energy(time_1, a_1, b_1), '--', color = "green", label=f"$\Delta T.E. = {a_1:.5f}t {'+' if b_1 >= 0 else ''}{b_1:.5f}$")
        ax2.plot(time_2, self.linear_total_energy(time_2, a_2, b_2), '--', color = "red", label=f"$\Delta T.E. = {a_2:.5f}t {'+' if b_2 >= 0 else ''}{b_2:.5f}$")
        ax2.plot(time_3, self.linear_total_energy(time_3, a_3, b_3), '--', color = "brown", label=f"$\Delta T.E. = {a_3:.5f}t {'+' if b_3 >= 0 else ''}{b_3:.5f}$")
        ax2.plot(time_4, self.linear_total_energy(time_4, a_4, b_4), '--', color = "magenta", label=f"$\Delta T.E. = {a_4:.5f}t {'+' if b_4 >= 0 else ''}{b_4:.5f}$")
        ax2.set_ylim([-2.37, 2.37])
        ax2.set_xlim([0,100])
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax2.set_ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize=self.f_size)
        ax2.set_xlabel('Time (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Adjust space between the title and subplots
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
        
        # Set labels and legends
        ax0.text(0.95, 0.95, f'(a)', transform=ax0.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax1.text(0.95, 0.95, f'(b)', transform=ax1.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
        ax2.text(0.95, 0.95, f'(c)', transform=ax2.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')


        # put legend on first subplot
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 3.51), prop={'size': 12}, ncol=2)

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=0.0)
        title = folder.replace('../','')
        plt.savefig(f"avg_popu_torsion_noise_{title}.pdf", bbox_inches='tight')
        plt.savefig(f"avg_popu_torsion_noise_{title}.png", bbox_inches='tight')
        plt.close()
        with open(f'ci_noise_linear_regression_{title}.out', 'w') as f3:
            f3.write('--------------------------------------------------------------\n')
            f3.write(f'Folder: {title}\n')
            f3.write(f'a_00_mean: {a_0}\n')
            f3.write(f'b_00_mean: {b_0}\n')
            f3.write(f'a_00_error: {bs_error_95_0[0]}\n')
            f3.write(f'b_00_error: {bs_error_95_0[1]}\n')
            f3.write(f'a_08_mean: {a_1}\n')
            f3.write(f'b_08_mean: {b_1}\n')
            f3.write(f'a_08_error: {bs_error_95_1[0]}\n')
            f3.write(f'b_08_error: {bs_error_95_1[1]}\n')
            f3.write(f'a_07_mean: {a_2}\n')
            f3.write(f'b_07_mean: {b_2}\n')
            f3.write(f'a_07_error: {bs_error_95_2[0]}\n')
            f3.write(f'b_07_error: {bs_error_95_2[1]}\n')
            f3.write(f'a_06_mean: {a_3}\n')
            f3.write(f'b_06_mean: {b_3}\n')
            f3.write(f'a_06_error: {bs_error_95_3[0]}\n')
            f3.write(f'b_06_error: {bs_error_95_3[1]}\n')
            f3.write(f'a_05_mean: {a_4}\n')
            f3.write(f'b_05_mean: {b_4}\n')
            f3.write(f'a_05_error: {bs_error_95_4[0]}\n')
            f3.write(f'b_05_error: {bs_error_95_4[1]}\n')
            f3.write('--------------------------------------------------------------')
            f3.close() 
 

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

    def plot_1d_curves(self, xms_caspt2, sa_casscf, sa_oo_vqe):
        # Load data for each parameter
        hop_e = self._hop_10(xms_caspt2, sa_casscf, sa_oo_vqe, "e_gap.dat")
        hop_d = self._hop_10(xms_caspt2, sa_casscf, sa_oo_vqe, "dihe_2014.dat")
        hop_a = self._hop_10(xms_caspt2, sa_casscf, sa_oo_vqe, "angle_014.dat")
        hop_p = self._hop_10(xms_caspt2, sa_casscf, sa_oo_vqe, "pyr_3210.dat")
        
        data = [
            (hop_e, self.bins.ene, 'Energy Gap (eV)', [0, 3], 0, f'(a)'),
            (hop_d, self.bins.hnch, r'$\mathbf{\sphericalangle H_3C_1N_2H_5}$ (degrees)', [0, 180], 109, f'(b)'),
            (hop_a, self.bins.hnc, r'$\mathbf{\sphericalangle C_1N_2H_5}$ (degrees)', [0, 180], 111, f'(c)'),
            (hop_p, self.bins.pyr, 'Pyramidalization (degrees)', [0, 180], 34, f'(d)'),
        ]
    
        plt.rcParams['font.size'] = self.fs_rcParams
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.supylabel('Distribution of Hops', fontweight='bold', fontsize=18)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.9, hspace=0.2)
    
        for ax, (hop, bins, xlabel, xlim, meci, i_label) in zip(axs.flatten(), data):
            bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers
            
            # Compute counts for each method
            counts_cas, _ = np.histogram(hop.cas, bins=bins)
            counts_vqe, _ = np.histogram(hop.vqe, bins=bins)
    
            # Smooth curves
            x_new = np.linspace(bin_centers[0], bin_centers[-1], 300)
            cas_smooth = make_interp_spline(bin_centers, counts_cas, k=3)(x_new)
            vqe_smooth = make_interp_spline(bin_centers, counts_vqe, k=3)(x_new)
    
            # Plot
            ax_1, = ax.plot(x_new, cas_smooth, label="", color=self.colors[1], lw=2)
            ax.scatter(bin_centers, counts_cas, color=self.colors[1], zorder=5)
    
            ax_2, = ax.plot(x_new, vqe_smooth, label="", color=self.colors[2], lw=2)
            ax.scatter(bin_centers, counts_vqe, color=self.colors[2], zorder=5)

    
            # Set labels and legends
            ax.text(0.95, 0.95, i_label, transform=ax.transAxes,
             fontsize=self.f_size, fontweight='bold', va='top', ha='right')
            ax.set_xlim(xlim)
            ax.set_xlabel(xlabel, fontsize=self.f_size, fontweight='bold')

            if meci != 0:
                ax.axvline(meci,label="MECI",linestyle='--', c = 'purple')
        # Add legend only once
        axs[0,0].legend([ax_1,ax_2],
        [self.labels[1], self.labels[2]],
        loc='upper center', bbox_to_anchor=(1, 1.2),
        prop={'size': 14}, ncol=2
        )   

        plt.savefig("number_of_hops_curves.pdf", bbox_inches='tight')
        plt.savefig("number_of_hops_curves.png", bbox_inches='tight')
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
        #ax00.hist(hop_e.xms, bins = self.bins.ene, ec = self.colors[0], label=self.labels[0] ,fc='none', lw=2)
        ax00.hist(hop_e.cas, bins = self.bins.ene, ec = self.colors[1], label=self.labels[1] ,fc='none', lw=2)
        ax00.hist(hop_e.vqe, bins = self.bins.ene, ec = self.colors[2], label=self.labels[2] ,fc='none', lw=2)
        ax00.set_xlim([0,3])
        ax00.xaxis.set_major_locator(ticker.MultipleLocator(0.6))
        ax00.set_xlabel('Energy Gap (eV)', fontweight='bold', fontsize= self.f_size)
            
        # the 2nd subplot
        ax01 = plt.subplot(gs[0,1])
        #ax01.hist(hop_d.xms, bins = self.bins.hnch, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax01.hist(hop_d.cas, bins = self.bins.hnch, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax01.hist(hop_d.vqe, bins = self.bins.hnch, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax01.set_xlim([0,180])
        ax01.axvline(109,label="MECI",linestyle='--', c = 'purple')
        ax01.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax01.set_xlabel('$\mathbf{\sphericalangle H_3C_1N_2H_5(degrees)}$', fontsize=self.f_size)

        # the 3rd subplot
        ax10 = plt.subplot(gs[1,0])
        #ax10.hist(hop_a.xms, bins = self.bins.hnc, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax10.hist(hop_a.cas, bins = self.bins.hnc, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax10.hist(hop_a.vqe, bins = self.bins.hnc, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax10.axvline(111,label="MECI",linestyle='--', c = 'purple')
        ax10.set_xlim([0,180])
        ax10.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax10.set_xlabel('$\mathbf{\sphericalangle C_1N_2H_5(degrees)}$', fontsize=self.f_size)

        # the 4th subplot
        ax11 = plt.subplot(gs[1,1])
        #ax11.hist(hop_p.xms, bins = self.bins.pyr, ec = self.colors[0], label="" ,fc='none', lw=2)
        ax11.hist(hop_p.cas, bins = self.bins.pyr, ec = self.colors[1], label="" ,fc='none', lw=2)
        ax11.hist(hop_p.vqe, bins = self.bins.pyr, ec = self.colors[2], label="" ,fc='none', lw=2)
        ax11.axvline(34,label="MECI",linestyle='--', c = 'purple')
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
        #ax01.legend(bbox_to_anchor=(0.98, 0.9), frameon=False)
        #ax10.legend(bbox_to_anchor=(0.98, 0.9), frameon=False)
        #ax11.legend(bbox_to_anchor=(0.98, 0.9), frameon=False)

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
        filter_2 = self.filter_files(folder)
        #cur = pop.to_numpy()[:,1:] # removing time column
        #tor = torsion.to_numpy()[:,1:] # removing time column
        cur = pop[filter_2].to_numpy() 
        tor = torsion[filter_2].to_numpy() 
        mdsteps,trajs = cur.shape 
        print("Traj in get_histogram:",trajs)
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
        filter_2 = self.filter_files(folder)
        #hop = pop.to_numpy()[:,1:] # removing time column
        #ene_d = e_gap.to_numpy()[:,1:] # removing time column
        ene_d = e_gap[filter_2].to_numpy() 
        hop = pop[filter_2].to_numpy() 
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

    def monoexponetial_S1(self, t, t_d, t_e):
        return np.exp(-(t -t_d)/t_e)

    def linear_total_energy(self, t, a, b):
        return a*t + b

    def confidence_interval_95_bootstrap(self, function, t_data, data):
        # Number of bootstrap samples
        num_bootstrap_samples = 1000
        
        # Initialize arrays to store bootstrap estimates
        bootstrap_a = np.zeros(num_bootstrap_samples)
        bootstrap_b = np.zeros(num_bootstrap_samples)

        # Bootstrap procedure
        for i in range(num_bootstrap_samples):
            # Create a bootstrap sample
            bootstrap_indices = np.random.choice(len(t_data), len(t_data), replace=True)
            bootstrap_t = t_data[bootstrap_indices]
            bootstrap_data = data[bootstrap_indices]

            # Perform monoexponential fit
            data_f, _ = curve_fit(function, bootstrap_t, bootstrap_data)

            # Store bootstrap estimates
            bootstrap_a[i] = data_f[0]
            bootstrap_b[i] = data_f[1]

        # Calculate 95% confidence intervals
        confidence_interval_a = np.percentile(bootstrap_a, [2.5, 97.5])
        confidence_interval_b = np.percentile(bootstrap_b, [2.5, 97.5])
        a_error = (confidence_interval_a[1]-confidence_interval_a[0])/2
        b_error = (confidence_interval_b[1]-confidence_interval_b[0])/2
        a_mean = confidence_interval_a[1]-a_error
        b_mean = confidence_interval_b[1]-b_error
        return [a_mean, b_mean], [a_error, b_error] 


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

    def plot_total_energy_fitted(self, folder): 
        #noise
        time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_10/etot.dat')
        time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        #time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_06/etot.dat')
        time_3, noise_3, std_3 = self.get_noise_ave(folder,'variance_00/etot.dat')
        #fitted
        params_0, cv_noise_0 = curve_fit(self.linear_total_energy, time_0, noise_0)
        a_0 = params_0[0]
        b_0 = params_0[1]
        params_1, cv_noise_1 = curve_fit(self.linear_total_energy, time_1, noise_1)
        a_1 = params_1[0]
        b_1 = params_1[1]
        #params_2, cv_noise_2 = curve_fit(self.linear_total_energy, time_2, noise_2)
        #a_2 = params_2[0]
        #b_2 = params_2[1]
        params_3, cv_noise_3 = curve_fit(self.linear_total_energy, time_3, noise_3)
        a_3 = params_3[0]
        b_3 = params_3[1]

        fig, ax = plt.subplots()
        #noise
        plt.plot(time_3, noise_3, color = "blue", label = "no noise", lw=2, alpha=0.8)
        plt.plot(time_0, noise_0, color = self.n_colors[0], label = r"$\sigma^2$=1.0e-10", lw=2)
        plt.plot(time_1, noise_1, color = self.n_colors[1], label = r"$\sigma^2$=1.0e-08", lw=2)
        #plt.plot(time_2, noise_2, color = self.n_colors[2], label = r"$\sigma^2$=1.0e-06", lw=2)
        #fitted
        plt.plot(time_3, self.linear_total_energy(time_3, a_3, b_3), '--', label=f"$y = {a_3:.8f}x {b_3:+.8f}$")
        plt.plot(time_0, self.linear_total_energy(time_0, a_0, b_0), '--', label=f"$y = {a_0:.8f}x {b_0:+.8f}$")
        plt.plot(time_1, self.linear_total_energy(time_1, a_1, b_1), '--', label=f"$y = {a_1:.8f}x {b_1:+.8f}$")
        #plt.plot(time_2, self.linear_total_energy(time_2, a_2, b_2), '--', label=f"$y = {a_2:.8f}x {b_2:+.8f}$")

        plt.xlim([self.t_0, self.t_max])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel('$\mathbf{\Delta\ Total\ Energy\ (eV)}$', fontsize = 16)
        ax.spines['right'].set_visible(True)
        plt.ylim([-0.05, 2.37])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.31), prop={'size': 12}, ncol=2, frameon=False)
        ax1 = ax.twinx()
        ax1.set_ylim([-0.05, 2.37])
        ax1.tick_params(labelsize=15)
        ax1.set_ylabel(" ")
        title = folder.replace('../','')
        plt.savefig(f"total_energy_fitted_{title}.pdf", bbox_inches='tight')
        plt.savefig(f"total_energy_fitted_{title}.png", bbox_inches='tight')
        plt.close()

    def energy_diff_slope_vs_dt_curve(self):
        dt = [7,12,25,50]
        var_00 = []
        var_08 = []
        var_07 = []
        var_06 = []
        var_05 = []
        err_00 = []
        err_08 = []
        err_07 = []
        err_06 = []
        err_05 = []
        for i in dt:
            field = open(f"ci_noise_linear_regression_noise_sa_oo_vqe_{i:03d}.out", 'r+')
            for line in field:
                if "a_00_mean:" in line:
                    var_00.append(float(line.split()[1]))
                elif "a_08_mean" in line:
                    var_08.append(float(line.split()[1]))
                elif "a_07_mean" in line:
                    var_07.append(float(line.split()[1]))
                elif "a_06_mean" in line:
                    var_06.append(float(line.split()[1]))
                elif "a_05_mean" in line:
                    var_05.append(float(line.split()[1]))
                elif "a_00_error:" in line:
                    err_00.append(float(line.split()[1]))
                elif "a_08_error:" in line:
                    err_08.append(float(line.split()[1]))
                elif "a_07_error:" in line:
                    err_07.append(float(line.split()[1]))
                elif "a_06_error:" in line:
                    err_06.append(float(line.split()[1]))
                elif "a_05_error:" in line:
                    err_05.append(float(line.split()[1]))
        dt = [0.07,0.12,0.25,0.5]
        plt.rcParams['font.size'] = self.fs_rcParams
        # Plot data points and connect them for each noise level
        plt.plot(dt, var_00, color='blue', label='no noise', lw=2, marker='D')
        plt.errorbar(dt, var_00, yerr=err_00, fmt="D", color='blue')  # Add error bars
        
        plt.plot(dt, var_08, color=self.n_colors[0], label=r"$\sigma^2$=1.0e-08", lw=2, marker='D')
        plt.errorbar(dt, var_08, yerr=err_08, fmt="D", color=self.n_colors[0])
        
        plt.plot(dt, var_07, color=self.n_colors[1], label=r"$\sigma^2$=1.0e-07", lw=2, marker='D')
        plt.errorbar(dt, var_07, yerr=err_07, fmt="D", color=self.n_colors[1])

        plt.plot(dt, var_06, color=self.n_colors[2], label=r"$\sigma^2$=1.0e-06", lw=2, marker='D')
        plt.errorbar(dt, var_06, yerr=err_06, fmt="D", color=self.n_colors[2])

        plt.plot(dt, var_05, color=self.n_colors[3], label=r"$\sigma^2$=1.0e-05", lw=2, marker='D')
        plt.errorbar(dt, var_05, yerr=err_05, fmt="D", color=self.n_colors[3])

        # Labels and title
        plt.xlabel('dt (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.ylabel('Slope (eV/fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), prop={'size': 14}, ncol=2)
        plt.savefig("energy_diff_slope_vs_dt_curve.pdf", bbox_inches='tight')
        plt.savefig("energy_diff_slope_vs_dt_curve.png", bbox_inches='tight')
        plt.close()

    def energy_diff_slope_vs_dt(self):
        #folders = ["../noise_sa_oo_vqe_007","../noise_sa_oo_vqe_012","../noise_sa_oo_vqe_025"]
        #for folder in folders:
        #    #noise
        #    time_0, noise_0, std_0 = self.get_noise_ave(folder,'variance_10/etot.dat')
        #    time_1, noise_1, std_1 = self.get_noise_ave(folder,'variance_08/etot.dat')
        #    time_2, noise_2, std_2 = self.get_noise_ave(folder,'variance_06/etot.dat')
        #    time_3, noise_3, std_3 = self.get_noise_ave(folder,'variance_00/etot.dat')
        #    #fitted
        #    params_0, bs_error_95_0 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_0, noise_0)
        #    a_0 = params_0[0]
        #    b_0 = params_0[1]
        #    params_1, bs_error_95_1 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_1, noise_1)
        #    a_1 = params_1[0]
        #    b_1 = params_1[1]
        #    params_2, bs_error_95_2 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_2, noise_2)
        #    a_2 = params_2[0]
        #    b_2 = params_2[1]
        #    params_3, bs_error_95_3 = self.confidence_interval_95_bootstrap(self.linear_total_energy,time_3, noise_3)
        #    a_3 = params_3[0]
        #    b_3 = params_3[1]
        sl_025 = [0.0007329738307754767,0.0007255966508638325,0.000853636977203613,0.005424441916756572]
        sl_012 = [0.00043295168761318714,0.0005008339963588384,0.0,0.0036646894254784776]
        sl_007 = [0.000501993347644126,0.0003075635065117048,0.0007321623116374798,0.003097630738128245]
        err_025 = [1.6306112513965758e-05,1.6548679769461388e-05,1.8380585871653624e-05,0.00012069119156790089]
        err_012 = [5.649392540810552e-06,6.013448430348327e-06,0.0,8.156184883128408e-05]
        err_007 = [4.751646555039267e-06,3.3010062331441695e-06,6.720950734316459e-06,6.980455796699024e-05]
        dt = [1,2,3]
        dt_labels = ["0.07","0.12","0.25"]

        plt.rcParams['font.size'] = self.fs_rcParams
        # Plot histograms for dt = 0.07
        plt.bar(dt[0], sl_007[0], align='center', edgecolor='blue', fill=False, label="", lw=2)
        plt.bar(dt[0], sl_007[1], align='center', edgecolor=self.n_colors[0], fill=False, label="", lw=2)
        plt.bar(dt[0], sl_007[2], align='center', edgecolor=self.n_colors[1], fill=False, label="", lw=2)
        plt.bar(dt[0], sl_007[3], align='center', edgecolor=self.n_colors[2], fill=False, label="", lw=2)
        ## Error Bar for dt = 0.07
        #plt.errorbar(dt[0], sl_007[0], yerr=err_007[0], fmt="D", color="blue")
        #plt.errorbar(dt[0], sl_007[1], yerr=err_007[1], fmt="D", color=self.n_colors[0])
        #plt.errorbar(dt[0], sl_007[2], yerr=err_007[2], fmt="D", color=self.n_colors[1])
        #plt.errorbar(dt[0], sl_007[3], yerr=err_007[3], fmt="D", color=self.n_colors[2])
        # Plot histograms for dt = 0.12
        plt.bar(dt[1], sl_012[0], align='center', edgecolor='blue', fill=False, label="", lw=2)
        plt.bar(dt[1], sl_012[1], align='center', edgecolor=self.n_colors[0], fill=False, label="", lw=2)
        plt.bar(dt[1], sl_012[2], align='center', edgecolor=self.n_colors[1], fill=False, label="", lw=2)
        plt.bar(dt[1], sl_012[3], align='center', edgecolor=self.n_colors[2], fill=False, label="", lw=2)
        ## Error Bar for dt = 0.12
        #plt.errorbar(dt[1], sl_012[0], yerr=err_012[0], fmt="D", color="blue")
        #plt.errorbar(dt[1], sl_012[1], yerr=err_012[1], fmt="D", color=self.n_colors[0])
        #plt.errorbar(dt[1], sl_012[2], yerr=err_012[2], fmt="D", color=self.n_colors[1])
        #plt.errorbar(dt[1], sl_012[3], yerr=err_012[3], fmt="D", color=self.n_colors[2])
        # Plot histograms for dt = 0.25
        plt.bar(dt[2], sl_025[0], align='center', edgecolor='blue', fill=False, label='no noise', lw=2)
        plt.bar(dt[2], sl_025[1], align='center', edgecolor=self.n_colors[0], fill=False, label=r"$\sigma^2$=1.0e-10", lw=2)
        plt.bar(dt[2], sl_025[2], align='center', edgecolor=self.n_colors[1], fill=False, label=r"$\sigma^2$=1.0e-08", lw=2)
        plt.bar(dt[2], sl_025[3], align='center', edgecolor=self.n_colors[2], fill=False, label=r"$\sigma^2$=1.0e-06", lw=2)
        ## Error Bar for dt = 0.25
        #plt.errorbar(dt[2], sl_025[0], yerr=err_025[0], fmt="D", color="blue")
        #plt.errorbar(dt[2], sl_025[1], yerr=err_025[1], fmt="D", color=self.n_colors[0])
        #plt.errorbar(dt[2], sl_025[2], yerr=err_025[2], fmt="D", color=self.n_colors[1])
        #plt.errorbar(dt[2], sl_025[3], yerr=err_025[3], fmt="D", color=self.n_colors[2])
        
        # Labels and title
        plt.xticks(dt, dt_labels)
        plt.xlabel('dt (fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.ylabel('Slope (eV/fs)', fontweight = 'bold', fontsize =self.f_size)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), prop={'size': 14}, ncol=2)
        plt.savefig("energy_diff_slope_vs_dt.pdf", bbox_inches='tight')
        plt.savefig("energy_diff_slope_vs_dt.png", bbox_inches='tight')
        plt.close()
        
    
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
    #noise_sa_oo_vqe_050 = "../noise_sa_oo_vqe_050"
    #noise_sa_oo_vqe_025 = "../noise_sa_oo_vqe_025"
    #noise_sa_oo_vqe_012 = "../noise_sa_oo_vqe_012"
    #noise_sa_oo_vqe_007 = "../noise_sa_oo_vqe_007"
    method = os.getcwd()
    #time in fs
    t_0 = 0
    t_max = 200 
    lower_2_a = "True"
    out = PlotComb(t_0, t_max, lower_2_a)
    #out.plot_population_adi(index,xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.plot_1d_histogram(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    #out.plot_1d_histogram_2_plots(xms_caspt2,sa_casscf,sa_oo_vqe, 17)
    #out.plot_1d_histogram_2_plots_samen(xms_caspt2,sa_casscf,sa_oo_vqe, 8)
    #out.plot_1d_histogram_2_plots_samen_energy(xms_caspt2,sa_casscf,sa_oo_vqe, 20)
    #out.plot_1d_histogram_2_plots_energy(xms_caspt2,sa_casscf,sa_oo_vqe, 31)
    ##out.plot_1d_curves(xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.plot_1d_histogram_4_plots_S1_S0(xms_caspt2,sa_casscf,sa_oo_vqe)
    #out.print_stat(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_torsion_ave(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_torsion_ave_qy(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_population_adi_fitted(sa_casscf)
    #out.plot_population_adi_fitted(sa_oo_vqe)
    #out.plot_av_popu_torsion_bend(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_variance_noise(noise_sa_oo_vqe)
    #out.plot_av_popu_noise(noise_sa_oo_vqe)
    ##noise
    for i in ["007","012","025","050"]:
        folder = "../noise_sa_oo_vqe_" + i
        out.plot_av_popu_torsion_noise(folder)
    ##noise
    #out.plot_av_popu_diff_ene(xms_caspt2, sa_casscf, sa_oo_vqe)
    #out.plot_one_method_av_popu_diff_ene(method)
    #out.get_torsion_qy_ave(xms_caspt2)
    #out.get_torsion_qy_ave(sa_oo_vqe)
    #out.get_torsion_qy_ave(sa_casscf)
    #out.get_torsion_qy_ave_2(xms_caspt2)
    #out.get_torsion_qy_ave_2(sa_oo_vqe)
    #out.get_torsion_qy_ave_2(sa_casscf)
    #out.get_torsion_qy_ave_noise(noise_sa_oo_vqe)
    #out.plot_total_energy_fitted(noise_sa_oo_vqe)
    #out.energy_diff_slope_vs_dt()
    ##out.energy_diff_slope_vs_dt_curve()
    ##out.plot_1d_histogram_QY_time(xms_caspt2,sa_casscf,sa_oo_vqe, 7)
    ##out.plot_2d_histogram_QY_time(xms_caspt2,sa_casscf,sa_oo_vqe, 7)
