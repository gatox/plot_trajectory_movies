import csv
import sys
from time import sleep
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pandas import (read_csv, DataFrame)
from collections import (namedtuple, Counter)
from matplotlib.colors import ListedColormap


fs = 0.02418884254
aa = 0.529177208 
ev = 27.211324570273
fs_ylabel = 18
fs_xlabel = 18
fs_yticks = 18
fs_xticks = 18
fs_rcParams = '14'
t_0 = 0
t_max = 200 #fs
skip = "yes"
#skip = "not"

def compare(val, comp, ep):
    lower = np.abs(val-ep) 
    upper = np.abs(val+ep)
    myBool = True
    if comp > lower and comp < upper:
        return myBool
    return not myBool

def no_time(data):
    res = read_csv(data)       
    res = res.to_numpy()[:,1:] # removing time column
    return res 

def con_int_two_angles(angle_1, angle_2):
    angle_alpha = read_csv(angle_1)       
    angle_phi = read_csv(angle_2)         
    angle_alpha = angle_alpha.to_numpy()[:,1:] # removing time column
    angle_phi = angle_phi.to_numpy()[:,1:] # removing time column
    return angle_alpha, angle_phi

def skip_traj(skip):
    if skip == "yes":
        trajs = []
        read = open("trajectories_ok_list", 'r+') 
        return [line.strip().split("/")[2] for line in read]
    return None

def read_prop():
    sampling = open("sampling.inp", 'r+')    
    for line in sampling:
        if "n_conditions" in line:
            trajs = int(line.split()[2])
    traj_allowed = skip_traj(skip)
    if  skip == "yes":
        trajs = len(traj_allowed)
    prop = open("prop.inp", 'r+')    
    tully = None
    for line in prop:
        if "dt" in line:
            tully = "yes"
            dt = int(float(line.split()[2]))
        elif "mdsteps" in line:
            mdsteps = int(line.split()[2])
        elif "nstates" in line:
            nstates = int(line.split()[2])
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
                properties = namedtuple("properties", "dt mdsteps nstates states trajs")
                return properties(dt, mdsteps, nstates, states, trajs)
            elif method == "LandauZener":
                properties = namedtuple("properties", "dt mdsteps nstates states trajs")
                return properties(timestep/fs, int(time_final/timestep), nstates, [i for i in range(nstates)], trajs)

def _add_hops_to_density(name):
    prop = read_prop()
    trajs = prop.trajs
    mdsteps = prop.mdsteps
    nstates = prop.nstates
    pop_name = "pop.dat"
    pop = read_csv(pop_name)
    time = pop['time']
    time = time.to_numpy()
    hop = pop.to_numpy()[:,1:] # removing time column
    para = no_time(name+".dat") 
    for i in range(trajs):          #trajectories
        for j in range(1,mdsteps):   #time_steps 
            x = time[j]
            param = para[j,i] 
            if hop[j-1,i]==1 and hop[j,i]==0:
                hop_10 = plt.scatter(x,param,color='purple', marker='x',s=17, alpha=1,zorder=10)
            elif hop[j-1,i]==0 and hop[j,i]==1:
                hop_01 = plt.scatter(x,param,color='gold', marker='+',s=17, alpha=1, zorder=10)
    return [hop_10,hop_01]

def _high_180(y_i,y_f):
    if y_i>150 and y_f<0: 
        y_nan = y_f*np.nan
    elif y_i<-150 and y_f>0:
        y_nan = y_f*np.nan
    else:
        y_nan = y_f
    return y_nan

def _plot_traj(name):
    prop = read_prop()
    trajs = prop.trajs
    mdsteps = prop.mdsteps
    nstates = prop.nstates
    pop_name = "pop.dat"
    pop = read_csv(pop_name)
    time = pop['time']
    time = time.to_numpy()
    hop = pop.to_numpy()[:,1:] # removing time column
    para = no_time(name+".dat") 
    for i in range(trajs):          #trajectories
        for j in range(1,mdsteps):   #time_steps 
            t_i = time[j-1]
            t_f = time[j]
            p_i = para[j-1,i]
            p_f = _high_180(para[j-1,i],para[j,i])
            print(t_i,t_f,p_i,p_f)
            if hop[j-1,i]==1:
                color = '#ff7f0e' #orange
                s_1, = plt.plot([t_i,t_f], [p_i,p_f],color=color, linewidth=0.6, alpha=0.8)
            elif hop[j-1,i]==0:
                color = '#1f77b4' #blue
                s_0, = plt.plot([t_i,t_f], [p_i,p_f],color=color, linewidth=0.6, alpha=0.8)
    if "etot" in name: 
        return [s_1, s_0]
    else:
        hop_leb = _add_hops_to_density(name) 
        return [hop_leb[0], hop_leb[1], s_1, s_0]

def _no_density_three_plots_dist(file_0,file_1,file_2):
    plt.rcParams['font.size'] = '20' 
    plt.figure(figsize=(6,12))
    # set height ratios for subplots
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    # the first subplot
    ax0 = plt.subplot(gs[0])
    ax0.set_ylabel("$\mathbf{C_1-N_2 (\AA)}$", fontweight = 'bold', fontsize = 15)
    ax0.set_ylim([0.5, 2.8])
    traj_leb_0 = _plot_traj(file_0.replace('.dat',''))

    # the second subplot
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.set_ylabel("$\mathbf{N_2-H_5 (\AA)}$", fontweight = 'bold', fontsize = 15)
    ax1.set_ylim([0.5, 2.8])
    traj_leb_1 = _plot_traj(file_1.replace('.dat',''))
    plt.setp(ax0.get_xticklabels(), visible=False)

    # the third subplot
    ax2 = plt.subplot(gs[2], sharex = ax0)
    ax2.set_ylabel("$\mathbf{C_1-H_3 (\AA)}$", fontweight = 'bold', fontsize = 15)
    ax2.set_ylim([0.5, 2.8])
    traj_leb_2 = _plot_traj(file_2.replace('.dat',''))

    plt.setp(ax1.get_xticklabels(), visible=False)
    # put legend on first subplot
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)
    labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
    ax0.legend(traj_leb_2,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=4)

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.xlim([0, 200])
    plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 15)
    plt.savefig("CN_NH_CH_time.pdf", bbox_inches='tight')
    #plt.savefig("CN_NH_CH_time.png", bbox_inches='tight')
    plt.close()

def _no_density_three_plots(file_0,file_1,file_2):
    plt.rcParams['font.size'] = '20' 
    plt.figure(figsize=(6,12))
    # set height ratios for subplots
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    # the first subplot
    ax0 = plt.subplot(gs[0])
    ax0.set_ylabel("$\mathbf{Pyram.}$", fontweight = 'bold', fontsize = 15)
    ax0.set_ylim([-180, 180])
    traj_leb_0 = _plot_traj(file_0.replace('.dat',''))

    # the second subplot
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.set_ylabel("$\mathbf{\sphericalangle C_1N_2H_5}$", fontweight = 'bold', fontsize = 15)
    ax1.set_ylim([-1, 180])
    traj_leb_1 = _plot_traj(file_1.replace('.dat',''))
    plt.setp(ax0.get_xticklabels(), visible=False)

    # the third subplot
    ax2 = plt.subplot(gs[2], sharex = ax0)
    ax2.set_ylabel("$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$", fontweight = 'bold', fontsize = 15)
    ax2.set_ylim([-180, 180])
    traj_leb_2 = _plot_traj(file_2.replace('.dat',''))

    plt.setp(ax1.get_xticklabels(), visible=False)
    # put legend on first subplot
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=3)
    labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
    ax0.legend(traj_leb_2,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=4)

    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.xlim([0, 200])
    plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
    plt.savefig("Pyra_Ange_Dihe_time.pdf", bbox_inches='tight')
    #plt.savefig("Pyra_Ange_Dihe_time.png", bbox_inches='tight')
    plt.close()

def _plots(filename, xstart, xstop, outfile):
    traj_leb = _plot_traj(filename.replace('.dat',''))
    plt.xlim([t_0, t_max])
    plt.ylim([xstart, xstop])
    labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
    plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
    plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)

def _plot_energy(filename, xstart, xstop, t_f, outfile, ax):
    #traj_leb = _plot_traj(filename.replace('.dat',''))
    name = filename.replace('.dat','')
    prop = read_prop()
    trajs = prop.trajs
    mdsteps = prop.mdsteps
    nstates = prop.nstates
    pop_name = "pop.dat"
    pop = read_csv(pop_name)
    time = pop['time']
    time = time.to_numpy()
    hop = pop.to_numpy()[:,1:] # removing time column
    para = no_time(name+".dat") 
    print(len(para),len(hop), trajs, mdsteps)
    for i in range(trajs):          #trajectories
        for j in range(1,mdsteps):   #time_steps 
            t_i = time[j-1]
            t_f = time[j]
            p_i = para[j-1,i]
            p_f = _high_180(para[j-1,i],para[j,i])
            if hop[j-1,i]==1:
                color = '#ff7f0e' #orange
                plt.plot([t_i,t_f], [p_i,p_f],color=color, linewidth=0.6, alpha=0.8)
            elif hop[j-1,i]==0:
                color = '#1f77b4' #blue
                plt.plot([t_i,t_f], [p_i,p_f],color=color, linewidth=0.6, alpha=0.8)
    
    #plt.xlim([t_0, t_max])
    plt.xlim([t_0, t_f])
    plt.ylim([xstart, xstop])
    ax.spines['right'].set_visible(True)
    ax.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
    #labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
    #plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
    plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
    plt.ylabel('$\mathbf{Energy (eV)}$', fontsize = fs_ylabel) 
    ax1 = ax.twinx()
    ax1.set_ylim([xstart, xstop])
    ax1.set_ylabel(" ")
    plt.savefig(outfile, bbox_inches='tight')
    plt.savefig(outfile.replace(".pdf",".png"), bbox_inches='tight')
    plt.close()

def _no_density_plots(filename, xstart, xstop, t_f, outfile):
    plt.rcParams['font.size'] = fs_rcParams
    fig, ax = plt.subplots()
    if filename == "dis_r12.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{C_1-N_2 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r13.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{C_1-H_3 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r14.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{C_1-H_4 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r25.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{N_2-H_5 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_3014.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{\sphericalangle H_4C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_2013.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{\sphericalangle C_3N_1N_2C_4 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_2014.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "angle_014.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "etot.dat":
        _plot_energy(filename, xstart, xstop, t_f, outfile, ax)
    elif filename == "pyr_3210.dat":
        _plots(filename, xstart, xstop, outfile)
        plt.ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()

def traj_plots(results, label):
    for name in results:
        if "dis" in  name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", 0.5, 2.6, outfile=label+"_"+name+".pdf")
        if "dihe" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", -180, 180, outfile=label+"_"+name+".pdf")
        if "angle" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", -1, 180, outfile=label+"_"+name+".pdf")
        if "etot" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", -1, 1, 200, outfile=label+"_"+name+".pdf")
        if "pyr" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", -180, 180, outfile=label+"_"+name+".pdf")

if __name__ == '__main__':
    ##label = sys.argv[1] 
    #results = ["etot", "dis_r12", "dis_r25", "dis_r14", "dis_r13", "angle_014", "pyr_3210"]
    #results = ["dihe_3014", "dihe_2014", "etot", "dis_r12", "dis_r25", "dis_r14", "dis_r13", "angle_014", "pyr_3210"]
    results = ["etot"]
    traj_plots(results, "etot")
    #traj_plots(results, label)
    #_no_density_three_plots("pyr_3210","angle_014","dihe_2014")
    #_no_density_three_plots_dist("dis_r12","dis_r25","dis_r13")
