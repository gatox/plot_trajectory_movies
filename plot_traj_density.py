import csv
import sys
from time import sleep
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
fs_rcParams = '12'
t_0 = 0
t_max = 200 #fs
skip = "yes"

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

def _high_360(y_i,y_f):
    if y_i>300 and y_f<50 and y_f<y_i: 
        y_nan = y_f*np.nan
    elif y_i<50 and y_f>300 and y_f>y_i:
        y_nan = y_f*np.nan
    else:
        y_nan = y_f
    return y_nan

def _check_radian(name, var):
    if "dihe" or "angle" or "pyr" in name:
        if var < 0:
            var = 360 + var
        x = np.radians(var)
    else:
        x =  var
    return x

def _check_angle_scatter(x,y):
    if "dihe" or "angle" or "pyr" in name:
        a = y
        b = x
    else:
        a = x
        b = y
    return a,b

def _check_angle_plot(w,x,y,z):
    if "dihe" or "angle" or "pyr" in name:
        a = y
        b = z
        c = w
        d = x
    else:
        a = w
        b = x
        c = y
        d = z
    return a,b,c,d

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
    #for i in range(0,trajs,8):          #trajectories
        for j in range(1,mdsteps):   #time_steps 
            x = time[j]
            if "dihe_polar_2" in name:
                if para[j,i]<0:
                    param = 360 + para[j,i] 
                else:
                    param = para[j,i] 
            else:
                param = _check_radian(name, para[j,i]) 
                x, param = _check_angle_scatter(x,param)
            if hop[j-1,i]==1 and hop[j,i]==0:
                hop_10 = plt.scatter(x,param,color='purple', marker='x',s=17, alpha=1,zorder=10)
            elif hop[j-1,i]==0 and hop[j,i]==1:
                hop_01 = plt.scatter(x,param,color='gold', marker='+',s=17, alpha=1, zorder=10)
    return [hop_10,hop_01]

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
    #for i in range(0,trajs,8):          #trajectories
        tra_neg =1000
        for j in range(1,mdsteps):   #time_steps 
            t_i = time[j-1]
            t_f = time[j]
            if "dihe_polar_2" in name:
                if para[j-1,i]<0:
                    p_i = 360 + para[j-1,i] 
                else:
                    p_i = para[j-1,i] 
                if para[j,i]<0:
                    p_f = 360 + para[j,i] 
                else:
                    p_f = para[j,i] 
                p_f = _high_360(p_i,p_f)
                #if j < 100 and hop[j,i]==1:
                #    if 300< p_i <325 and 300<p_f<325 and p_i >p_f: 
                #        tra_neg = i
                #if i == tra_neg:
                #    p_i = -p_i
                #    p_f = -p_f
            else:
                p_i = _check_radian(name,para[j-1,i])
                p_f = _check_radian(name,para[j,i])
                t_i,t_f,p_i,p_f = _check_angle_plot(t_i,t_f,p_i,p_f)
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

def broadening_gaussian(x, x0, intensity, sigma):
    """Apply gaussian broadening"""
    return intensity/sigma * np.exp(- ((x-x0)/sigma)**2)


def broaden(x0, start, stop, nsteps, sigma):
    x = np.linspace(start, stop, nsteps)
    if x0 == 'nan':
        return None
    x0 = float(x0)
    y = np.array([broadening_gaussian(_x, x0, 1.0, sigma) for _x in x])
    scale = np.sum(y)
    if scale == 0:
        return y
    return y/scale

def get_heatmap(filename, start, stop, nsteps=101, sigma=0.03):
    x = np.linspace(start, stop, nsteps)
    times = []
    yvales = []
    with open(filename, 'r') as fh:
        reader = csv.DictReader(fh)
        for row in tqdm(reader):
            times.append(float(row['time']))
            y = np.zeros(x.shape)
            for k, val in row.items():
                if k == 'time':
                    continue
                _y = broaden(val, start, stop, nsteps, sigma)
                if _y is not None:
                    y += _y
            yvales.append(y)
            sleep(0.0001)
    return x, np.array(times), np.array(yvales).T

def create_heatmap(filename, xstart, xstop, nsteps=101, sigma=0.03, cmap='nipy_spectral_r', outfile='r12.pdf', legend='yes'):
    plt.rcParams['font.size'] = fs_rcParams
    fig, ax = plt.subplots()
    xs, times, values = get_heatmap(filename, xstart, xstop, nsteps=nsteps, sigma=sigma)
    t, y = np.meshgrid(times, xs)
    #c = ax.pcolormesh(t, y, values, cmap=cmap, vmin=0., vmax=np.max(values))
    c = ax.pcolormesh(t, y, values, cmap=cmap, vmin=0., vmax=14)
    if legend =="yes" and filename !="etot.dat":
        fig.colorbar(c, ticks=np.linspace(0, 14, 8, dtype=int))
    else: 
        fig.colorbar(c, ticks=np.linspace(0, 14, 8, dtype=int), location = 'right', anchor=(1.14, 0.5))
        #fig.colorbar(c, ticks=np.linspace(0, np.max(values), 10, dtype=int))
    if filename == "dis_r12.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-N_2 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r13.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-H_3 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r14.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-H_4 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r25.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{N_2-H_5 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_3014.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle H_4C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_2014.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "angle_014.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([-1, 180])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "etot.dat":
        #handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        ax.spines['right'].set_visible(True)
        ax.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
        #labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        #plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{Energy (eV)}$', fontsize = fs_ylabel) 
        ax1 = ax.twinx()
        ax1.set_ylim([xstart, xstop])
        ax1.set_ylabel(" ")
        plt.savefig(outfile, bbox_inches='tight')
        plt.savefig(outfile.replace(".pdf",".png"), bbox_inches='tight')
        plt.close()
    elif filename == "pyr_3210.dat":
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()

def _no_density_plots(filename, xstart, xstop, outfile, legend='yes'):
    plt.rcParams['font.size'] = fs_rcParams
    if filename == "dis_r12.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-N_2 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r13.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-H_3 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r14.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{C_1-H_4 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dis_r25.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{N_2-H_5 (\AA)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_3014.dat":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Set radial ticks at selected angles
        ax.set_xticks(np.radians(np.arange(xstart, xstop,90)))
        ax.set_xticklabels(['0°', '90°', '180°', '270°'])
        traj_leb = _plot_traj(filename.replace('.dat',''))
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=4)
        #plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        #plt.ylabel('$\mathbf{\sphericalangle H_4C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_2013.dat":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Set radial ticks at selected angles
        ax.set_xticks(np.radians(np.arange(xstart, xstop,90)))
        ax.set_xticklabels(['0°', '90°', '180°', '270°'])
        traj_leb = _plot_traj(filename.replace('.dat',''))
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=4)
        #plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        #plt.ylabel('$\mathbf{\sphericalangle H_4C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_polar.dat":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Set radial ticks at selected angles
        ax.set_ylim(0, 100)
        ax.set_yticks(np.array([0,25,50,75,100]))
        ax.set_yticklabels(['0', '', '50', '75', '100'])
        #ax.set_yticks(np.array([0,25,50,75,100,125,150,200]))
        #ax.set_yticklabels(['0', '', '50', '75', '100', '125', '150', '200'])
        ax.set_xticks(np.radians(np.arange(xstart, xstop,90)))
        ax.set_xticklabels(['0°', '90°', '180°', '270°'])
        traj_leb = _plot_traj(filename.replace('.dat',''))
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        ax.grid(alpha=0.4)
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 12}, ncol=4)
        #plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        #plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "dihe_polar_2.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.axhline(90,label="",linestyle='--', c = 'black')
        plt.axhline(270,label="",linestyle='--', c = 'black')
        plt.xlim([t_0, t_max])
        plt.ylim([-1, 360])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$", "$S_1$", "$S_0$"]
        plt.legend(traj_leb,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=4)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle H_3C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "angle_014.dat":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([-1, 180])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{\sphericalangle C_1N_2H_5 (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()
    elif filename == "etot.dat":
        fig, ax = plt.subplots()
        traj_leb = _plot_traj(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
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
    elif filename == "pyr_3210.dat":
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        handles = _add_hops_to_density(filename.replace('.dat',''))
        plt.xlim([t_0, t_max])
        plt.ylim([xstart, xstop])
        labels = [r"$S_1$ $\rightarrow$ $S_0$",r"$S_0$ $\rightarrow$ $S_1$"]
        plt.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 12}, ncol=2)
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = fs_xlabel)
        plt.ylabel('$\mathbf{Pyramidalization (degrees)}$', fontsize = fs_ylabel) 
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()

def _density_plots(results, label):
    for name in results:
        if "dis" in  name:
            print("Making density plot:",name+".dat")
            create_heatmap(name+".dat", 0.5, 2.6, nsteps=500, sigma=0.001, outfile=label+"_"+name+".pdf", legend="yes")
        if "dihe" in name:
            print("Making density plot:",name+".dat")
            create_heatmap(name+".dat", -180, 180, nsteps=300, sigma=0.1, outfile=label+"_"+name+".pdf", legend="yes")
        if "angle" in name:
            print("Making density plot:",name+".dat")
            create_heatmap(name+".dat", -180, 180, nsteps=300, sigma=0.1, outfile=label+"_"+name+".pdf", legend="yes")
        if "etot" in name:
            print("Making density plot:",name+".dat")
            #create_heatmap(name+".dat", -1, 1, nsteps=20, sigma=0.01, outfile=label+"_"+name+".pdf", legend="yes")
            create_heatmap(name+".dat", -1, 1, nsteps=200, sigma=0.001, outfile=label+"_"+name+".pdf", legend="yes")
        if "pyr" in name:
            print("Making density plot:",name+".dat")
            create_heatmap(name+".dat", -180, 180, nsteps=300, sigma=0.1, outfile=label+"_"+name+".pdf", legend="yes")

def _traj_plots(results, label):
    for name in results:
        if "dis" in  name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", 0.5, 2.6, outfile=label+"_"+name+".pdf", legend="yes")
        if "dihe" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", 0, 360, outfile=label+"_"+name+".pdf", legend="yes")
        if "angle" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", 0, 360, outfile=label+"_"+name+".pdf", legend="yes")
        if "etot" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", -1, 1, outfile=label+"_"+name+".pdf", legend="yes")
        if "pyr" in name:
            print("Making plot:",name+".dat")
            _no_density_plots(name+".dat", 0, 360, outfile=label+"_"+name+".pdf", legend="yes")

if __name__ == '__main__':
    label = sys.argv[1] 
    #plot_type = sys.argv[2]
    plot_type = "density" 
    #results = ["dihe_3014", "dihe_2014", "etot", "dis_r12", "dis_r25", "dis_r14", "dis_r13", "angle_014", "pyr_3210"]
    #results = ["dihe_3014", "dihe_2014", "etot", "dis_r12", "dis_r25", "dis_r14", "dis_r13"] 
    #results = ["dihe_polar_2", "dihe_polar"]
    #results = ["dihe_polar_2"]
    results = ["etot"]
    if plot_type == "density":
        _density_plots(results, label)
    elif plot_type == "traj":
        _traj_plots(results, label)
