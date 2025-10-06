import sys
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from matplotlib import gridspec
# from jinja2 import Template #build templates
import numpy as np
# from numpy import copy 
from pysurf.database import PySurfDB
# from pysurf.system import Molecule
# from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
# import matplotlib.animation as animation
# from moviepy.editor import VideoFileClip, CompositeVideoClip, clips_array
# from collections import namedtuple

class PlotsH2:
    
    def __init__(self):
        self.fs = 0.02418884254
        self.ev = 27.211324570273
        self.aa = 0.529177208
        self.fs_rcParams = '20'
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
        self.markers = list(Line2D.filled_markers)[:8]
        self.titles = ["Noisless","Noise/Conv_Tol: 1.0e-2","Noise/Conv_Tol: 1.0e-3", "Noise/Conv_Tol: 1.0e-4"]
        
    def read_db(self, output):
        db = PySurfDB.load_database(output, read_only=True)
        time = db["time"][0:]*self.fs
        return db, time
    
    def dis_two_atmos(self, data, atom_1, atom_2):
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        dimer = []
        for i,m in enumerate(data):
            dimer.append(float(m[atom_2][2])-float(m[atom_1][2]))
        return dimer
    
    def plot_pos_vel_3db(self,output_1,output_2,output_3,output_4):
        db_1, _ = self.read_db(output_1)
        db_2, _ = self.read_db(output_2)
        db_3, _ = self.read_db(output_3)
        db_4, _ = self.read_db(output_4)
        fig, ax = plt.subplots()
        crd_1 = np.array(self.dis_two_atmos(db_1["crd"], 0, 1))
        vel_1 = np.array(self.dis_two_atmos(db_1["veloc"], 0, 1))
        crd_2 = np.array(self.dis_two_atmos(db_2["crd"], 0, 1))
        vel_2 = np.array(self.dis_two_atmos(db_2["veloc"], 0, 1))
        crd_3 = np.array(self.dis_two_atmos(db_3["crd"], 0, 1))
        vel_3 = np.array(self.dis_two_atmos(db_3["veloc"], 0, 1))
        crd_4 = np.array(self.dis_two_atmos(db_4["crd"], 0, 1))
        vel_4 = np.array(self.dis_two_atmos(db_4["veloc"], 0, 1))
        ax.plot(crd_1, vel_1 / 1e-3, color=self.colors[0], label = self.titles[0])
        ax.plot(crd_2, vel_2 / 1e-3, color=self.colors[1], linestyle='--', label = self.titles[1])
        ax.plot(crd_3, vel_3 / 1e-3, color=self.colors[2], linestyle='--', label = self.titles[2])
        ax.plot(crd_4, vel_4 / 1e-3, color=self.colors[3], linestyle='--', label = self.titles[3])
        ax.scatter(crd_1[0], vel_1[0] / 1e-3, color='r', marker=self.markers[0], s=40, label='')
        ax.scatter(crd_1[-1], vel_1[-1] / 1e-3, color='g', marker=self.markers[1], s=40, label='')
        ax.scatter(crd_2[0], vel_2[0] / 1e-3, color='r', marker=self.markers[2], s=40, label='')
        ax.scatter(crd_2[-1], vel_2[-1] / 1e-3, color='g', marker=self.markers[3], s=40, label='')
        ax.scatter(crd_3[0], vel_3[0] / 1e-3, color='r', marker=self.markers[4], s=40, label='')
        ax.scatter(crd_3[-1], vel_3[-1] / 1e-3, color='g', marker=self.markers[5], s=40, label='')
        ax.scatter(crd_4[0], vel_4[0] / 1e-3, color='r', marker=self.markers[6], s=40, label='')
        ax.scatter(crd_4[-1], vel_4[-1] / 1e-3, color='g', marker=self.markers[7], s=40, label='')

        ax.set_xlabel('Position (a.u.)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Velocity (a.u.)', fontweight='bold', fontsize=16)

        # Add scaling factor note for velocity axis
        ax.text(
            0.0, 1.0, "1e-3",
            transform=ax.transAxes,  # relative to axes coords
            ha='left', va='bottom',
            fontsize=12
        )
        plt.title("H2_dynamics/STO-3G/PNOF4/1000_shots/AER/IBM_pittsburgh/Opt_lvel=3", y=1.2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.195), prop={'size': 12}, ncol=2,frameon=False)
        fig.savefig("post_vel_h2_3.pdf", bbox_inches='tight')
        plt.close(fig)
    
    def plot_pos_vel(self, output):
        db, _ = self.read_db(output)
        fig, ax = plt.subplots()
        crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
        vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))
        # # Convert velocity to momentum (multiply by H mass in a.u.)
        # m_H = 1837.15258739092  # hydrogen atom mass in a.u.
        # mom = vel * m_H
        ax.plot(crd, vel / 1e-3, color='blue')
        ax.scatter(crd[0], vel[0] / 1e-3, color='r', s=40, label='Initial point')
        ax.scatter(crd[-1], vel[-1] / 1e-3, color='g', s=40, label='Final point')

        ax.set_xlabel('Position (a.u.)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Velocity (a.u.)', fontweight='bold', fontsize=16)

        # Add scaling factor note for velocity axis
        ax.text(
            0.0, 1.0, "1e-3",
            transform=ax.transAxes,  # relative to axes coords
            ha='left', va='bottom',
            fontsize=12
        )
        ax.legend()
        fig.savefig("post_vel_h2.pdf", bbox_inches='tight')
        plt.close(fig)
        
    def plot_time_total_energy_3db(self, output_1,output_2,output_3,output_4):
        db_1, time_1 = self.read_db(output_1)
        db_2, time_2 = self.read_db(output_2)
        db_3, time_3 = self.read_db(output_3)
        db_4, time_4 = self.read_db(output_4)
        etot_1 = np.array(list(db_1["etot"]), dtype=float)
        etot_2 = np.array(list(db_2["etot"]), dtype=float)
        etot_3 = np.array(list(db_3["etot"]), dtype=float)
        etot_4 = np.array(list(db_4["etot"]), dtype=float)
        plt.plot(time_1,(etot_1-etot_1[0])/1e-3,color=self.colors[0], label = self.titles[0])
        plt.plot(time_2,(etot_2-etot_2[0])/1e-3,color=self.colors[1], linestyle='--', label = self.titles[1]) 
        plt.plot(time_3,(etot_3-etot_3[0])/1e-3,color=self.colors[2], linestyle='--', label = self.titles[2]) 
        plt.plot(time_4,(etot_4-etot_4[0])/1e-3,color=self.colors[3], linestyle='--', label = self.titles[3])
        plt.axhline(0,linestyle=':', c = 'black')
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel(r'$\Delta$ Total Energy (mHa)', fontweight = 'bold', fontsize = 16)
        plt.xlim([0, 10])
        plt.title("H2_dynamics/STO-3G/PNOF4/1000_shots/AER/IBM_pittsburgh/Opt_lvel=3", y=1.2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), prop={'size': 12}, ncol=2,frameon=False)
        plt.savefig("time_etotal_h2_3.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_time_total_energy(self, output):
        db, time = self.read_db(output) 
        etot = np.array(list(db["etot"]), dtype=float)
        plt.plot(time,(etot-etot[0])/1e-3,color='blue',label = '') 
        plt.axhline(0,linestyle=':', c = 'black')
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel(r'$\Delta$ Total Energy (mHa)', fontweight = 'bold', fontsize = 16)
        plt.xlim([0, 10])
        plt.savefig("time_etotal_h2.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_time_parmameter_3db(self, output_1,output_2,output_3,output_4):
        db_1, time_1 = self.read_db(output_1)
        db_2, time_2 = self.read_db(output_2)
        db_3, time_3 = self.read_db(output_3)
        db_4, time_4 = self.read_db(output_4)
        parameter_1 = np.array(list(db_1["parameter"]), dtype=float)
        parameter_2 = np.array(list(db_2["parameter"]), dtype=float)
        parameter_3 = np.array(list(db_3["parameter"]), dtype=float)
        parameter_4 = np.array(list(db_4["parameter"]), dtype=float)
        plt.plot(time_1,parameter_1,color=self.colors[0], label = self.titles[0])
        plt.plot(time_2,parameter_2,color=self.colors[1], linestyle='--', label = self.titles[1]) 
        plt.plot(time_3,parameter_3,color=self.colors[2], linestyle='--', label = self.titles[2])
        plt.plot(time_4,parameter_4,color=self.colors[3], linestyle='--', label = self.titles[3]) 
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel(r'Optimal $\boldsymbol{\theta}$', fontweight = 'bold', fontsize=16)
        plt.xlim([0, 10])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19), prop={'size': 12}, ncol=2,frameon=False)
        plt.title("H2_dynamics/STO-3G/PNOF4/1000_shots/AER/IBM_pittsburgh/Opt_lvel=3", y=1.2)
        plt.savefig("time_parameter_h2_3.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_time_parmameter(self, output):
        db, time = self.read_db(output)
        parameter = np.array(list(db["parameter"]), dtype=float)
        plt.plot(time,parameter,color='blue',label = '') 
        plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        plt.ylabel(r'Optimal $\boldsymbol{\theta}$', fontweight = 'bold', fontsize=16)
        plt.xlim([0, 10])
        plt.savefig("time_parameter_h2.pdf", bbox_inches='tight')
        plt.close()

if __name__=="__main__":
    db_1 = sys.argv[1]  #results.db  
    db_2 = sys.argv[2]
    db_3 = sys.argv[3]
    db_4 = sys.argv[4]
    picture = PlotsH2()
    #picture.plot_pos_vel(db_1)
    #picture.plot_time_total_energy(db_1)
    #picture.plot_time_parmameter(db_1)
    picture.plot_pos_vel_3db(db_1,db_2,db_3,db_4)
    picture.plot_time_total_energy_3db(db_1,db_2,db_3,db_4)
    picture.plot_time_parmameter_3db(db_1,db_2,db_3,db_4)
    
