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
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:8]
        self.markers = list(Line2D.filled_markers)
        #self.titles = ["Noisless","Noise/Conv_Tol: 1.0e-2","Noise/Conv_Tol: 1.0e-3", "Noise/Conv_Tol: 1.0e-4"]
        #self.titles = ["Noisless","Noise/Conv_Tol: 1.0e-2","Real/Conv_Tol: 1.0e-2"]
        self.shots = 1000
        #self.global_title = f"H2_dynamics/STO-3G/PNOF4/{self.shots}_shots/AER/IBM_pittsburgh/Opt_lvel=3"
        #self.global_title = f"H2_dynamics/STO-3G/PNOF4/{self.shots}_shots"
        self.global_title = f"H2_dynamics/STO-3G/PNOF4/Opt_circuits"
        self.titles = ["adam","sgd","slsqp","l-bfgs-b","spsa"]
        self.col = 3

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

    def plot_pos_vel_3db(self, *outputs):
        """
        Plot position vs velocity for 1–4 (or more) database outputs.
        Example: self.plot_pos_vel_3db(output_1, output_2, output_3)
        """
        fig, ax = plt.subplots()

        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))

            # Use modular indexing for colors/titles/markers if needed
            color = self.colors[i % len(self.colors)]
            title = self.titles[i % len(self.titles)]

            ax.plot(crd, vel / 1e-3, color=color, linestyle='--', label=title)
            ax.scatter(crd[0], vel[0] / 1e-3, color='r', marker=self.markers[(2*i) % len(self.markers)], s=40)
            ax.scatter(crd[-1], vel[-1] / 1e-3, color='g', marker=self.markers[(2*i + 1) % len(self.markers)], s=40)

        ax.set_xlabel('Position (a.u.)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Velocity (a.u.)', fontweight='bold', fontsize=16)
        ax.text(0.0, 1.0, "1e-3", transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
        plt.title(self.global_title, y=1.2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.195),
                prop={'size': 12}, ncol=self.col, frameon=False)

        fig.savefig(f"post_vel_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.close(fig)

    
    # def plot_pos_vel(self, output):
    #     db, _ = self.read_db(output)
    #     fig, ax = plt.subplots()
    #     crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
    #     vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))
    #     # # Convert velocity to momentum (multiply by H mass in a.u.)
    #     # m_H = 1837.15258739092  # hydrogen atom mass in a.u.
    #     # mom = vel * m_H
    #     ax.plot(crd, vel / 1e-3, color='blue')
    #     ax.scatter(crd[0], vel[0] / 1e-3, color='r', s=40, label='Initial point')
    #     ax.scatter(crd[-1], vel[-1] / 1e-3, color='g', s=40, label='Final point')

    #     ax.set_xlabel('Position (a.u.)', fontweight='bold', fontsize=16)
    #     ax.set_ylabel('Velocity (a.u.)', fontweight='bold', fontsize=16)

    #     # Add scaling factor note for velocity axis
    #     ax.text(
    #         0.0, 1.0, "1e-3",
    #         transform=ax.transAxes,  # relative to axes coords
    #         ha='left', va='bottom',
    #         fontsize=12
    #     )
    #     ax.legend()
    #     fig.savefig("post_vel_h2.pdf", bbox_inches='tight')
    #     plt.close(fig)

    def plot_time_total_energy_3db(self, *outputs):
        """
        Plot total time vs energy for 1–4 (or more) database outputs.
        Example: self.plot_time_total_energy_3db(output_1, output_2, output_3)
        """
        plt.figure()

        for i, output in enumerate(outputs):
            db, time = self.read_db(output)
            etot = np.array(list(db["etot"]), dtype=float)
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            plt.plot(
                time,
                (etot - etot[0]) / 1e-3,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label
            )

        plt.axhline(0, linestyle=':', c='black')
        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel(r'$\Delta$ Total Energy (mHa)', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        plt.title(self.global_title, y=1.2)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.19),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        plt.savefig(f"time_etotal_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.close()
        
    # def plot_time_total_energy(self, output):
    #     db, time = self.read_db(output) 
    #     etot = np.array(list(db["etot"]), dtype=float)
    #     plt.plot(time,(etot-etot[0])/1e-3,color='blue',label = '') 
    #     plt.axhline(0,linestyle=':', c = 'black')
    #     plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
    #     plt.ylabel(r'$\Delta$ Total Energy (mHa)', fontweight = 'bold', fontsize = 16)
    #     plt.xlim([0, 10])
    #     plt.savefig("time_etotal_h2.pdf", bbox_inches='tight')
    #     plt.close()

    def plot_time_parameter_3db(self, *outputs):
        """
        Plot the evolution of 'parameter' vs Time for 1–4 (or more) database outputs.
        Example:
            self.plot_time_parameter_3db(output_1, output_2, output_3)
        """
        plt.figure()

        for i, output in enumerate(outputs):
            db, time = self.read_db(output)
            parameter = np.array(list(db["parameter"]), dtype=float)

            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            plt.plot(
                time,
                parameter,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label
            )

        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel(r'Optimal $\boldsymbol{\theta}$', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.19),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        plt.title(self.global_title, y=1.2)
        plt.savefig(f"time_parameter_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.close()


        
    # def plot_time_parameter(self, output):
    #     db, time = self.read_db(output)
    #     parameter = np.array(list(db["parameter"]), dtype=float)
    #     plt.plot(time,parameter,color='blue',label = '') 
    #     plt.xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
    #     plt.ylabel(r'Optimal $\boldsymbol{\theta}$', fontweight = 'bold', fontsize=16)
    #     plt.xlim([0, 10])
    #     plt.savefig("time_parameter_h2.pdf", bbox_inches='tight')
    #     plt.close()

if __name__ == "__main__":
    # Collect all database arguments after the script name
    db_files = sys.argv[1:]  # e.g. python script.py db1.db db2.db db3.db

    if not db_files:
        print("Usage: python script.py <db1> <db2> [<db3> ...]")
        sys.exit(1)

    picture = PlotsH2()

    # Call all plotting functions with variable number of arguments
    picture.plot_pos_vel_3db(*db_files)
    picture.plot_time_total_energy_3db(*db_files)
    picture.plot_time_parameter_3db(*db_files)

    
