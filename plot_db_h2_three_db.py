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
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:10]
        #self.colors = [self.colors[0], self.colors[3], self.colors[4]]
        self.markers = list(Line2D.filled_markers)
        #self.titles = ["Noisless","Noise/Conv_Tol: 1.0e-2","Noise/Conv_Tol: 1.0e-3", "Noise/Conv_Tol: 1.0e-4"]
        # self.titles = ["Noisless","Noise/Conv_Tol: 1.0e-2 (SGD/1000)","Real/Conv_Tol: 1.0e-2 (SDG/1000)", "Real/Conv_Tol: 1.0e-3 (ADAM/10000)"]
        #self.titles = ["Simulator/Conv_Tol: 1.0e-7","Real/Conv_Tol:1.0e-2/Res_Lev:0(SGD/1000)","Real/Conv_Tol:1.0e-3/Res_Lev:0(ADAM/10000)","Hybrid_Ene_Param/Conv_Tol:1.0e-3/Res_Lev:0(SLSQP/10000)","Hybrid_Ene/Conv_Tol:1.0e-7/Res_Lev:0(SLSQP/10000)","Hybrid_Ene/Conv_Tol:1.0e-7/Res_Lev:2(SLSQP/10000)"]
        #self.titles = ["Simulator/Conv_Tol: 1.0e-7","Hybrid_Ene/Conv_Tol: 1.0e-7 (SLSQP/10000)"]
        #self.titles = [r"$X_0:\ 2\ Bohr$",r"$X_0:\ 1\ Bohr$",r"$X_0:\ 1.13\ Bohr$", r"$X_0:\ 1.95\ Bohr$"]
        self.titles = [r"NOFVQE; $X_0:\ 0.6\ \AA$ (1.13 Bohr)", r"VQE; $X_0:\ 0.6\ \AA$ (1.13 Bohr)"]
        #self.titles = ["Simulator/Conv_Tol: 1.0e-7","Hybrid_Ene/Conv_Tol:1.0e-7/Res_Lev:0(SLSQP/10000)","Hybrid_Ene/Conv_Tol:1.0e-7/Res_Lev:2(SLSQP/10000)"]
        self.shots = 10000
        #self.global_title = f"H2_dynamics/STO-3G/PNOF4/{self.shots}_shots/AER/IBM_pittsburgh/Opt_lvel=3"
        #self.global_title = f"H2_dynamics/STO-3G/PNOF4/{self.shots}_shots"
        self.global_title = f"H2_dynamics/STO-3G/PNOF/Opt_Level_3"
        #self.titles = ["adam","sgd","slsqp","l-bfgs-b","spsa","cobyla","cmaes"]
        self.col = 2
        self.y = 1.3
        self.bohr_to_ang = 0.52917721092

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
    
    def dis_two_atmos_grad(self, data, atom_1, atom_2):
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        dimer = []
        for i,m in enumerate(data):
            dimer.append(float(m[0][atom_2][2])-float(m[0][atom_1][2]))
        return dimer
    
    def rdm1_from_triangle(self, tri_vec, norb):
        """
        Convert a flattened upper-triangular RDM (including diagonal)
        into a full symmetric norb×norb matrix.
        """
        rdm = np.zeros((norb, norb))
        idx = 0
        for i in range(norb):
            for j in range(i, norb):
                rdm[i, j] = tri_vec[idx]
                rdm[j, i] = tri_vec[idx]  # symmetric
                idx += 1
        return rdm

    def plot_time_total_energy(self, *outputs):
        """
        Plot total time vs energy for 1–4 (or more) database outputs.
        Example: self.plot_time_total_energy(output_1, output_2, output_3)
        """
        plt.figure()

        for i, output in enumerate(outputs):
            db, time = self.read_db(output)
            etot = np.array(list(db["etot"]), dtype=float)
            rel_etot = (etot - etot[0]) / 1e-3
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]
            min_index = np.argmin(rel_etot)
            max_index = np.argmax(rel_etot)
            print(f"Min value of total energy for {output}: {rel_etot.min()} (mHa)")
            print(f"Max value of total energy for {output}: {rel_etot.max()} (mHa)")
            print(f"Time for the min of total energy for {output}: {time[min_index][0]} (fs) in md_step {min_index}")
            print(f"Time for the max of total energy for {output}: {time[max_index][0]} (fs) in md_step {max_index}")

            plt.plot(
                time,
                rel_etot,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw =2
            )

        plt.axhline(0, linestyle=':', c='black')
        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel(r'$\Delta$ Total Energy (mHa)', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        #plt.ylim([-6.4, 6.4])
        #plt.title(self.global_title, y=self.y)
        # plt.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, self.y),
        #     prop={'size': 12},
        #     ncol=self.col,
        #     frameon=False
        # )
        #plt.savefig(f"time_etotal_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.savefig(f"time_etotal_h2.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_time_parameter(self, *outputs):
        """
        Plot the evolution of 'parameter' vs Time for 1–4 (or more) database outputs.
        Example:
            self.plot_time_parameter(output_1, output_2, output_3)
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
                linestyle='-',
                #linestyle='--' if i > 0 else '-',
                label=label,
                lw =2
            )

        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel(r'Optimal $\boldsymbol{\theta}$', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, self.y),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        #plt.title(self.global_title, y=self.y)
        #plt.savefig(f"time_parameter_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.savefig(f"time_parameter_h2.pdf", bbox_inches='tight')
        plt.close()

    def plot_time_distance(self, *outputs):
        """
        Plot the evolution of 'crd' vs Time for 1–4 (or more) database outputs.
        Example:
            self.plot_time_distance(output_1, output_2, output_3)
        """
        plt.figure()

        for i, output in enumerate(outputs):
            db, time = self.read_db(output)
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            min_index = np.argmin(crd)
            print(f"Min value of position for {output}: {crd.min()} (a.u.)")
            print(f"Time for the min of position for {output}: {time[min_index][0]} (fs) in md_step {min_index}")
            print(f"Difference between initial position and after a cycle at same position: {crd[0]- crd[62]} (a.u.)")

            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            plt.plot(
                time,
                crd,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw =2
            )

        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel('Position (a.u.)', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        # plt.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, self.y),
        #     prop={'size': 12},
        #     ncol=self.col,
        #     frameon=False
        # )
        #plt.title(self.global_title, y=self.y)
        #plt.savefig(f"time_distance_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.savefig(f"time_distance_h2.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_time_gs_energy(self, *outputs):
        """
        Plot the evolution of 'gs_energy' vs Time for 1–4 (or more) database outputs.
        Example:
            self.plot_time_gs_energy(output_1, output_2, output_3)
        """
        plt.figure()

        for i, output in enumerate(outputs):
            db, time = self.read_db(output)
            ene_gs = np.array(list(db["energy"]), dtype=float)

            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            plt.plot(
                time,
                ene_gs,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw =2
            )

        plt.xlabel('Time (fs)', fontweight='bold', fontsize=16)
        plt.ylabel('GS Energy (Ha)', fontweight='bold', fontsize=16)
        plt.xlim([0, 10])
        # plt.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, self.y),
        #     prop={'size': 12},
        #     ncol=self.col,
        #     frameon=False
        # )
        #plt.title(self.global_title, y=self.y)
        #plt.savefig(f"time_distance_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.savefig(f"time_gs_energy_h2.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_position_gs_energy(self, *outputs):
        """
        Plot the evolution of 'gs_energy' vs Time for 1–4 (or more) database outputs.
        Example:
            self.plot_time_gs_energy(output_1, output_2, output_3)
        """

        fig, ax1 = plt.subplots()

        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            ene_gs = np.array(list(db["energy"]), dtype=float)
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            ax1.plot(
                crd,
                ene_gs,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw =2
            )
            
        # Lower x-axis (Bohr / a.u.)
        ax1.set_xlabel("Position (a.u.)", fontweight="bold", fontsize=16)

        # Left y-axis
        ax1.set_ylabel("GS Energy (Ha)", fontweight="bold", fontsize=16)

        # Upper x-axis (Angstrom)
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ticks_bohr = ax1.get_xticks()
        ax2.set_xticks(ticks_bohr)
        ax2.set_xticklabels(np.round(ticks_bohr * self.bohr_to_ang, 2))
        ax2.set_xlabel("Position (Å)", fontweight="bold", fontsize=16)
        #plt.ylim([-1.14, -1.11])
        # plt.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, self.y),
        #     prop={'size': 12},
        #     ncol=self.col,
        #     frameon=False
        # )
        #plt.title(self.global_title, y=self.y)
        #plt.savefig(f"time_distance_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        plt.savefig(f"distance_gs_energy_h2.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_position_force_energy(self, *outputs):
        """
        Plot the evolution of 'position' vs forces for 1–4 (or more) database outputs.
        Example:
            self.plot_position_forces(output_1, output_2, output_3)
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        plt.subplots_adjust(hspace=0.0)   # no space between subplots
        pot_min_1 = +np.inf
        pot_max_1 = -np.inf
        pot_min_2 = +np.inf
        pot_max_2 = -np.inf
        
        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            
            force = -np.array(self.dis_two_atmos_grad(db["gradient"], 0, 1))
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))
            ene_gs = np.array(list(db["energy"]), dtype=float)
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]
            
            vel = vel/1e-3
            axs[0].plot(crd, vel, color=color, linestyle='--', label=label, lw =2)
            axs[0].scatter(crd[0], vel[0], color=color, marker='o', s=40)
            axs[0].scatter(crd[-1], vel[-1], color=color, marker='s', s=40)
            axs[1].plot(crd, ene_gs, color=color, linestyle='--', lw =2)
            axs[1].scatter(crd[0], ene_gs[0], color=color, marker='o', s=40)
            axs[1].scatter(crd[-1], ene_gs[-1], color=color, marker='s', s=40)
            axs[2].plot(crd, force, color=color, linestyle='--', lw =2)
            axs[2].scatter(crd[0], force[0], color=color, marker='o', s=40)
            axs[2].scatter(crd[-1], force[-1], color=color, marker='s', s=40)
            # update global min/max
            pot_min_1 = min(pot_min_1, ene_gs.min())
            pot_max_1 = max(pot_max_1, ene_gs.max())
            pot_min_2 = min(pot_min_2, force.min())
            pot_max_2 = max(pot_max_2, force.max())
            
        #axs[1].set_ylim(-1.146,-1.075)
        pad_1 = 0.1 * (pot_max_1 - pot_min_1)
        axs[1].set_ylim(pot_min_1 - pad_1, pot_max_1 + pad_1)
        pad_2 = 0.1 * (pot_max_2 - pot_min_2)
        axs[2].set_ylim(pot_min_2 - pad_2, pot_max_2 + pad_2)
        axs[0].text(0.0, 1.0, "1e-3", transform=axs[0].transAxes, ha='left', va='bottom', fontsize=12)
        axs[0].set_ylabel("Velocity (a.u.)", fontweight="bold", fontsize=16)
        axs[1].set_ylabel(r"$\bf{E_{pot}}$ (Ha)", fontweight="bold", fontsize=16)
        axs[2].set_ylabel("Force (Ha/Bohr)", fontweight="bold", fontsize=16)
        axs[2].set_xlabel("Position (a.u.)", fontweight="bold", fontsize=16)

        # Left y-axis (Ha/Bohr)
        
        axs[0].legend(
            loc='upper center',
            bbox_to_anchor=(0.5, self.y),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        #plt.title(self.global_title, y=self.y)
        plt.savefig(f"distance_force_energy_h2_au.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_position_force_energy_ang(self, *outputs):
        """
        Plot the evolution of 'position' vs forces for 1–4 (or more) database outputs.
        Example:
            self.plot_position_forces(output_1, output_2, output_3)
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        plt.subplots_adjust(hspace=0.0)   # no space between subplots
        pot_min_1 = +np.inf
        pot_max_1 = -np.inf
        pot_min_2 = +np.inf
        pot_max_2 = -np.inf
        
        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            
            force = -np.array(self.dis_two_atmos_grad(db["gradient"], 0, 1))
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))
            ene_gs = np.array(list(db["energy"]), dtype=float)
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]
            
            vel = vel/1e-3
            crd = crd* self.bohr_to_ang
            force = force* self.bohr_to_ang
            axs[0].plot(crd, vel, color=color, linestyle='--', label=label, lw =2)
            axs[0].scatter(crd[0], vel[0], color=color, marker='o', s=40)
            axs[0].scatter(crd[-1], vel[-1], color=color, marker='s', s=40)
            axs[1].plot(crd, ene_gs, color=color, linestyle='--', lw =2)
            axs[1].scatter(crd[0], ene_gs[0], color=color, marker='o', s=40)
            axs[1].scatter(crd[-1], ene_gs[-1], color=color, marker='s', s=40)
            axs[2].plot(crd, force, color=color, linestyle='--', lw =2)
            axs[2].scatter(crd[0], force[0], color=color, marker='o', s=40)
            axs[2].scatter(crd[-1], force[-1], color=color, marker='s', s=40)
            # update global min/max
            pot_min_1 = min(pot_min_1, ene_gs.min())
            pot_max_1 = max(pot_max_1, ene_gs.max())
            pot_min_2 = min(pot_min_2, force.min())
            pot_max_2 = max(pot_max_2, force.max())
            
        #axs[1].set_ylim(-1.146,-1.075)
        pad_1 = 0.1 * (pot_max_1 - pot_min_1)
        axs[1].set_ylim(pot_min_1 - pad_1, pot_max_1 + pad_1)
        pad_2 = 0.1 * (pot_max_2 - pot_min_2)
        axs[2].set_ylim(pot_min_2 - pad_2, pot_max_2 + pad_2)
        axs[0].text(0.0, 1.0, "1e-3", transform=axs[0].transAxes, ha='left', va='bottom', fontsize=12)
        axs[0].set_ylabel("Velocity (a.u.)", fontweight="bold", fontsize=16)
        axs[1].set_ylabel(r"$\bf{E_{pot}}$ (Ha)", fontweight="bold", fontsize=16)
        axs[2].set_ylabel("Force (Ha/Å)", fontweight="bold", fontsize=16)
        axs[2].set_xlabel("Position (Å)", fontweight="bold", fontsize=16)

        # Left y-axis (Ha/Bohr)
        
        axs[0].legend(
            loc='upper center',
            bbox_to_anchor=(0.5, self.y),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        #plt.title(self.global_title, y=self.y)
        plt.savefig(f"distance_force_energy_h2_ang.pdf", bbox_inches='tight')
        plt.close()
        
    def plot_pos_vel(self, *outputs):
        """
        Plot position vs velocity for 1–4 (or more) database outputs.
        Example: self.plot_pos_vel(output_1, output_2, output_3)
        """
        fig, ax = plt.subplots()

        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            vel = np.array(self.dis_two_atmos(db["veloc"], 0, 1))

            # Use modular indexing for colors/titles/markers if needed
            color = self.colors[i % len(self.colors)]
            title = self.titles[i % len(self.titles)]
            
            crd = crd
            vel = vel/1e-3
            
            ax.plot(crd, vel, color=color, linestyle='--', label=title, lw =2)
            ax.scatter(crd[0], vel[0], color='black', marker='o', s=40)
            ax.scatter(crd[-1], vel[-1], color='black', marker='s', s=40)

        ax.set_xlabel('Position (a.u.)', fontweight='bold', fontsize=16)
        ax.set_ylabel('Velocity (a.u.)', fontweight='bold', fontsize=16)
        ax.text(0.0, 1.0, "1e-3", transform=ax.transAxes, ha='left', va='bottom', fontsize=12)
        #plt.title(self.global_title, y=self.y)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, self.y),
                 prop={'size': 12}, ncol=self.col, frameon=False)

        #fig.savefig(f"post_vel_h2_3_{self.shots}_shots.pdf", bbox_inches='tight')
        fig.savefig(f"post_vel_h2_au.pdf", bbox_inches='tight')
        plt.close(fig)
        
    def plot_position_gs_energy_ang(self, *outputs):

        fig, ax1 = plt.subplots()

        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            ene_gs = np.array(list(db["energy"]), dtype=float)
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            # Only keep the region matching the paper
            crd = crd* self.bohr_to_ang

            ax1.plot(
                crd,
                ene_gs,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw=2
            )

        # Lower x-axis (Bohr)
        ax1.set_xlabel("Position (Å)", fontweight="bold", fontsize=16)
        ax1.set_ylabel("GS Energy (Ha)", fontweight="bold", fontsize=16)

        plt.savefig("distance_gs_energy_h2_ang.pdf", bbox_inches="tight")
        plt.close()

    def plot_position_force_ang(self, *outputs):

        fig, ax1 = plt.subplots()

        for i, output in enumerate(outputs):
            db, _ = self.read_db(output)
            force = -np.array(self.dis_two_atmos_grad(db["gradient"], 0, 1))
            crd = np.array(self.dis_two_atmos(db["crd"], 0, 1))
            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            crd = crd* self.bohr_to_ang
            force = force* self.bohr_to_ang

            ax1.plot(
                crd,
                force,
                color=color,
                linestyle='--' if i > 0 else '-',
                label=label,
                lw=2
            )

        # Lower x-axis (Å)
        ax1.set_xlabel("Position (Å)", fontweight="bold", fontsize=16)

        # Left y-axis (Ha/Å)
        ax1.set_ylabel("Force (Ha/Å)", fontweight="bold", fontsize=16)

        plt.savefig("distance_force_h2_ang.pdf", bbox_inches="tight")
        plt.close()

            
    def plot_time_relative_energies(self, *outputs):
        """
        Plot relative Total, Potential, and Kinetic energies in three stacked
        subplots with no vertical spacing. Only the bottom plot has axis labels.
        """

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        plt.subplots_adjust(hspace=0.0)   # no space between subplots
        # subplot 2 global range tracking
        pot_min_1 = +np.inf
        pot_max_1 = -np.inf
        pot_min_2 = +np.inf
        pot_max_2 = -np.inf
        for i, output in enumerate(outputs):
            db, time = self.read_db(output)

            # ----- UPDATE THIS PART TO MATCH YOUR DB FORMAT -----
            e_tot = np.array(list(db["etot"]), dtype=float)
            e_pot = np.array(list(db["epot"]), dtype=float)
            e_kin = np.array(list(db["ekin"]), dtype=float)
            # -----------------------------------------------------

            # relative values
            rel_tot = (e_tot - e_tot[0])/ 1e-3
            rel_pot = e_pot #- e_pot[0]
            rel_kin = e_kin #- e_kin[0]

            color = self.colors[i % len(self.colors)]
            label = self.titles[i % len(self.titles)]

            # Plot each on its subplot
            axs[0].plot(time, rel_tot, color=color, linestyle='-', lw=2, label=label)
            axs[1].plot(time, rel_pot, color=color, linestyle='-', lw=2)
            axs[2].plot(time, rel_kin, color=color, linestyle='-', lw=2)
            # update global min/max
            pot_min_1 = min(pot_min_1, rel_pot.min())
            pot_max_1 = max(pot_max_1, rel_pot.max())
            pot_min_2 = min(pot_min_2, rel_kin.min())
            pot_max_2 = max(pot_max_2, rel_kin.max())
        # Titles on left side
        # axs[0].set_ylabel("ΔE_tot (mHa)", fontsize=14, fontweight='bold')
        # axs[0].set_ylim(-6,6)
        # axs[1].set_ylabel("ΔE_pot (Ha)", fontsize=14, fontweight='bold')
        # axs[2].set_ylabel("ΔE_kin (Ha)", fontsize=14, fontweight='bold')
        # reference lines on chemical accuracy: 1.6x10e-3 Ha (1kcal/mol)
        chem_low  = -1.6
        chem_high =  1.6
        axs[0].hlines(y=chem_low, xmin=time[0], xmax=time[-1], linestyle='--', linewidth=1.5, color='black')
        axs[0].hlines(y=chem_high, xmin=time[0], xmax=time[-1], linestyle='--', linewidth=1.5, color='black')
        axs[0].set_ylabel(r"$\boldsymbol{\Delta} \bf{E_{tot}}$ (mHa)", fontsize=14, fontweight='bold')
        axs[0].set_ylim(-2.6,2.6)
        axs[1].set_ylabel(r"$\bf{E_{pot}}$ (Ha)", fontsize=14, fontweight='bold')
        pad_1 = 0.1 * (pot_max_1 - pot_min_1)
        axs[1].set_ylim(pot_min_1 - pad_1, pot_max_1 + pad_1)
        axs[2].set_ylabel(r"$\bf{E_{kin}}$ (Ha)", fontsize=14, fontweight='bold')
        pad_2 = 0.1 * (pot_max_2 - pot_min_2)
        axs[2].set_ylim(pot_min_2 - pad_2, pot_max_2 + pad_2)
        # Only bottom has X-label
        axs[2].set_xlabel("Time (fs)", fontsize=16, fontweight='bold')
        # Legend only on the top subplot
        #axs[0].legend(frameon=False, fontsize=12)
        axs[0].legend(
            loc='upper center',
            bbox_to_anchor=(0.5, self.y),
            prop={'size': 12},
            ncol=self.col,
            frameon=False
        )
        fig.savefig("relative_energies.pdf", bbox_inches='tight')
        plt.close()

    def plot_avg_rdm1(self, *outputs):
        """ 
        First output is taken as the reference.
        Remaining outputs are QC results.
        Usage:
            self.plot_avg_rdm1(ref_db, qc_db1, qc_db2, ...)
        """

        # -------------------
        # Split inputs
        # -------------------
        reference_output = outputs[0]
        qc_outputs = outputs[1:]

        # --- Read reference ---
        db_ref, _ = self.read_db(reference_output)
        rdm_ref_list = [np.array(r) for r in db_ref["rdm1_opt"]]

        # Determine number of orbitals
        tri_len = len(rdm_ref_list[0])
        norb = int((np.sqrt(8*tri_len + 1) - 1) / 2)  # invert n(n+1)/2

        # Build reference full matrices
        rdm_ref_full = np.array([
            self.rdm1_from_triangle(r, norb) for r in rdm_ref_list
        ])
        avg_ref = rdm_ref_full.mean(axis=0)

        # ---- Plot reference ----
        plt.figure()
        plt.imshow(avg_ref)
        plt.colorbar()
        plt.xticks(range(norb))
        plt.yticks(range(norb))
        plt.title("Averaged 1-RDM (Reference)")
        plt.savefig("avg_rdm1_reference.pdf", bbox_inches='tight')
        plt.close()


        # -------- Process QC outputs ---------
        for i, output in enumerate(qc_outputs):
            db_qc, _ = self.read_db(output)
            rdm_qc_list = [np.array(r) for r in db_qc["rdm1_opt"]]

            rdm_qc_full = np.array([
                self.rdm1_from_triangle(r, norb) for r in rdm_qc_list
            ])
            avg_qc = rdm_qc_full.mean(axis=0)

            label = self.titles[i % len(self.titles)]

            # --- Plot QC RDM ---
            plt.figure()
            plt.imshow(avg_qc)
            plt.colorbar()
            plt.xticks(range(norb))
            plt.yticks(range(norb))
            plt.title(f"Averaged 1-RDM ({label})")
            plt.savefig(f"avg_rdm1_{label}.pdf", bbox_inches='tight')
            plt.close()

            # --- Plot QC - reference difference ---
            diff = avg_qc - avg_ref

            plt.figure()
            plt.imshow(diff)
            plt.colorbar()
            plt.xticks(range(norb))
            plt.yticks(range(norb))
            plt.title(f"RDM1 Difference: {label} - Reference")
            plt.savefig(f"rdm1_diff_{label}_vs_reference.pdf", bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Collect all database arguments after the script name
    db_files = sys.argv[1:]  # e.g. python script.py db1.db db2.db db3.db

    if not db_files:
        print("Usage: python script.py <db1> <db2> [<db3> ...]")
        sys.exit(1)

    picture = PlotsH2()

    # Call all plotting functions with variable number of arguments
    # picture.plot_time_total_energy(*db_files)
    # picture.plot_time_distance(*db_files)
    # picture.plot_time_gs_energy(*db_files)
    # picture.plot_pos_vel(*db_files)
    picture.plot_position_force_energy(*db_files)
    # picture.plot_position_gs_energy_ang(*db_files)
    # picture.plot_position_force_ang(*db_files)
    picture.plot_time_relative_energies(*db_files)
    picture.plot_position_force_energy_ang(*db_files)
    #picture.plot_time_parameter(*db_files)


    # NEW function
    #picture.plot_avg_rdm1(*db_files)
    
