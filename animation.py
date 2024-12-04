#import sys
import shutil
import subprocess
import numpy as np
from numpy import copy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jinja2 import Template #build templates
from pysurf.database import PySurfDB
from pysurf.system import Molecule
from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
from moviepy.editor import VideoFileClip, clips_array

tpl = Template("""{{natoms}}
{% for atomid, crd in mol %} 
{{mol.format(atomid, crd)}} {% endfor %}

""")

class PlotDB:

    tpl = tpl

    def __init__(self, output, plot_e):
        self.plot_e = plot_e 
        self.output = PySurfDB.load_database(output, read_only=True)
        self.nstates = self.output.dimensions["nstates"]
        self.fs = 0.02418884254
        self.ev = 27.211324570273
        self.aa = 0.529177208
        self.ene_min = np.array(self.output["energy"]).min()
        self.ene_max = np.array(self.output["energy"]).max()
        self.times = self.output["time"][0:]*self.fs
        self.dim = len(self.times)
        if self.plot_e == "ene":
            plt.rcParams['font.size'] = '7' 
            self.fig, self.ax = plt.subplots(figsize=(3.5, 2.57))
            self.transition_point, = self.ax.plot([], [], 'o', color='black', markersize=5, zorder=5)
        elif self.plot_e == "ene_ang":
            plt.rcParams['font.size'] = '18' 
            self.fig = plt.figure(figsize=(7, 10))
            gs = self.fig.add_gridspec(2, 1, hspace=0)
            self.ax0 = self.fig.add_subplot(gs[0])
            self.ax1 = self.fig.add_subplot(gs[1])
            self.transition_point, = self.ax0.plot([], [], 'o', color='black', markersize=10, zorder=5)
        elif self.plot_e == "ene_ang_ene_diff":
            plt.rcParams['font.size'] = '18' 
            self.fig = plt.figure(figsize=(7, 12))
            gs = self.fig.add_gridspec(3, 1, hspace=0)
            self.ax0 = self.fig.add_subplot(gs[0])
            self.ax1 = self.fig.add_subplot(gs[1])
            self.ax2 = self.fig.add_subplot(gs[2])
            self.transition_point, = self.ax0.plot([], [], 'o', color='black', markersize=10, zorder=5)
        else:
            raise SystemExit("This option does not exit")

    def dihedral(self, a, b, c, d):
        crd = self.output["crd"]
        torsion = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)])
            vec_b = np.array(m[int(b)])
            vec_c = np.array(m[int(c)])
            vec_d = np.array(m[int(d)])
            torsion.append(calc_dihedral(Vector(vec_a),Vector(vec_b),Vector(vec_c),Vector(vec_d))* 180 / np.pi)
        return torsion

    def angle(self, a, b, c):
        crd = self.output["crd"]
        angle = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)])
            vec_b = np.array(m[int(b)])
            vec_c = np.array(m[int(c)])
            angle.append(calc_angle(Vector(vec_a),Vector(vec_b),Vector(vec_c))* 180 / np.pi)
        return angle 

    def pyramidalization_angle(self, a, b, c, o):
        crd = self.output["crd"]
        angle = []
        for i,m in enumerate(crd):
            vec_a = np.array(m[int(a)]) - np.array(m[int(o)])
            vec_b = np.array(m[int(b)]) - np.array(m[int(o)])
            vec_c = np.array(m[int(c)]) - np.array(m[int(o)])
            vec_u = np.cross(vec_a, vec_b)
            d_cu = np.dot(vec_c,vec_u)
            cr_cu = np.cross(vec_c, vec_u)
            n_cr_cu = np.linalg.norm(cr_cu)
            res = np.math.atan2(n_cr_cu,d_cu)
            angle.append(90 - np.degrees(res))
        return angle 

    def high_180(self, y):
        y_nan = y.copy()
        for i in range(1,int(len(y))):
            #if (y[i]-y[i+1])>0 and y[i]>179 and y[i+1]<0:
            #    y_nan[i] = y[i]*np.nan
            #elif (y[i]-y[i+1])<0 and y[i]<-179 and y[i+1]>0:
            #    y_nan[i] = y[i]*np.nan
            if y[i-1]>0 and y[i]<0: 
                y_nan[i] = y[i]*np.nan
            elif y[i-1]<0 and y[i]>0:
                y_nan[i] = y[i]*np.nan
        return y_nan

    def write_traj_movie_xyz(self):
        db_file = "sampling.db"
        db = PySurfDB.load_database(db_file, read_only=True)
        atomids = copy(db['atomids'])
        crd = self.output["crd"]
        natoms = len(atomids)
        molecule = Molecule(atomids, None) 
        filename = 'traj.xyz'
        traj_xyz = open(filename, 'w')
        for i in range(self.dim):
            molecule.crd = np.array(crd[i])*self.aa  # coordinates must be in Angstrom units 
            traj_xyz.write(self.tpl.render(natoms=natoms,mol=molecule))
        traj_xyz.close()

    def make_movie_mpg_mp4(self):
        output_mpg = "traj.mpg"  
        output_mp4 = "traj.mp4" 
        #Making the traj.xyz file
        self.write_traj_movie_xyz()
        # VMD script with the desired visualization and movie settings
        script_file = "/Users/edisonsalazar/Desktop/Postdoct_Position/Xmaris/plot_trajectory_movies/traj.tcl"
        # Run VMD with the generated script
        try:
            vmd_path = "/Applications/VMD 1.9.4a57-x86_64-Rev12.app/Contents/Resources/VMD.app/Contents/MacOS/VMD"
            # Generate frames using VMD
            subprocess.run([vmd_path, "-e", script_file], check=True)
            print(f"Frames generated successfully in './frames/'")
            # Combine frames into an MPG movie using FFmpeg
            subprocess.run([
                "ffmpeg", "-framerate", "30", "-i", "./frames/frame_%d.tga",
                "-c:v", "mpeg2video", "-q:v", "2", output_mpg
            ], check=True)
            print(f"Movie generated successfully in MPG format: {output_mpg}")
            # Convert the MPG movie to MP4 format using FFmpeg
            subprocess.run([
                "ffmpeg", "-i", output_mpg, "-c:v", "libx264", output_mp4
            ], check=True)
            print(f"Movie successfully converted to MP4 format: {output_mp4}")
        finally:
            # Clean up the script file
            shutil.rmtree("frames")

    def plot_energy_angles_ene_diff_vs_time(self):
        # Positions of atoms defined only for the CH2NH molecule 
        atom_1=0
        atom_2=1
        atom_3=2
        atom_4=3
        atom_5=4

        # the first subplot
        self.ax0.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 16)
        self.ax0.set_xlim(self.times.min(), self.times.max())
        self.ax0.set_ylim(- 1, (self.ene_max-self.ene_min)*self.ev + 1)
        self.ax0.set_facecolor('white')
        handles1 = []
        labels1 = []
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            line, = self.ax0.plot(self.times,(ene-self.ene_min)*self.ev)
            handles1.append(line)
            labels1.append('$S_%i$'%i)
        self.transition_point.set_data([], [])
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line0 = self.ax0.axvline(x[0],linestyle='--', c = 'purple')
                handles1.append(line0)
                labels1.append(fr"Hop:($S_{int(curr[0])} \to S_{int(new[0])}$) at {int(x[0])} fs")

        # the second subplot
        self.ax1.set_ylabel("Angle (degrees)", fontweight = 'bold', fontsize = 16)
        self.ax1.set_xlim(self.times.min(), self.times.max())
        dihedral = self.dihedral(atom_3, atom_1, atom_2, atom_5)
        dihedral_nan = self.high_180(dihedral)
        angle = self.angle(atom_1, atom_2, atom_5)
        pira = self.pyramidalization_angle(atom_3, atom_4, atom_2, atom_1)
        line1, = self.ax1.plot(self.times, dihedral_nan, color='blue')
        line2, = self.ax1.plot(self.times, angle, color='darkgreen',linestyle='--')
        line3, = self.ax1.plot(self.times, pira, color='black',linestyle='-.')
        handles2 = [line1,line2,line3]
        labels2 = ['HCNH','CNH', 'pyram.']
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line1 = self.ax1.axvline(x[0],linestyle='--', c = 'purple')
        plt.setp(self.ax0.get_xticklabels(), visible=False)

        # the third subplot

        self.ax2.set_xlabel('Time (fs)', fontweight = 'bold', fontsize = 16)
        self.ax2.set_ylabel('$\mathbf{\Delta(E_{t_i}-E_{t_0})(eV)}$', fontsize = 16)
        self.ax2.set_xlim(self.times.min(), self.times.max())
        etot = np.array(self.output["etot"])
        epot = np.array(self.output["epot"])
        ekin = np.array(self.output["ekin"])
        line1_3, = self.ax2.plot(self.times, (etot-etot[0])*self.ev, color='orange')
        line2_3, = self.ax2.plot(self.times, (epot-epot[0])*self.ev, color='purple',linestyle='-.')
        line3_3, = self.ax2.plot(self.times, (ekin-ekin[0])*self.ev, color='black',linestyle='--')
        handles3 = [line1_3,line2_3,line3_3]
        labels3 = ['$\Delta E_{tot}$','$\Delta E_{pot}$', '$\Delta E_{kin}$']
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line1_3 = self.ax2.axvline(x[0],linestyle='--', c = 'purple')
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top padding as needed

        self.ax0.legend(handles1,labels1,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 14}, ncol=len(labels1))
        self.ax1.legend(handles2,labels2,loc='upper center', bbox_to_anchor=(0.5, 2.35), prop={'size': 14}, ncol=len(labels2))
        self.ax2.legend(handles3,labels3,loc='lower center', bbox_to_anchor=(0.5, -0.4), prop={'size': 14}, ncol=len(labels3))
        return self.transition_point,

    def plot_energy_angles_vs_time(self):
        # Positions of atoms defined only for the CH2NH molecule 
        atom_1=0
        atom_2=1
        atom_3=2
        atom_4=3
        atom_5=4

        # the first subplot
        self.ax0.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 16)
        self.ax0.set_xlim(self.times.min(), self.times.max())
        self.ax0.set_ylim(- 1, (self.ene_max-self.ene_min)*self.ev + 1)
        self.ax0.set_facecolor('white')
        handles1 = []
        labels1 = []
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            line, = self.ax0.plot(self.times,(ene-self.ene_min)*self.ev)
            handles1.append(line)
            labels1.append('$S_%i$'%i)
        self.transition_point.set_data([], [])
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line0 = self.ax0.axvline(x[0],linestyle='--', c = 'purple')
                handles1.append(line0)
                labels1.append(fr"Hop:($S_{int(curr[0])} \to S_{int(new[0])}$) at {int(x[0])} fs")

        # the second subplot
        self.ax1.set_ylabel("Angle (degrees)", fontweight = 'bold', fontsize = 16)
        self.ax1.set_xlabel("Time (fs)", fontweight = 'bold', fontsize = 16)
        self.ax1.set_xlim(self.times.min(), self.times.max())
        dihedral = self.dihedral(atom_3, atom_1, atom_2, atom_5)
        dihedral_nan = self.high_180(dihedral)
        angle = self.angle(atom_1, atom_2, atom_5)
        pira = self.pyramidalization_angle(atom_3, atom_4, atom_2, atom_1)
        line1, = self.ax1.plot(self.times, dihedral_nan, color='blue')
        line2, = self.ax1.plot(self.times, angle, color='darkgreen',linestyle='--')
        line3, = self.ax1.plot(self.times, pira, color='black',linestyle='-.')
        handles2 = [line1,line2,line3]
        labels2 = ['HCNH','CNH', 'pyram.']
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line1 = self.ax1.axvline(x[0],linestyle='--', c = 'purple')
        plt.setp(self.ax0.get_xticklabels(), visible=False)
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top padding as needed

        self.ax0.legend(handles1,labels1,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 16}, ncol=len(labels1))
        self.ax1.legend(handles2,labels2,loc='upper center', bbox_to_anchor=(0.5, 2.27), prop={'size': 16}, ncol=len(labels2))
        return self.transition_point,

    def plot_energy_vs_time(self):
        self.ax.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 8)
        self.ax.set_xlabel("Time (fs)", fontweight = 'bold', fontsize = 8)
        self.ax.set_xlim(self.times.min(), self.times.max())
        self.ax.set_ylim(- 1, (self.ene_max-self.ene_min)*self.ev + 1)
        self.ax.set_facecolor('white')
        handles = []
        labels = []
        for i in range(self.nstates):
            ene = np.array(self.output["energy"])[:,i] 
            line, = self.ax.plot(self.times,(ene-self.ene_min)*self.ev)
            handles.append(line)
            labels.append('$S_%i$'%i)
        self.transition_point.set_data([], [])
        for i in range(1, self.dim):
            if self.output["currstate"][i] != self.output["currstate"][i-1]:
                curr = self.output["currstate"][i-1]
                new = self.output["currstate"][i]
                x = self.times[i]
                line0 = self.ax.axvline(x[0],linestyle='--', c = 'purple')
                handles.append(line0)
                labels.append(fr"Hop:($S_{int(curr[0])} \to S_{int(new[0])}$) at {int(x[0])} fs")
        plt.tight_layout()
        #plt.subplots_adjust(top=0.9)  # Adjust the top padding as needed
        self.ax.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 8}, ncol=len(labels))
        return self.transition_point,

    def update_plot(self, i):
        ene_i = int(self.output["currstate"][i])  
        ene_curr = (np.array(self.output["energy"])[i,ene_i] - self.ene_min)*self.ev
        self.transition_point.set_data(self.times[i], ene_curr)
        return self.transition_point,

    def create_ene_ang_animation(self):
        if self.plot_e == "ene":
            ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.plot_energy_vs_time, frames=len(self.times), blit=True)
            energy_animation_file = 'ene_animation.mp4'
        elif self.plot_e == "ene_ang":
            ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.plot_energy_angles_vs_time, frames=len(self.times), blit=True)
            energy_animation_file = 'ene_ang_animation.mp4'
        elif self.plot_e == "ene_ang_ene_diff":
            ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.plot_energy_angles_ene_diff_vs_time, frames=len(self.times), blit=True)
            energy_animation_file = 'ene_ang_animation.mp4'
        else:   
            raise SystemExit("This option does not exit")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(energy_animation_file, writer=writer)
        plt.close()
        return energy_animation_file

    def create_animation_combined(self):
        energy_animation_file = self.create_ene_ang_animation()

        #create the movie in mp4 format saved as traj.mp4
        self.make_movie_mpg_mp4() 
        # Load the trajectory video
        trajectory_clip = VideoFileClip("traj.mp4")

        # Load the energy animation video
        energy_clip = VideoFileClip(energy_animation_file)

        # Combine the two videos side by side
        final_clip = clips_array([[trajectory_clip, energy_clip]])

        # Save the final combined video
        if self.plot_e == "ene":
            combined_animation_file = 'ene_combined_video.mp4'
        elif self.plot_e == "ene_ang":
            combined_animation_file = 'ene_combined_video.mp4'
        elif self.plot_e == "ene_ang_ene_diff":
            combined_animation_file = 'ene_combined_video.mp4'
        else:   
            raise SystemExit("This option does not exit")
        final_clip.write_videofile(combined_animation_file, fps=trajectory_clip.fps)

if __name__ == '__main__':
    # Database. In PySurf it is called results.db
    #db = sys.argv[1] 
    db = "results.db" 
    # Type of movie: energies (ene) or energies and angles (ene_ang)
    #type_movie = sys.argv[2]
    #type_movie = "ene_ang"
    type_movie = "ene_ang_ene_diff"
    # Calling the class PlotDB
    plot_db = PlotDB(db,type_movie)
    # Function to create an animation 
    plot_db.create_animation_combined()

