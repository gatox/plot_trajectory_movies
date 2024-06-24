import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pysurf.database import PySurfDB
from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
from pysurf.system import Molecule
from moviepy.editor import VideoFileClip, clips_array

class PlotDB:

    def __init__(self, output):
        self.output = PySurfDB.load_database(output, read_only=True)
        self.nstates = self.output.dimensions["nstates"]
        self.fs = 0.02418884254
        self.ev = 27.211324570273
        self.aa = 0.529177208
        self.ene_min = np.array(self.output["energy"]).min()
        self.times = self.output["time"][0:]*self.fs
        self.dim = len(self.times)
        self.fs_rcParams = '18'
        self.S0_energies =  (np.array(self.output["energy"])[:,0] - self.ene_min)*self.ev
        self.S1_energies =  (np.array(self.output["energy"])[:,1] - self.ene_min)*self.ev

        plt.rcParams['font.size'] = self.fs_rcParams
        self.fig = plt.figure(figsize=(9, 12))
        gs = self.fig.add_gridspec(2, 1, hspace=0)
        self.ax0 = self.fig.add_subplot(gs[0])
        self.ax1 = self.fig.add_subplot(gs[1])
        self.transition_point, = self.ax0.plot([], [], 'o', color='black', markersize=10, zorder=5)

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

    def plot_energy_angles_vs_time(self):
        atom_1=0
        atom_2=1
        atom_3=2
        atom_4=3
        atom_5=4

        # the first subplot
        self.ax0.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 18)
        self.ax0.set_xlim(self.times.min(), self.times.max())
        self.ax0.set_ylim(min(self.S0_energies) - 1, max(self.S1_energies) + 1)
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
        self.ax1.set_ylabel("Angle (degrees)", fontweight = 'bold', fontsize = 18)
        self.ax1.set_xlabel("Time (fs)", fontweight = 'bold', fontsize = 18)
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

        # put legend on first subplot
        handles = handles1 + handles2
        labels = labels1 + labels2
        col = len(labels)    
        self.ax0.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.2), prop={'size': 18}, ncol=col)
        #self.ax1.legend(handles2,labels2,loc='lower center', bbox_to_anchor=(0.5, -0.3), prop={'size': 18}, ncol=3)
        return self.transition_point,

    def init_plot(self):
        self.ax.set_ylabel("Energy (eV)", fontweight = 'bold', fontsize = 16)
        self.ax.set_xlabel("Time (fs)", fontweight = 'bold', fontsize = 16)
        self.ax.set_xlim(self.times.min(), self.times.max())
        self.ax.set_ylim(min(self.S0_energies) - 1, max(self.S1_energies) + 1)
        line1, = self.ax.plot(self.times, self.S1_energies, color='#ff7f0e', zorder=1)
        line0, = self.ax.plot(self.times, self.S0_energies, color='#1f77b4', zorder=1)
        self.transition_point.set_data([], [])
        handles = [line0,line1]
        labels = ['$S_0$','$S_1$']
        self.ax.legend(handles,labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), prop={'size': 14}, ncol=2)
        return self.transition_point,

    def update_plot(self, i):
        ene_i = int(self.output["currstate"][i])  
        ene_curr = (np.array(self.output["energy"])[i,ene_i] - self.ene_min)*self.ev
        self.transition_point.set_data(self.times[i], ene_curr)
        return self.transition_point,

    def create_animation_3(self):
        ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.plot_energy_angles_vs_time, frames=len(self.times), blit=True)
        ani.save('energy_animation_1.mp4', writer='ffmpeg', fps=30)
        plt.close()

    def create_animation(self):
        ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, frames=len(self.times), blit=True)
        ani.save('energy_animation_0.mp4', writer='ffmpeg', fps=30)
        plt.close()

    def create_animation_2(self, trajectory_file):
        #ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, frames=len(self.times), blit=True)
        ani = animation.FuncAnimation(self.fig, self.update_plot, init_func=self.plot_energy_angles_vs_time, frames=len(self.times), blit=True)
        energy_animation_file = 'energy_animation.mp4'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(energy_animation_file, writer=writer)
        plt.close()

        # Load the trajectory video
        trajectory_clip = VideoFileClip(trajectory_file)

        # Load the energy animation video
        energy_clip = VideoFileClip(energy_animation_file)

        # Combine the two videos side by side
        final_clip = clips_array([[trajectory_clip, energy_clip]])

        # Save the final combined video
        final_clip.write_videofile('combined_video.mp4', fps=trajectory_clip.fps)

def movie(traj_movie):
    plot_db = PlotDB('results.db')
    #plot_db.create_animation()
    #plot_db.create_animation_3()
    plot_db.create_animation_2(traj_movie)

if __name__ == '__main__':
    traj_movie = sys.argv[1]
    movie(traj_movie)

