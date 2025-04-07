import sys
import numpy as np
from saoovqe import SAOOVQE
from jinja2 import Template #build templates
from scipy.spatial.transform import Rotation as R
from Bio.PDB.vectors import (Vector, calc_dihedral, calc_angle)
from pysurf.system import Molecule 


#geometry in units bohr
tpl = Template("""
units    angstrom
0 1
N        0.275456150 -0.240463585  0.082827336 
C       -0.151506941 -0.072806736  1.414485066 
H        0.048673562 -0.905337539  2.082910657 
H       -0.029500464  0.924442070  1.827502235 
H       -0.521206987 -0.360696091 -0.534627765 
symmetry c1
nocom
noreorient 
""")


class GeoPara:

    tpl = tpl
    
    def __init__(self):
        self.aa = 0.529177208
        self.ev = 27.211324570273

    def dihedral(self, a, b, c, d):
        vec_a = a
        vec_b = b
        vec_c = c
        vec_d = d
        return round(calc_dihedral(Vector(vec_a),Vector(vec_b),Vector(vec_c),Vector(vec_d))* 180 / np.pi,2)

    def angle(self, a, b, c):
        vec_a = a
        vec_b = b
        vec_c = c
        return round(calc_angle(Vector(vec_a),Vector(vec_b),Vector(vec_c))* 180 / np.pi,2)

    def pyramidalization_angle(self, a, b, c, o):
        vec_a = a - o
        vec_b = b - o
        vec_c = c - o
        vec_u = np.cross(vec_a, vec_b)
        d_cu = np.dot(vec_c,vec_u)
        cr_cu = np.cross(vec_c, vec_u)
        n_cr_cu = np.linalg.norm(cr_cu)
        res = np.math.atan2(n_cr_cu,d_cu)
        return round(90 - np.degrees(res),2)

    def read_xyz_file(self,inputfile):
        atoms = []
        with open(inputfile, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:  # Skip the first two lines (atom count and comment)
                parts = line.split()
                atom = parts[0]
                coordinates = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                atoms.append((atom, coordinates))
        return atoms

    def get_angle_dihe_pyr(self, inputfile):
        atoms = self.read_xyz_file(inputfile)
        atom_1 = atoms[0][1] 
        atom_2 = atoms[1][1] 
        atom_3 = atoms[2][1] 
        atom_4 = atoms[3][1] 
        atom_5 = atoms[4][1] 
        dihedral = self.dihedral(atom_3, atom_2, atom_1, atom_5)
        angle = self.angle(atom_2, atom_1, atom_5)
        pyr = self.pyramidalization_angle(atom_3, atom_4, atom_1, atom_2)
        print("dihedral: ", dihedral)
        print("angle: ", angle)
        print("pyramidalization: ", pyr)

    def do_saoovqe(self):
        chg = 0
        mult = 1
        basis = 'cc-pvdz'
        nelec_active = 4  # Number of active electrons in the Active-Space
        frozen_indices = [i for i in range(6)]
        active_indices = [i for i in range(6, 9)]
        virtual_indices = [i for i in range(9, 43)]
        num_qubits = 2 * len(active_indices)  # Total number of qubits necessary
        do_oo_process = True
        noise = False
        noise_mean=None
        noise_sd=None
        noise_energy_after_sa_vqe = False
        noise_before_orb_opt_phase = False
        noise_final_state_resolution = False
        noise_vqe_cost_function_energy = False
        noise_rdms_gradient = False
        noise_tdms_nacs = False
        string_geo = self.tpl.render() 
        print(string_geo)
        saoovqe_class = SAOOVQE(string_geo,
                  basis,
                  nelec_active,
                  frozen_indices,
                  active_indices,
                  virtual_indices,
                  tell_me=True,
                  w_a=0.5,
                  w_b=0.5,
                  delta=1e-5,
                  print_timings=False, # Use this if you want to compute all the timings...
                  do_oo_process=True,
                  add_noise=noise,
                  noise_mean=noise_mean,
                  noise_sd=noise_sd,
                  ucc_ansatz=["fermionic_SAAD", "fast"][1],
                  initial_param_values=None)
        saoovqe_class.vqe_kernel()
        """Saving energies, gradients and NACs"""
        self.energies = [saoovqe_class.e_a, saoovqe_class.e_b]
        nac = []
        natoms = 5
        for i in range(natoms):
            dx,dy,dz = saoovqe_class.get_gradient(i,state)
            nx,ny,nz = saoovqe_class.get_nac(i)
            nac.append([-nx,-ny,-nz])
        self.nacs = nac

    def _nacs(self):
        nacs = {}
        nacs.update({(0,1):np.array(self.nacs)})
        nacs.update({(1,0):-np.array(self.nacs)})
        return nacs

    def rotate_torsion(self, coords, idx1, idx2, idx3, idx4, angle):
        atoms = coords
        """ Rotate the dihedral angle around the axis passing through idx2 and idx3. """
        p1, p2, p3, p4 = np.array(coords[idx1][1:]), np.array(coords[idx2][1:]), np.array(coords[idx3][1:]), np.array(coords[idx4][1:])
        
        # Define rotation axis (from atom 2 to atom 3)
        axis = (p3 - p2)
        axis /= np.linalg.norm(axis)  # Normalize
        
        # Translate system so p3 is at origin
        translated_coords = np.array([np.array(c[1:]) - p3 for c in coords])
        
        # Apply rotation to p4 (and all atoms connected to p4)
        rotation = R.from_rotvec(np.radians(angle) * axis)  # Rotation matrix
        for i in range(len(coords)):
            if i in {idx4}:  # Rotate only atom 4 (rigid scan)
                translated_coords[i] = rotation.apply(translated_coords[i])
        
        # Translate back
        new_coords = translated_coords + p3
        
        # Reconstruct the coordinates list
        new_atoms = [[atoms[i][0], *new_coords[i]] for i in range(len(coords))]
        
        return new_atoms

    def print_input_files(self, chg, mult, atoms, idx1, idx2, idx3, idx4, torsion_angles):
        for angle in torsion_angles:
            # Rotate molecule
            new_atoms = self.rotate_torsion(atoms, idx1, idx2, idx3, idx4, angle)
    
            # Generate input file
            input_filename = f"ene_grad_nacs_{angle+90}.input"
            with open(input_filename, "w") as f:
                f.write(f"units    Angstrom\n")
                f.write(f"{chg} {mult}\n")
                for atom in new_atoms:
                    f.write(f"{atom[0]} {atom[1]:.8f} {atom[2]:.8f} {atom[3]:.8f}\n")
                f.write(f"symmetry c1\n")
                f.write(f"nocom\n")
                f.write(f"noreorient\n")

if __name__=='__main__':
    # CI CH2NH coordinates (atomic symbol, x, y, z)
    atoms = [
        ["N",  0.291124387, -0.234814432,  0.082917480],
        ["C", -0.034946007, -0.071025849,  1.392596482],
        ["H",  0.003949762, -0.919203484,  2.070158087],
        ["H", -0.093975620,  0.928125232,  1.814758706],
        ["H", -0.544237201, -0.357943347, -0.487333226]
    ]
    # Define bending angle scan parameters
    torsion_angles = np.arange(-80, 81, 10)  # 10° to 170° in 10° steps

    # Define the indices for the torsion angle 3-1-2-5 (zero-based index)
    idx1, idx2, idx3, idx4 = 2, 0, 1, 4  # H-C-N-H

    result = GeoPara()
    result.print_input_files(0,1, atoms, idx1, idx2, idx3, idx4, torsion_angles)
