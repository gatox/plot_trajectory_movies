import os
import re
import sys
import contextlib
import shutil

from pathlib import Path

import numpy as np

ANG_TO_BOHR = 1.8897259886

class GenCubeHAtoms:

    def __init__(self, natoms, distance_bohr, charge=0, multiplicity=1):
        self.natoms = natoms
        self.dis = distance_bohr
        self.charge = charge
        self.mult = multiplicity

    def generate_particles(self, origin=(0, 0, 0)):
        x0, y0, z0 = origin
        particles = []

        for i in range(self.natoms):
            for j in range(self.natoms):
                for k in range(self.natoms):
                    x = x0 + i * self.dis
                    y = y0 + j * self.dis
                    z = z0 + k * self.dis
                    particles.append((x, y, z))

        return particles

    def to_xyz(self, filepath):
        particles = self.generate_particles()
        n_total = len(particles)

        with open(filepath, "w") as f:
            f.write("units bohr\n")
            f.write(f"{self.charge} {self.mult}\n")

            for (x, y, z) in particles:
                f.write(f"H {x:.6f} {y:.6f} {z:.6f}\n")

        print(f"Written: {filepath} ({n_total} atoms)")
        
    def extract_vector_after(self, label, text):
        """
        Extract a numpy array following a specified label string.
        Supports single-line, multi-line, and scientific notation formats.
        """
        # Matches label -> optional colon -> optional spaces/newlines -> [...]
        pattern = rf"{re.escape(label)}\s*:?\s*\[(.*?)\]"
        
        # re.DOTALL ensures the expression matches across multiple lines
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return None
            
        # Standardize spacing by replacing newlines with spaces
        clean_content = m.group(1).replace("\n", " ").strip()
        
        # Safely convert whitespace-separated floats into a numpy array
        return np.fromstring(clean_content, sep=" ")
        
    def extract_save_opts_params_ON(self, filename):
        """
        Extract and save optimal parameters and occupation numbers from 
        a nofvqe output file.
        """
        base = os.path.splitext(filename)[0]
        out_file = base + ".out"
        out_file = Path(out_file)
        
        
        text = out_file.read_text(encoding="utf-8", errors="ignore")
        raw_n = self.extract_vector_after("n_full RAW", text)
        sim_n = self.extract_vector_after("Reference simulator occupation vector", text)
        print(f"Saving n_raw and n_sim from output file in {filename}")
        np.save("nofvqe_qc_eva_n.npy",raw_n)
        np.save("nofvqe_sim_n.npy",sim_n)

    def run_pynof(self, filename, pnof, basis, hf_energy):
        import pynof
        
        base = os.path.splitext(filename)[0]
        out_file = base + ".out"
        
        # Initial parameters
        C_MO = None
        ON = None
        
        # Checking if there are guests
        if os.path.isfile(os.getcwd() + "/pynof_C.npy"):
            C_MO = pynof.read_C()
            ON = pynof.read_n()

        # 2. Redirect the terminal output to that .out file
        with open(out_file, "w") as f:
            # Save original terminal address
            original_stdout_fd = sys.stdout.fileno()
            saved_stdout_fd = os.dup(original_stdout_fd)
            try:
                # Point terminal output to our file
                os.dup2(f.fileno(), original_stdout_fd)
                
                # --- RUN PYNOF ---
                # Reading xyz file and generating mol file
                _, _, _, _, _, mol = pynof.read_mol(filename)
                p = pynof.param(mol,str(basis))
                p.ipnof=int(pnof)
                p.RI = False
                if hf_energy:
                    import psi4
                    psi4.set_options({
                        'scf__maxiter': 600,
                        #'scf__guess': 'sad',           # Better starting density than the core Hamiltonian
                        #'scf__damping_percentage': 40, # Mixes 40% of old density to stop oscillations
                        'scf__soscf': True             # Second-order converter (slower but much more robust)
                    })
                    E = pynof.compute_energy(mol,p,hf_energy=hf_energy)
                else:
                    E,C,gamma,fmiug0 = pynof.compute_energy(mol,p, C=C_MO,n=ON)
                sys.stdout.flush() 
            finally:
                # Restore the terminal output
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.close(saved_stdout_fd)
        
    def _run_nofvqe(self, filename, pnof, basis, C_MO, init_param, C_MO_opt=None, n_opt=None, n_raw=None, params_opt=None, energy_eval=False):
        from nofvqe import NOFVQE
        
        base = os.path.splitext(filename)[0]
        out_file = base + ".out"
        if energy_eval:
            out_file = "eva_" + out_file
            dev = "real"
        else:
            dev = "simulator"
        
        # Initial parameters
        functional="pnof"+str(pnof)
        conv_tol=1e-6
        max_iterations = 200
        gradient="analytics"
        d_shift=1e-4
        dev=dev
        opt_circ="slsqp"
        n_shots=10000
        optimization_level=3
        resilience_level=2
        pair_double = True
        comp = NOFVQE(
                filename, 
                functional=functional, 
                conv_tol=conv_tol, 
                init_param=init_param, 
                basis=basis, 
                max_iterations=max_iterations,
                opt_circ=opt_circ,
                gradient=gradient,
                pair_double=pair_double,
                d_shift=d_shift,
                C_MO=C_MO,
                dev=dev,
                n_shots=n_shots,
                optimization_level=optimization_level,
                resilience_level=resilience_level,
                    )
        C_opt=None 
        # 1. Redirect the terminal output to that .out file
        with open(out_file, "w") as f:
            # Save original terminal address
            original_stdout_fd = sys.stdout.fileno()
            saved_stdout_fd = os.dup(original_stdout_fd)
            try:
                # Point terminal output to our file
                os.dup2(f.fileno(), original_stdout_fd)
                # Energy evaluated from Optimal C_MO and ONs
                if energy_eval:
                    comp._evaluate_energy(C_MO_opt, n_opt, n_raw, params_opt)
                else:
                    # --- RUN PYNOF ---
                    # Reading xyz file and generating mol file
                    E_min, params_opt, _, _, _, _, _, C_opt, _ = comp.run_nofvqe()
                sys.stdout.flush() 
            finally:
                # Restore the terminal output
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.close(saved_stdout_fd)
        return C_opt, params_opt
    
    def copy_C_MO(self, subfolder_path, obj):

        src = os.path.join(subfolder_path, obj)
        dst = os.path.join(os.getcwd(), obj)

        print("Source:", src)
        print("Destination:", dst)

        if os.path.isfile(src):
            shutil.copy2(src, dst)

        
if __name__ == "__main__":
    
    base_dir = sys.argv[1]
    sub_dir = None
    # sub_dir = Path(sub_dir).expanduser()
    # sub_dir = Path(
    #     "~/Desktop/Edison/PostDoc_DIPC/cube_h_2/noiseless_cube_h2_nofvqe_sto-3g/noise_sim_0_cube_h2_nofvqe/"
    # ).expanduser()
    copy_C_MO = False
    
    if sub_dir is not None:
        copy_C_MO =True
    
    # ---- USER INPUT ----
    natoms = 4  # number per dimension (Hn cube → nxnxn)
    pnof = 7 # type of functional: pnof4 = 4, pnof5 =5 ,pnof7 = 7 and gnof =8
    basis = "sto-3g" # basis set
    method = "nofvqe" # Either nofvqe or pynof
    hf_energy = False # To compute HF energy only with pynof
    
    distances_ang = [
        1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
        2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0
    ]
    
    #number_folder = "noise_sim_0_"

    number_folder = "t_9_"
    
    if base_dir is None:
        if hf_energy:
            base_dir = number_folder + "cube_h"+ str(natoms) + "_" + method + "_HF_energy"
        else: 
            base_dir = number_folder + "cube_h"+ str(natoms) + "_" + method

    # ---- CREATE BASE DIR ----
    os.makedirs(base_dir, exist_ok=True)
    
    previous_folder = None
    C_MO = None
    init_param = None
    for d_ang in distances_ang:

        # Convert to bohr
        d_bohr = d_ang * ANG_TO_BOHR

        # Folder name
        folder_name = f"cube_h_{natoms}_d_{d_ang:.3f}_ang"
        folder_path = os.path.join(base_dir, folder_name)
        if copy_C_MO:
            subfolder_path = os.path.join(sub_dir, folder_name)

        os.makedirs(folder_path, exist_ok=True)

        # File name
        file_name = f"{folder_name}.xyz"
        
        file_path = os.path.join(folder_path, file_name)
        
        # --- NEW: COPY FILES FROM PREVIOUS FOLDER ---
        if previous_folder is not None:
            for filename in ["pynof_n.npy", "pynof_C.npy"]:
                src = os.path.join(previous_folder, filename)
                dst = os.path.join(folder_path, filename)
                
                if os.path.isfile(src):
                    shutil.copy(src, dst)  # Copy to the NEW folder

        # Generate system
        cal = GenCubeHAtoms(natoms, d_bohr)
        
        if not os.path.isfile(file_path):
            cal.to_xyz(file_path)
            
        # Save where we are now
        original_dir = os.getcwd()      
        # Jump into the specific folder
        os.chdir(folder_path)          
        
        
        try:
            # Note: since we are NOW inside 'folder_path', 
            # we use 'file_name' instead of 'file_path'
            if method == "pynof":
                cal.run_pynof(file_name, pnof, basis, hf_energy)
            elif method == "nofvqe":
                C_MO_opt = None
                n_opt = None
                n_raw = None
                if not os.path.isfile("nofvqe_sim_n.npy") and os.path.isfile("pynof_C.npy"):
                    print("")
                    print("**********************")
                    print("* Reading C_MO guess *")
                    print("**********************")
                    print("")
                    C_MO = np.load("pynof_C.npy")
                    energy_eval = False
                #if os.path.isfile("nofvqe_sim_params.npy") and os.path.isfile("nofvqe_C.npy"):
                #    print("")
                #    print("************************************")
                #    print("* Reading C_MO and params as guess *")
                #    print("************************************")
                #    print("")
                #    C_MO = np.load("nofvqe_C.npy")
                #    init_param = np.load("nofvqe_sim_params.npy")
                #    energy_eval = False
                if os.path.isfile("nofvqe_sim_params.npy") and os.path.isfile("nofvqe_C.npy"):
                    print("")
                    print("********************************************************************")
                    print("* Reading C_MO and Params. as guess to evaluate the Ene. in the QC *")
                    print("********************************************************************")
                    print("")
                    C_MO_opt = np.load("nofvqe_C.npy")
                    n_opt = np.load("nofvqe_sim_n.npy")
                    params_opt = np.load("nofvqe_sim_params.npy")
                    energy_eval = True
                    
                    if os.path.isfile("nofvqe_qc_eva_n.npy"):
                        n_raw = np.load("nofvqe_qc_eva_n.npy")
                        
                else:
                    
                    energy_eval = False
                #cal.extract_save_opts_params_ON(file_name)
                #cal.copy_C_MO(subfolder_path,"nofvqe_C.npy")
                C_MO, init_param = cal._run_nofvqe(file_name, pnof, basis, C_MO, init_param, C_MO_opt, n_opt, n_raw, params_opt, energy_eval)      
        finally:
            os.chdir(original_dir)      # Always jump back to the main folder
            
        # Update previous_folder for the next step in the loop
        previous_folder = folder_path
