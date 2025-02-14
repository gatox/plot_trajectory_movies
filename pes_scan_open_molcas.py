import numpy as np
from scipy.spatial.transform import Rotation as R

# Original CH2NH coordinates (atomic symbol, x, y, z)
atoms = [
    ["C", -2.11743511,  0.89512660, -0.02741574],
    ["N", -0.88238728,  1.17541371,  0.04299534],
    ["H", -2.82786282,  1.31197674, -0.76383305],
    ["H", -2.53729642,  0.17868499,  0.69474660],
    ["H", -0.63740446,  1.85324286, -0.69052436]
]

# Define torsion scan parameters
torsion_angles = np.arange(0, 181, 10)  # 0° to 180° in 10° steps

# Define the indices for the torsion angle 3-1-2-5 (zero-based index)
idx1, idx2, idx3, idx4 = 2, 0, 1, 4  # H-C-N-H

def rotate_torsion(coords, idx1, idx2, idx3, idx4, angle):
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

for angle in torsion_angles:
    # Rotate molecule
    new_atoms = rotate_torsion(atoms, idx1, idx2, idx3, idx4, angle)
    
    # Write new .xyz file
    xyz_filename = f"CH2NH_{angle}.xyz"
    with open(xyz_filename, "w") as xyz_file:
        xyz_file.write(f"{len(new_atoms)}\nCH2NH torsion angle scan: {angle} degrees\n")
        for atom in new_atoms:
            xyz_file.write(f"{atom[0]} {atom[1]:.8f} {atom[2]:.8f} {atom[3]:.8f}\n")
    
    # Generate OpenMolcas input file
    input_filename = f"ene_grad_nacs_{angle}.input"
    with open(input_filename, "w") as f:
        f.write(f"""&GATEWAY
Title= CH2NH PES Scan - Torsion {angle}°
Coord= {xyz_filename}
Basis= cc-pvdz
Group= NoSym

>> Do while
&SEWARD

&SCF

&RASSCF

 charge = 0
 spin = 1
 levshft = 1.0
 nactel = 4
 ras2 = 3
 frozen = 0
 deleted = 0
 ciroot = 2, 2, 1
 thrs = 1e-08, 0.0001, 0.0001
 expert

&ALASKA
 root = 1
 pnew
 show
 verbose

&ALASKA
 nac = 1, 2
 show
 verbose

>> EndDo
        """)
    
    print(f"Generated: {xyz_filename}, {input_filename}")

