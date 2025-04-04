import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_bending_angle(coords, idx_C, idx_N, idx_H):
    """Compute the bending angle CNH in degrees."""
    pC = np.array(coords[idx_C][1:])
    pN = np.array(coords[idx_N][1:])
    pH = np.array(coords[idx_H][1:])

    NC_vector = pC - pN
    NH_vector = pH - pN

    dot_product = np.dot(NC_vector, NH_vector)
    angle = np.degrees(np.arccos(dot_product / (np.linalg.norm(NC_vector) * np.linalg.norm(NH_vector))))

    return angle

def rotate_bend(coords, idx_C, idx_N, idx_H, angle):
    """ Rotate the CNH bond angle while keeping C and N fixed. """
    
    # Get positions
    pC = np.array(coords[idx_C][1:])  # Carbon (C)
    pN = np.array(coords[idx_N][1:])  # Nitrogen (N)
    pH = np.array(coords[idx_H][1:])  # Hydrogen (H)
    
    # Vector from N to C (fixed axis)
    NC_vector = pC - pN
    NC_vector /= np.linalg.norm(NC_vector)  # Normalize

    # Vector from N to H
    NH_vector = pH - pN

    # Find the normal to the CNH plane
    normal_vector = np.cross(NC_vector, NH_vector)
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize

    # Define rotation matrix about the normal vector
    rotation = R.from_rotvec(np.radians(angle) * normal_vector)
    
    # Rotate H around N
    new_H = pN + rotation.apply(NH_vector)
    
    # Update coordinates
    new_coords = coords.copy()
    new_coords[idx_H] = [coords[idx_H][0], *new_H]

    # Calculate the new bending angle
    new_angle = calculate_bending_angle(new_coords, idx_C, idx_N, idx_H)

    return new_coords, new_angle

def print_input_files(atoms, idx_C, idx_N, idx_H, bending_angles):
    for angle in bending_angles:
        # Rotate molecule
        new_atoms, bending_angle = rotate_bend(atoms, idx_C, idx_N, idx_H, angle)
        
        # Write new .xyz file
        xyz_filename = f"CH2NH_{angle+90}.xyz"
        with open(xyz_filename, "w") as xyz_file:
            xyz_file.write(f"{len(new_atoms)}\nCH2NH torsion angle scan: {angle+90} degrees\n")
            for atom in new_atoms:
                xyz_file.write(f"{atom[0]} {atom[1]:.8f} {atom[2]:.8f} {atom[3]:.8f}\n")
        
        # Generate OpenMolcas input file
        input_filename = f"ene_grad_nacs_{angle+90}.input"
        with open(input_filename, "w") as f:
            f.write(f"""&GATEWAY
    Title= CH2NH PES Scan - Torsion {angle+90}°
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

if __name__=="__main__":

    ## Original CH2NH coordinates (atomic symbol, x, y, z)
    #atoms = [
    #    ["C", -2.11743511, 0.89512660, -0.02741574],
    #    ["N", -0.88238728, 1.17541371,  0.04299534],
    #    ["H", -2.82786282, 1.31197674, -0.76383305],
    #    ["H", -2.53729642, 0.17868499,  0.69474660],
    #    ["H", -0.74364259, 1.98005188,  0.66815726]
    #]

    # CI CH2NH coordinates (atomic symbol, x, y, z)
    atoms = [
        ["N",  0.29112439, -0.23481443,  0.08291748],
        ["C", -0.03494601, -0.07102585,  1.39259648],
        ["H",  0.00394976, -0.91920348,  2.07015809],
        ["H", -0.09397562,  0.92812523,  1.81475871],
        ["H", -0.54423720, -0.35794335, -0.48733323]
    ]
    # Define bending angle scan parameters
    bending_angles = np.arange(-80, 81, 10)  # 10° to 170° in 10° steps

    # Define the indices for the CNH angle (1-2-5)
    idx_C, idx_N, idx_H = 0, 1, 4  # C-N-H

    print_input_files(atoms, idx_C, idx_N, idx_H, bending_angles)
