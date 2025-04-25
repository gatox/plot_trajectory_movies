import numpy as np
import matplotlib.pyplot as plt
import re

ev = 27.211324570273

# Define the torsion angles used in the PES scan
torsion_angles = np.arange(10, 171, 10)  # From 0° to 180° in steps of 10°

# Initialize lists to store extracted data
energy_root1 = []
energy_root2 = []
nac_norms = []

# Loop through all log files and extract information
ang,E0,E1,diff_E = "Angle","E_0(H)","E_1(H)","E_1-E_0(eV)"
report = open("diff_energy_torsion.out","w")
report.write(f"Energy differences report\n")
report.write(f"{ang:>2s} {E0:>9s} {E1:>11s} {diff_E:>14s}\n")
for angle in torsion_angles:
    log_filename = f"ene_grad_nacs_{angle}.log"

    try:
        with open(log_filename, "r") as log_file:
            log_content = log_file.readlines()

        # Extract energies
        E1, E2 = None, None
        for line in log_content:
            if "RASSCF root number  1 Total energy" in line:
                E1_match = re.search(r"[-+]?\d*\.\d+", line)
                if E1_match:
                    E1 = float(E1_match.group())
            if "RASSCF root number  2 Total energy" in line:
                E2_match = re.search(r"[-+]?\d*\.\d+", line)
                if E2_match:
                    E2 = float(E2_match.group())

        # Extract NAC vector and compute Euclidean norm
        nac_vector = []
        capture_nac = False
        for line in log_content:
            if "Total derivative coupling" in line:
                capture_nac = True
                continue  # Move to the next line

            if capture_nac:
                values = re.findall(r"[-+]?\d+\.\d+E[-+]?\d+", line)
                if values:
                    nac_vector.extend([float(v) for v in values])

        # Compute NAC norm
        nac_norm = np.linalg.norm(nac_vector) if nac_vector else None

        # Store values
        energy_root1.append(E1)
        energy_root2.append(E2)
        nac_norms.append(nac_norm)
        report.write(f"{angle:>3.0f}{E1:>12.3f}{E2:>12.3f}{(E2-E1)*ev:>12.3f}\n")


    except FileNotFoundError:
        print(f"Warning: {log_filename} not found. Skipping...")
report.close()

# Convert lists to NumPy arrays
energy_root1 = np.array(energy_root1)
energy_root2 = np.array(energy_root2)
nac_norms = np.array(nac_norms)

# Plot the results
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'hspace': 0})

# Energy plot
axs[0].plot(torsion_angles, ((energy_root1-energy_root1.min())*ev), marker='o', linestyle='-', color='b', label="E_0")
axs[0].plot(torsion_angles, ((energy_root2-energy_root1.min())*ev), marker='s', linestyle='-', color='g', label="E_1")
axs[0].set_ylabel("Energy (eV)")
axs[0].legend()
#axs[0].grid(True)

# NAC Norm plot
axs[1].plot(torsion_angles, nac_norms, marker='d', linestyle='-', color='r', label="NAC Norm")
axs[1].set_xlabel("Torsion Angle (degrees)")
axs[1].set_ylabel("NAC Norm")
axs[1].legend()
#axs[1].grid(True)

# Save the figure
plt.savefig("PES_Scan_Energy_NACs_eV.pdf", bbox_inches='tight')
plt.savefig("PES_Scan_Energy_NACs_eV.png", dpi=300, bbox_inches='tight')

