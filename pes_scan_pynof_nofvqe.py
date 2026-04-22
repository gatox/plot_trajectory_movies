import re
from pathlib import Path

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern-like
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "axes.linewidth": 1.2,
    "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt


import numpy as np


def extract_energies_from_out(out_file: Path):
    """
    Extract HF, NOF, and correlation energies from a PyNOF output file.
    """
    text = out_file.read_text(encoding="utf-8", errors="ignore")

    hf_match = re.search(r"HF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
    nof_match = re.search(r"Final NOF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
    corr_match = re.search(r"Correlation Energy\s*=\s*([-\d\.Ee+]+)", text)

    hf = float(hf_match.group(1)) if hf_match else None
    nof = float(nof_match.group(1)) if nof_match else None
    corr = float(corr_match.group(1)) if corr_match else None

    return hf, nof, corr


def extract_distance_from_folder(folder_name: str):
    """
    Extract distance in angstrom from folder name like:
    cube_h_2_d_4.000_ang
    """
    m = re.search(r"_d_([0-9]+\.[0-9]+)_ang", folder_name)
    if not m:
        raise ValueError(f"Could not extract distance from folder name: {folder_name}")
    return float(m.group(1))


def compute_pes(base_dir="cube_h2_pynof", save_prefix="pynof_pes", guess=False, hf_only=False):
    """
    Reads all output folders inside base_dir:
      - HF energy vs distance
      - NOF energy vs distance

    Saves:
      - save_prefix.dat
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    distances = []
    hf_energies = []
    nof_energies = []
    corr_energies = []

    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue

        try:
            d_ang = extract_distance_from_folder(folder.name)
        except ValueError:
            continue

        out_files = list(folder.glob("*.out"))
        if not out_files:
            print(f"[WARNING] No .out file found in {folder}")
            continue

        out_file = out_files[0]
        hf, nof, corr = extract_energies_from_out(out_file)

        if hf is None or corr is None:
            print(f"[WARNING] Could not extract HF energy from {out_file}")
            #continue

        distances.append(d_ang)
        hf_energies.append(hf)
        nof_energies.append(nof)
        corr_energies.append(corr if corr is not None else np.nan)

    if not distances:
        raise RuntimeError("No valid data found to plot.")

    # Sort by distance
    distances = np.array(distances)
    hf_energies = np.array(hf_energies)
    nof_energies = np.array(nof_energies)
    corr_energies = np.array(corr_energies)

    idx = np.argsort(distances)
    distances = distances[idx]
    hf_energies = hf_energies[idx]
    nof_energies = nof_energies[idx]
    corr_energies = corr_energies[idx]
    
    # Save data table
    if guess:
        
        data_out = np.column_stack([distances, nof_energies])
        header = "distance_ang  NOF_energy_Ha"
        np.savetxt(f"{save_prefix}.dat", data_out, header=header, fmt="%.8f")
        
        return distances, nof_energies
    elif hf_only:
        
        data_out = np.column_stack([distances, hf_energies])
        header = "distance_ang  NOF_energy_Ha"
        np.savetxt(f"{save_prefix}.dat", data_out, header=header, fmt="%.8f")
        
        return distances, hf_energies
    else:
        data_out = np.column_stack([distances, hf_energies, nof_energies, corr_energies])
        header = "distance_ang  HF_energy_Ha  NOF_energy_Ha  Corr_energy_Ha"
        np.savetxt(f"{save_prefix}.dat", data_out, header=header, fmt="%.8f")
        
        return distances, hf_energies, nof_energies, corr_energies

def plot_pes(data_1, data_2, save_prefix="pes_scan"):
    arr1 = np.loadtxt(data_1)
    arr2 = np.loadtxt(data_2)

    distances = arr1[:, 0]
    hf_energies = arr2[:, 1]
    energies_pynof = arr1[:, 1]
    energies_nofvqe = arr2[:, 2]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(
        distances,
        hf_energies,
        linestyle="--",
        marker="o",
        markersize=5,
        label=r"HF",
    )
    ax.plot(
        distances,
        energies_pynof,
        linestyle="-",
        marker="s",
        markersize=5,
        label=r"PyNOF (PNOF7)",
    )
    ax.plot(
        distances,
        energies_nofvqe,
        linestyle="-",
        marker="^",
        markersize=5,
        label=r"NOF-VQE (PNOF7)",
    )

    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Energy (Ha)")
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.eps", format="eps")
    fig.savefig(f"{save_prefix}.png", dpi=300)
    
def plot_pes_hf(data_1, data_2, save_prefix="pes_scan_HF"):
    arr1 = np.loadtxt(data_1)
    arr2 = np.loadtxt(data_2) #HF

    distances = arr1[:, 0]
    hf_energies = arr2[:, 1]
    energies_pynof = arr1[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(
        distances,
        hf_energies,
        linestyle="--",
        marker="o",
        markersize=5,
        label=r"HF",
    )
    ax.plot(
        distances,
        energies_pynof,
        linestyle="-",
        marker="s",
        markersize=5,
        label=r"PyNOF (PNOF7)",
    )
    # ax.plot(
    #     distances,
    #     hf_energies,
    #     linestyle="-",
    #     marker="^",
    #     markersize=5,
    #     label=r"NOF-VQE (PNOF7)",
    # )

    ax.set_ylim(-32.0,-22.2)
    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Energy (Ha)")
    ax.legend(frameon=False)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.eps", format="eps")
    fig.savefig(f"{save_prefix}.png", dpi=300)

if __name__ == "__main__":
    # compute_pes(base_dir="cube_h4_pynof", save_prefix="cube_h4_pynof_pes", guess=True)
    # compute_pes(base_dir="cube_h4_pynof_HF_energy", save_prefix="cube_h4_pynof_HF_energy_pes", hf_only=True)
    # compute_pes(base_dir="cube_h2_nofvqe", save_prefix="cube_h2_nofvqe_pes")
    # pynof_data = "cube_h2_pynof_pes.dat"
    # nofvqe_data = "cube_h2_nofvqe_pes.dat" 
    # plot_pes(pynof_data,nofvqe_data)
    pynof_data = "cube_h4_pynof_pes.dat"
    # nofvqe_data = "cube_h4_nofvqe_pes.dat" 
    hf_data = "cube_h4_pynof_HF_energy_pes.dat"
    # plot_pes(pynof_data,nofvqe_data)
    plot_pes_hf(pynof_data,hf_data)