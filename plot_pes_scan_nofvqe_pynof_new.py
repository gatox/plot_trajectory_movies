import re
import os
from pathlib import Path

import matplotlib as mpl


mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "axes.labelsize": 18,
    "font.size": 16,
    "legend.fontsize": 13.5,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 1.5,
    "axes.linewidth": 1.5,
    "savefig.bbox": "tight",
})

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from scipy.interpolate import make_interp_spline


import numpy as np

def extract_bool(x):
    return x == "True"


def extract_vector_after(label, text):
    """
    Extract vector printed after a label, allowing line breaks.
    """
    pattern = rf"{re.escape(label)}:\s*\n\[(.*?)\]"
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    return np.fromstring(m.group(1).replace("\n", " "), sep=" ")


def extract_qc_summary(text):
    m = re.search(r"QC_ON_SUMMARY\s+(.*)", text)
    if not m:
        return {}

    out = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", m.group(1)):
        if value in ["True", "False"]:
            out[key] = extract_bool(value)
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def extract_run_record(out_file: Path, device="hybrid_real"):
    text = out_file.read_text(encoding="utf-8", errors="ignore")

    d_ang = extract_distance_from_folder(out_file.parent.name)

    hf, nof, corr, sim_nof, sim_corr = extract_energies_from_out(
        out_file, device=device
    )

    raw_n = extract_vector_after("n_full RAW", text)
    corr_n = extract_vector_after("Evaluated/corrected occupation vector", text)
    sim_n = extract_vector_after("Reference simulator occupation vector", text)

    summary = extract_qc_summary(text)

    abs_diff_e = None
    m = re.search(r"Abs_Diff_E\s*=\s*([-\d\.Ee+]+)", text)
    if m:
        abs_diff_e = float(m.group(1))

    diff_n = None
    m = re.search(r"\|\|Diff_n\|\|\s*=\s*([-\d\.Ee+]+)", text)
    if m:
        diff_n = float(m.group(1))

    record = {
        "distance": d_ang,
        "HF": hf,
        "E_qc": nof,
        "Corr_qc": corr,
        "E_sim": sim_nof,
        "Corr_sim": sim_corr,
        "Abs_Diff_E": abs_diff_e,
        "Diff_n": diff_n,
        "raw_n": raw_n,
        "corr_n": corr_n,
        "sim_n": sim_n,
    }

    record.update(summary)
    return record

def collect_qc_records(base_dirs, device="hybrid_real"):
    records = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)

        for folder in sorted(base_path.iterdir()):
            if not folder.is_dir():
                continue

            out_files = list(folder.glob("eva_*.out"))
            print("Records from:",out_files)
            if not out_files:
                continue

            try:
                rec = extract_run_record(out_files[0], device=device)
                rec["run"] = base_path.name
                records.append(rec)
            except Exception as e:
                print(f"[WARNING] failed reading {out_files[0]}: {e}")

    return records

def physics_filter(rec):
    return bool(rec.get("accepted_for_statistics", False))


def strict_reference_filter(rec, max_diff_n=1, max_abs_diff_e=0.2):
    if rec.get("Diff_n") is None or rec.get("Abs_Diff_E") is None:
        return False

    return (
        rec["Diff_n"] < max_diff_n
        and rec["Abs_Diff_E"] < max_abs_diff_e
    )
    
def summarize_by_distance(records, filter_fn=None):
    distances = sorted(set(r["distance"] for r in records))
    rows = []

    for d in distances:
        group = [r for r in records if r["distance"] == d]

        if filter_fn is not None:
            used = [r for r in group if filter_fn(r)]
        else:
            used = group

        if len(used) == 0:
            rows.append([d, len(group), 0, np.nan, np.nan, np.nan, np.nan])
            continue

        E_qc = np.array([r["E_qc"] for r in used], dtype=float)
        E_sim = np.array([r["E_sim"] for r in used], dtype=float)
        diff_e = np.array([r["Abs_Diff_E"] for r in used], dtype=float)
        diff_n = np.array([r["Diff_n"] for r in used], dtype=float)

        rows.append([
            d,
            len(group),
            len(used),
            np.mean(E_qc),
            np.std(E_qc, ddof=1) if len(used) > 1 else 0.0,
            np.mean(E_sim),
            np.mean(diff_e),
            np.mean(diff_n),
        ])

    arr = np.array(rows, dtype=float)

    header = (
        "distance n_total n_used "
        "E_qc_mean E_qc_std E_sim_mean Abs_Diff_E_mean Diff_n_mean"
    )

    return arr, header

def extract_energies_from_out(out_file: Path, device=None):
    """
    Extract HF, NOF, and correlation energies from a PyNOF output file.
    """
    text = out_file.read_text(encoding="utf-8", errors="ignore")

    hf_match = re.search(r"HF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
    if device.startswith("hybrid_"):
        nof_match = re.search(r"Final NOF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
        corr_match = re.search(r"Correlation Energy\s*=\s*([-\d\.Ee+]+)", text)
        sim_nof_match = re.search(r"Sim. Final NOF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
        sim_corr_match = re.search(r"Sim.     Correlation Energy\s*=\s*([-\d\.Ee+]+)", text)
    else:
        nof_match = re.search(r"Final NOF Total Energy\s*=\s*([-\d\.Ee+]+)", text)
        corr_match = re.search(r"Correlation Energy\s*=\s*([-\d\.Ee+]+)", text)

    hf = float(hf_match.group(1)) if hf_match else None
    nof = float(nof_match.group(1)) if nof_match else None
    corr = float(corr_match.group(1)) if corr_match else None
    sim_nof = float(sim_nof_match.group(1)) if sim_nof_match else None
    sim_corr = float(sim_corr_match.group(1)) if sim_corr_match else None

    return hf, nof, corr, sim_nof, sim_corr


def extract_distance_from_folder(folder_name: str):
    """
    Extract distance in angstrom from folder name like:
    cube_h_2_d_4.000_ang
    """
    m = re.search(r"_d_([0-9]+\.[0-9]+)_ang", folder_name)
    if not m:
        raise ValueError(f"Could not extract distance from folder name: {folder_name}")
    return float(m.group(1))


def compute_pes(base_dir="cube_h2_pynof", save_prefix="pynof_pes", guess=False, device=None, hf_only=False):
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
    sim_nof_energies = []
    sim_corr_energies = []

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
        hf, nof, corr, sim_nof, sim_corr = extract_energies_from_out(out_file, device)

        if hf is None or corr is None:
            print(f"[WARNING] Could not extract HF energy from {out_file}")
            #continue

        distances.append(d_ang)
        hf_energies.append(hf)
        nof_energies.append(nof)
        corr_energies.append(corr if corr is not None else np.nan)
        sim_nof_energies.append(sim_nof if sim_nof is not None else np.nan)
        sim_corr_energies.append(sim_corr if sim_corr is not None else np.nan)

    if not distances:
        raise RuntimeError("No valid data found to plot.")

    # Sort by distance
    distances = np.array(distances)
    hf_energies = np.array(hf_energies)
    nof_energies = np.array(nof_energies)
    corr_energies = np.array(corr_energies)
    sim_nof_energies = np.array(sim_nof_energies)
    sim_corr_energies = np.array(sim_corr_energies)

    idx = np.argsort(distances)
    distances = distances[idx]
    hf_energies = hf_energies[idx]
    nof_energies = nof_energies[idx]
    corr_energies = corr_energies[idx]
    sim_nof_energies = sim_nof_energies[idx]
    sim_corr_energies = sim_corr_energies[idx]

    # Save data table
    if guess:

        data_out = np.column_stack([distances, nof_energies])
        header = "distance_ang  NOF_energy_Ha"
        np.savetxt(f"{save_prefix}.dat", data_out, header=header, fmt="%.8f")

        return distances, nof_energies
    elif hf_only:

        data_out = np.column_stack([distances, hf_energies])
        header = "distance_ang  HF_energy_Ha"
        np.savetxt(f"{save_prefix}.dat", data_out, header=header, fmt="%.8f")

        return distances, hf_energies
    elif device.startswith("hybrid_"):

        data_out = np.column_stack([distances, hf_energies, nof_energies, corr_energies, sim_nof_energies, sim_corr_energies])
        header = "distance_ang  HF_energy_Ha  NOF_energy_Ha  Corr_energy_Ha  Sim_NOF_energy_Ha  Sim_Corr_energy_Ha"
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

def plot_pes_sim_qc(data_1, data_2, save_prefix="pes_H8_scan_hybrid"):
    arr1 = np.loadtxt(data_1) #simulation
    arr2 = np.loadtxt(data_2) #real QC

    distances = arr1[:, 0]
    hf_energies = arr1[:, 1]
    sim_nofvqe = arr1[:, 2]
    qc_nofvqe = arr2[:, 2]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # ax.plot(
    #     distances,
    #     hf_energies,
    #     linestyle="--",
    #     marker="o",
    #     markersize=5,
    #     label=r"HF",
    # )
    ax.plot(
        distances,
        sim_nofvqe,
        linestyle="--",
        marker="s",
        markersize=5,
        label=r"Noiseless ON-NOF-VQE",
    )
    ax.scatter(
        distances,
        qc_nofvqe,
        linestyle="-",
        marker="o",
        color="red",
        s=40,
        label=r"Quantum NOF-VQE",
    )

    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Energy (Ha)")
    ax.legend(frameon=True)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf")
    # fig.savefig(f"{save_prefix}.eps", format="eps")
    # fig.savefig(f"{save_prefix}.png", dpi=300)

def _accepted_rejected_qc_samples(data_qc_list, qc_mean, qc_std, ax):
    """
    Overlay individual accepted/rejected QC samples.

    Accepted points: faint filled circles.
    Rejected points: gray open circles.
    """

    accepted_label_used = False
    rejected_label_used = False

    for data_qc in data_qc_list:

        arr_qc = np.loadtxt(data_qc)

        dvals = arr_qc[:, 0]
        qcvals = arr_qc[:, 2]

        analysis_file = data_qc.replace(".dat", "_analysis.dat")

        if os.path.exists(analysis_file):
            analysis = np.loadtxt(analysis_file)

            if analysis.ndim == 1:
                analysis = analysis.reshape(1, -1)

            if len(analysis) != len(dvals):
                print(f"[WARNING] Analysis file size mismatch: {analysis_file}")
                accepted = np.ones(len(dvals), dtype=bool)
            else:
                accepted = analysis[:, 1].astype(bool)
        else:
            print(f"[WARNING] No analysis file found for {data_qc}. Assuming all accepted.")
            accepted = np.ones(len(dvals), dtype=bool)

        # Accepted errobar
        ax.errorbar(
                dvals[accepted],
                qc_mean[accepted],
                yerr=qc_std[accepted],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                fmt="o",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                markersize=5,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                capsize=3,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                label=r"Quantum NOF-VQE",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            )

        # Accepted points
        ax.scatter(
            dvals[accepted],
            qcvals[accepted],
            s=20,
            alpha=0.30,
            label="Accepted QC samples" if not accepted_label_used else None,
            zorder=2,
        )
        accepted_label_used = True

        # Rejected points
        if np.any(~accepted):
            ax.scatter(
                dvals[~accepted],
                qcvals[~accepted],
                s=42,
                facecolors="none",
                edgecolors="gray",
                linewidths=1.2,
                alpha=0.95,
                marker="o",
                label="Rejected QC samples" if not rejected_label_used else None,
                zorder=4,
            )
            rejected_label_used = True

def save_analysis_tables_by_run(records, base_path):

    runs = sorted(set(r["run"] for r in records))

    for run in runs:

        rec_run = [r for r in records if r["run"] == run]
        rec_run = sorted(rec_run, key=lambda x: x["distance"])

        rows = []

        for r in rec_run:

            rows.append([
                r["distance"],
                int(r.get("accepted_for_statistics", False)),
                r.get("Abs_Diff_E", np.nan),
                r.get("Diff_n", np.nan),
            ])

        rows = np.array(rows)

        run_name = Path(run).name

        out_file = Path(base_path) / f"{run_name}_pes_analysis.dat"

        np.savetxt(
            out_file,
            rows,
            header="distance accepted_for_statistics Abs_Diff_E Diff_n",
            fmt="%.10f",
        )

def _load_acceptance_mask(data_qc):
    """
    Load accepted_for_statistics mask from companion *_analysis.dat file.

    Expected columns:
        distance accepted_for_statistics Abs_Diff_E Diff_n
    """
    analysis_file = data_qc.replace(".dat", "_analysis.dat")

    if not os.path.exists(analysis_file):
        print(f"[WARNING] No analysis file found for {data_qc}. Assuming all accepted.")
        return None

    analysis = np.loadtxt(analysis_file)

    if analysis.ndim == 1:
        analysis = analysis.reshape(1, -1)
    
    
    #strict_reference_filter:
    #For diff_n
    max_diff_n=5
    #For abs_diff_E
    max_abs_diff_e=5
    return analysis[:, 1].astype(bool) & (analysis[:, 2] < max_abs_diff_e) & (analysis[:, 3] < max_diff_n)


def plot_pes_sim_qc_stats(
    data_qc_list,
    save_prefix="pes_H8_scan_hybrid_stats",
    # lower_ylim=(-4.5, -3.50),
    # upper_ylim=(-3.650, -3.150),
    lower_ylim=(-4.05, -3.70),
    #upper_ylim=(-3.650, -3.150),
):
    """
    O'Brien-style stacked discontinuous-energy plot.

    Upper panel:
        high-energy discarded QC samples.
        No x-axis labels/tick labels.
        No y-axis label.

    Lower panel:
        physical PEC region.
        Classical NOF-VQE, postselected QC samples,
        discarded QC samples, and postselected QC mean.
        Legend only here.

    Mean/std are computed using postselected QC samples only.
    """
    arr_ref = np.loadtxt(data_qc_list[0])
    
    distances = arr_ref[:, 0]
    hf_energy = arr_ref[:, 1]

    if arr_ref.shape[1] >= 5:
        sim_nofvqe = arr_ref[:, 4]
    else:
        sim_nofvqe = arr_ref[:, 2]

    n_runs = len(data_qc_list)
    n_points = len(distances)

    qc_energies_all = np.full((n_runs, n_points), np.nan)
    accepted_all = np.zeros((n_runs, n_points), dtype=bool)

    for irun, data_qc in enumerate(data_qc_list):

        arr_qc = np.loadtxt(data_qc)

        if not np.allclose(arr_qc[:, 0], distances):
            raise ValueError(
                f"Distances in {data_qc} do not match reference distances."
            )

        qc_energies_all[irun, :] = arr_qc[:, 2]

        accepted = _load_acceptance_mask(data_qc)

        if accepted is None:
            accepted = np.ones(n_points, dtype=bool)

        if len(accepted) != n_points:
            print(
                f"[WARNING] Acceptance mask size mismatch for {data_qc}. "
                "Assuming all accepted."
            )
            accepted = np.ones(n_points, dtype=bool)

        accepted_all[irun, :] = accepted

    # ==========================================================
    # Mean/std using postselected QC samples only
    # ==========================================================

    qc_mean = np.full(n_points, np.nan)
    qc_std = np.full(n_points, np.nan)
    qc_n_acc = np.zeros(n_points, dtype=int)
    qc_n_total = np.full(n_points, n_runs, dtype=int)

    for ip in range(n_points):

        vals = qc_energies_all[accepted_all[:, ip], ip]
        vals = vals[np.isfinite(vals)]

        qc_n_acc[ip] = len(vals)

        if len(vals) == 0:
            qc_mean[ip] = np.nan
            qc_std[ip] = np.nan
        elif len(vals) == 1:
            qc_mean[ip] = vals[0]
            qc_std[ip] = 0.0
        else:
            qc_mean[ip] = np.mean(vals)
            qc_std[ip] = np.std(vals, ddof=1)

    acceptance_rate = qc_n_acc / qc_n_total
    valid_mean = np.isfinite(qc_mean)

    # ==========================================================
    # Figure: two stacked windows
    # ==========================================================

    # fig, (ax_upper, ax_lower) = plt.subplots(
    #     2,
    #     1,
    #     sharex=True,
    #     gridspec_kw={'height_ratios': [1, 3]}
    # )
    
    fig, ax_lower = plt.subplots(figsize=(9, 5.5))
    
    # ==========================================================
    # Figure: HF energy
    # ==========================================================
    
    x_smooth = np.linspace(distances.min(), distances.max(), 300)
    spline_hf = make_interp_spline(distances, hf_energy, k=3)
    y_smooth_hf = spline_hf(x_smooth)
    # --- Plotting ---
    ax_lower.scatter(
        distances,
        hf_energy,
        marker="s",
        color="C2",  # green
        zorder=5,
        s=36,        # Use 's' to control marker size if needed
        label="",  # Keep empty
    )
    ax_lower.plot(
        x_smooth,
        y_smooth_hf,
        linestyle="--",
        zorder=5,
        linewidth=2.5,  # FIXED: singular 'linewidth' for lines
        color="C2",  # green
        label="",  # Keep empty to prevent duplicates
    )

    # --- Custom Legend Handle ---
    hf_handle = mlines.Line2D(
        [],
        [],
        color="C2",
        linestyle="--",
        linewidth=2.5,  # ADDED: matches your actual plot line thickness
        marker="s",
        markersize=6,  # Adjust size to match your scatter plots
        label=r"HF",
    )

    # ==========================================================
    # Upper panel: discarded high-energy sector only
    # ==========================================================

    discarded_label_used = False

    for irun in range(n_runs):

        discarded = ~accepted_all[irun]

        # if np.any(discarded):
        #     ax_upper.scatter(
        #         distances[discarded],
        #         qc_energies_all[irun, discarded],
        #         s=42,
        #         facecolors="none",
        #         edgecolors="gray",
        #         linewidths=1.2,
        #         alpha=0.95,
        #         marker="o",
        #         label=None,
        #         zorder=3,
        #     )

    # ax_upper.set_ylim(*upper_ylim)
    # ax_upper.tick_params(
    #     direction="in",
    #     top=True,
    #     right=True,
    #     bottom=False,
    #     labelbottom=False,
    # )
    # ax_upper.set_ylabel("")  # no y-label on upper panel
    # ax_upper.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # ==========================================================
    # Lower panel: physical PEC region
    # ==========================================================

    spline = make_interp_spline(distances, sim_nofvqe, k=3)
    y_smooth = spline(x_smooth)
    # --- Plotting ---
    ax_lower.scatter(
        distances,
        sim_nofvqe,
        marker="s",
        color="C0",  # blue
        zorder=5,
        s=36,        # Use 's' to control marker size if needed
        label="",  # Keep empty
    )
    ax_lower.plot(
        x_smooth,
        y_smooth,
        linestyle="--",
        zorder=5,
        linewidth=2.5,  # FIXED: singular 'linewidth' for lines
        color="C0",  # blue
        label="",  # Keep empty to prevent duplicates
    )

    # --- Custom Legend Handle ---
    nofvqe_handle = mlines.Line2D(
        [],
        [],
        color="C0",
        linestyle="--",
        linewidth=2.5,  # ADDED: matches your actual plot line thickness
        marker="s",
        markersize=6,  # Adjust size to match your scatter plots
        label=r"Noiseless ON-NOF-VQE",
    )

    # # Pass the custom handle to the legend
    # ax_lower.legend(handles=[nofvqe_handle])

    accepted_label_used = False
    discarded_label_used = False

    for irun in range(n_runs):

        accepted = accepted_all[irun]
        discarded = ~accepted

        # Postselected QC samples
        ax_lower.scatter(
            distances[accepted],
            qc_energies_all[irun, accepted],
            s=60,
            alpha=0.24,
            label=(
                r"Postselected QC samples"
                if not accepted_label_used
                else None
            ),
            zorder=5,
        )
        accepted_label_used = True

        # Discarded samples that fall in lower energy window
        if np.any(discarded):
            ax_lower.scatter(
                distances[discarded],
                qc_energies_all[irun, discarded],
                s=60,
                facecolors="none",
                edgecolors="gray",
                linewidths=2,
                alpha=0.95,
                marker="o",
                label=(
                    r"Discarded QC samples"
                    if not discarded_label_used
                    else None
                ),
                zorder=3,
            )
            discarded_label_used = True

    ax_lower.errorbar(
        distances[valid_mean],
        qc_mean[valid_mean],
        yerr=qc_std[valid_mean],
        fmt="o",
        markersize=8,
        capsize=3,
        linewidth=2.5,
        color ="C1", # orange
        label=r"Postselected QC mean",
        zorder=6,
    )

    ax_lower.set_ylim(*lower_ylim)

    ax_lower.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax_lower.set_ylabel(r"Energy (Ha)")
    
    # Fetch all automatically generated handles and labels from scatter/errorbar
    existing_handles, existing_labels = ax_lower.get_legend_handles_labels()

    # Combine your custom handle with the existing ones
    all_handles = [hf_handle] + [nofvqe_handle] + existing_handles
    all_labels = [hf_handle.get_label()] + [nofvqe_handle.get_label()] + existing_labels

    # Create a single legend containing everything
    ax_lower.legend(
        handles=all_handles,
        labels=all_labels,
        frameon=True,
        loc="best"
    )
    
    ax_lower.tick_params(direction="in", top=False, right=True)

    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.12,
        top=0.98,
        hspace=0.07,
    )

    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.png", dpi=300)

    # ==========================================================
    # Save averaged data
    # ==========================================================

    out = np.column_stack([
        distances,
        sim_nofvqe,
        qc_mean,
        qc_std,
        qc_n_acc,
        qc_n_total,
        acceptance_rate,
    ])

    np.savetxt(
        f"{save_prefix}_accepted_mean_std.dat",
        out,
        header=(
            "distance_ang  Sim_NOF_energy_Ha  "
            "QC_mean_postselected_Ha  QC_std_postselected_Ha  "
            "n_postselected  n_total  postselection_rate"
        ),
        fmt="%.10f",
    )

    return distances, sim_nofvqe, qc_mean, qc_std, qc_n_acc, acceptance_rate
     

def plot_pes_filtered(records, save_prefix="pes_qc_filtered"):
    stats_all, _ = summarize_by_distance(records, filter_fn=None)
    stats_phys, _ = summarize_by_distance(records, filter_fn=physics_filter)
    stats_strict, _ = summarize_by_distance(records, filter_fn=strict_reference_filter)

    d = stats_all[:, 0]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(
        d,
        stats_all[:, 5],
        linestyle="--",
        marker="s",
        markersize=5,
        label=r"Simulator NOF-VQE",
    )

    ax.errorbar(
        stats_all[:, 0],
        stats_all[:, 3],
        yerr=stats_all[:, 4],
        fmt="o",
        markersize=5,
        capsize=3,
        alpha=0.45,
        label=r"QC all",
    )

    ax.errorbar(
        stats_phys[:, 0],
        stats_phys[:, 3],
        yerr=stats_phys[:, 4],
        fmt="o",
        markersize=5,
        capsize=3,
        label=r"QC physical filter",
    )

    ax.errorbar(
        stats_strict[:, 0],
        stats_strict[:, 3],
        yerr=stats_strict[:, 4],
        fmt="^",
        markersize=5,
        capsize=3,
        label=r"QC strict filter",
    )

    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Energy (Ha)")
    ax.legend(frameon=True)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.png", dpi=300)

    np.savetxt(
        f"{save_prefix}_all.dat",
        stats_all,
        header="distance n_total n_used E_qc_mean E_qc_std E_sim_mean Abs_Diff_E_mean Diff_n_mean",
        fmt="%.10f",
    )

    np.savetxt(
        f"{save_prefix}_physical_filter.dat",
        stats_phys,
        header="distance n_total n_used E_qc_mean E_qc_std E_sim_mean Abs_Diff_E_mean Diff_n_mean",
        fmt="%.10f",
    )

    np.savetxt(
        f"{save_prefix}_strict_filter.dat",
        stats_strict,
        header="distance n_total n_used E_qc_mean E_qc_std E_sim_mean Abs_Diff_E_mean Diff_n_mean",
        fmt="%.10f",
    )

def plot_acceptance_rate(records, save_prefix="acceptance_rate"):
    distances = sorted(set(r["distance"] for r in records))

    rows = []

    for d in distances:
        group = [r for r in records if r["distance"] == d]
        n_total = len(group)
        n_phys = sum(physics_filter(r) for r in group)
        n_strict = sum(strict_reference_filter(r) for r in group)

        rows.append([
            d,
            n_total,
            n_phys / n_total,
            n_strict / n_total,
        ])

    arr = np.array(rows)

    fig, ax = plt.subplots(figsize=(6, 4.0))

    ax.plot(arr[:, 0], arr[:, 2], marker="o", label="physical filter")
    ax.plot(arr[:, 0], arr[:, 3], marker="s", label="strict filter")

    ax.set_xlabel(r"Distance ($\mathrm{\AA}$)")
    ax.set_ylabel("Accepted fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True)
    ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.png", dpi=300)

    np.savetxt(
        f"{save_prefix}.dat",
        arr,
        header="distance n_total physical_acceptance strict_acceptance",
        fmt="%.10f",
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
if __name__ == "__main__":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    if len(sys.argv) < 2:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        print("Usage: python pes_scan_pynof_nofvqe.py <base_dir>")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        sys.exit(1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    base_dir = sys.argv[1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    option = sys.argv[2]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    if option == "pes":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        compute_pes(base_dir=base_dir, save_prefix=f"{base_dir}_pes", device="hybrid_real")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # elif option == "plot":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    #     qc_run = os.listdir(os.getcwd())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    #     plot_pes_sim_qc_stats(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    #         qc_run,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    #         save_prefix="pes_H8_scan_hybrid_real_mean_std",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    #     )                        
    elif option == "analysis":
        base_path = Path(base_dir)

        qc_runs = [
            str(base_path / d)
            for d in os.listdir(base_path)
            if (base_path / d).is_dir()
            and d.startswith("t_")
            and "cube_h2_nofvqe" in d
        ]
        
        records = collect_qc_records(qc_runs, device="hybrid_real")

        save_analysis_tables_by_run(
            records,
            base_path,
        )
        
        plot_pes_filtered(
            records,
            save_prefix=str(base_path / "pes_H8_real_filtered"),
        )

        plot_acceptance_rate(
            records,
            save_prefix=str(base_path / "pes_H8_real_acceptance"),
        )     
    elif option == "plot":

        base_path = Path(base_dir)

        qc_data = sorted([
            str(base_path / f)
            for f in os.listdir(base_path)
            if f.startswith("t_")
            and f.endswith("_pes.dat")
        ])

        plot_pes_sim_qc_stats(
            qc_data,
            save_prefix=str(base_path / "pes_H8_scan_hybrid_real_mean_std"),
        )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    # compute_pes(base_dir="cube_h4_pynof", save_prefix="cube_h4_pynof_pes", guess=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    # compute_pes(base_dir="cube_h4_pynof_HF_energy", save_prefix="cube_h4_pynof_HF_energy_pes", hf_only=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    # compute_pes(base_dir="cube_h2_nofvqe", save_prefix="cube_h2_nofvqe_pes")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    # compute_pes(base_dir=base_dir, save_prefix=f"{base_dir}_pes", device="hybrid_real")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    #pynof_data = "cube_h2_pynof_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    #nofvqe_data = "cube_h2_nofvqe_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    #plot_pes(pynof_data,nofvqe_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    #pynof_data = "cube_h4_pynof_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    # nofvqe_data = "cube_h4_nofvqe_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    # nofvqe_data = "cube_h2_nofvqe_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    # hybrid_nofvqe_data = "t_0_hybrid_cube_h2_nofvqe_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    # hf_data = "cube_h4_pynof_HF_energy_pes.dat"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    # plot_pes(pynof_data,nofvqe_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    # plot_pes_hf(pynof_data,hf_data)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    #plot_pes_sim_qc(nofvqe_data,hybrid_nofvqe_data)                                                                                                  