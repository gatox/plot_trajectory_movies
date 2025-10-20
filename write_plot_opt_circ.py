import time
import os
import csv
import numpy as np
from nofvqe import NOFVQE

# ===============================
# User Parameters
# ===============================
xyz_file = "h2_bohr.xyz"
functional = "PNOF4"
conv_tol = 1e-2
init_param = 0.1
basis = "sto-3g"
max_iterations = 500
gradient = "analytics"
d_shift = 1e-4
C_MO = "guest_C_MO"
dev = "simulator"
n_shots = 10000
optimization_level = 3
resilience_level = 0

optimizers = ["adam", "sgd", "slsqp", "l-bfgs-b", "cobyla", "spsa", "cmaes"]

# Folder for CSV results
os.makedirs("results", exist_ok=True)

# ===============================
# Run optimizations
# ===============================
summary = {}

for opt_method in optimizers:
    print(f"\n=== Testing {opt_method.upper()} ===")

    cal = NOFVQE(
        xyz_file,
        functional=functional,
        conv_tol=conv_tol,
        init_param=init_param,
        basis=basis,
        max_iterations=max_iterations,
        opt_circ=opt_method,
        gradient=gradient,
        d_shift=d_shift,
        C_MO=C_MO,
        dev=dev,
        n_shots=n_shots,
        optimization_level=optimization_level,
        resilience_level=resilience_level,
    )

    crds = cal.crd
    start_time = time.time()
    E_hist, params_hist, *_ = cal._vqe(cal.ene_pnof4, init_param, crds, method=opt_method)
    runtime = time.time() - start_time

    # Store summary info
    summary[opt_method] = {
        "final_energy": E_hist[-1],
        "iterations": len(E_hist),
        "time_sec": runtime,
    }

    # ===============================
    # Save results per optimizer
    # ===============================
    csv_filename = f"results/{opt_method}_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header: iteration, energy, params...
        header = ["iteration", "energy", "param"]
        if isinstance(params_hist[0], (list, np.ndarray)):
            header += [f"param_{i}" for i in range(len(np.ravel(params_hist[0])))]
        writer.writerow(header)

        # Write each iteration
        for i, (E, p) in enumerate(zip(E_hist, params_hist)):
            row = [i, E] + list(np.ravel(p))
            writer.writerow(row)

    print(f"Saved results to {csv_filename}")

    # Remove cache file between runs
    if os.path.exists("pynof_C.npy"):
        os.remove("pynof_C.npy")

# ===============================
# Print Summary
# ===============================
print("\n=== Summary ===")
for k, v in summary.items():
    print(f"{k.upper():10s}: E = {v['final_energy']:.6f}, "
          f"iters = {v['iterations']}, time = {v['time_sec']:.2f}s")

# Save summary as CSV
with open("results/summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["optimizer", "final_energy", "iterations", "time_sec"])
    for k, v in summary.items():
        writer.writerow([k, v["final_energy"], v["iterations"], v["time_sec"]])
