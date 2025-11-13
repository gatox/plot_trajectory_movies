import time
import os
import csv
import numpy as np
from nofvqe import NOFVQE

class WritePlotOptCirc:

    def __init__(self, optimizer,conv_tol):
        self.xyz_file = "h2_bohr.xyz"
        self.functional = "PNOF4"
        self.optimizer = optimizer
        self.conv_tol = conv_tol
        self.init_param = 0.1
        self.basis = "sto-3g"
        self.max_iterations = 500
        self.gradient = "analytics"
        self.d_shift = 1e-4
        self.C_MO = "guest_C_MO"
        self.dev = "simulator"
        self.n_shots = 10000
        self.optimization_level = 3
        self.resilience_level = 0

    def write_csv(self):
        # ===============================
        # User Parameters
        # ===============================


        #optimizers = ["adam", "sgd", "slsqp", "l-bfgs-b", "cobyla", "spsa", "cmaes"]

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
                functional=self.functional,
                conv_tol=self.conv_tol,
                init_param=self.init_param,
                basis=self.basis,
                max_iterations=self.max_iterations,
                opt_circ=self.opt_method,
                gradient=self.gradient,
                d_shift=self.d_shift,
                C_MO=self.C_MO,
                dev=self.dev,
                n_shots=self.n_shots,
                optimization_level=self.optimization_level,
                resilience_level=self.resilience_level,
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

# # Save summary as CSV
# with open("results/summary.csv", "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["optimizer", "final_energy", "iterations", "time_sec"])
#     for k, v in summary.items():
#         writer.writerow([k, v["final_energy"], v["iterations"], v["time_sec"]])

if __name__ == "__main__":

