#Script for checking missing computed trajectories
#And submitting the missed trajectories 
import os
import numpy as np
from subprocess import run, CalledProcessError
from pysurf.database import PySurfDB


def read_files():
    add_traj = []
    file_types = ["cancelled_time_limit_iterations", "no_conv_scf_100_iterations", "trajectories_ok_list"]
    for i in file_types:
        file = open(i,'r+')
        for line in file:
            numb = line.split('./prop/traj_')[1]
            add_traj.append(int(numb))
    return sorted(add_traj)

def all_traj():
    return [int(i) for i in range(200)]

def ok_traj():
    with open("trajectories_ok_list", 'r') as read:
        ok_traj = sorted(int(line.split('./prop/traj_')[1]) for line in read)
    return ok_traj 

def pote_traj():
    ok_traj_old = ok_traj() 
    total_traj = all_traj()
    return [f"traj_{i:08d}" for i in sorted(set(total_traj) - set(ok_traj_old))]

def md_steps():
    with open("prop.inp", 'r') as prop:
        line = next((line for line in prop if "mdsteps" in line), None)
    return int(np.ceil(int(line.split()[2]) / 2)) if line else None

def new_traj_ok():
    mdsteps = md_steps()
    valid_trajs = pote_traj()  # Get the valid trajectory list as a set for fast lookup
    return [
        int(traj.split('traj_')[1]) for traj in os.listdir("prop")
        if traj in valid_trajs  # Only consider trajectories from read_ok_traj()
        and (subfolder := os.path.join("prop", traj)) and os.path.isdir(subfolder)  # Ensure it's a directory
        and (db := PySurfDB.load_database(os.path.join(subfolder, "results.db"), read_only=True))  # Load DB
        and len(db["currstate"]) >= mdsteps  # Check condition
    ]

def update_ok_traj_list():
    ok_traj_old = ok_traj()
    new_traj = new_traj_ok()
    added_traj = sorted(ok_traj_old+new_traj)
    with open("added_trajectories_ok_list", 'w') as f:
        for i in added_traj:
            f.write(f"./prop/traj_{i:08d}\n")
        f.close()

def compare():
    comp = read_files()
    total_traj = all_traj()
    no_sub = no_submitted_traj()
    with open(f'trajectories_submitted.out', 'w') as f:
        f.write('--------------------------------------------------------\n')
        f.write(f'Information of Trajectories:\n')
        f.write(f'The number of trajectories submitted: {len(total_traj)}\n')
        f.write(f'{total_traj}\n')
        f.write(f'The number of trajectories executed: {len(comp)}\n')
        f.write(f'{comp}\n')
        f.write(f'The number of trajectories running: {len((set(total_traj)-set(comp))-set(no_sub))}\n')
        f.write(f'{sorted(set(total_traj)-set(comp)-set(no_sub))}\n')
        f.write(f' The number of trajectories cancelled: {len(no_sub)}\n')
        f.write(f'{sorted(set(no_sub))}\n')
        f.write('--------------------------------------------------------')
        f.close()
    return sorted(set(no_sub))

def missing_traj():
    read = compare()
    return [f"traj_{i:08d}" for i in read]

def no_submitted_traj():
    no_sub = []
    for traj in os.listdir("prop"):
        subfolder = os.path.join("prop",traj)
        if os.path.isfile(os.path.join(subfolder,"saoovqe.in.out")):
            continue
        else:
            numb = int(traj.split('traj_')[1])
            no_sub.append(int(numb))
    return no_sub 
            
def submit_traj_missed():
    allowed = missing_traj()
    for traj in os.listdir("prop"):
        if traj in allowed: 
            subfolder = os.path.join("prop",traj)
            try:
                run(['sbatch cpu_long_saoovqe.sh'], cwd=subfolder, check=True, shell=True)
            except KeyboardInterrupt or CalledProcessError:
                break
            print("Submitting", subfolder)

if __name__=="__main__":
    #request = input("available features: traj_missed, compare_inf, add_ok_list\n")
    #request = 'add_ok_list' 
    update_ok_traj_list()
    #if request == 'traj_missed':
    #    submit_traj_missed()
    #elif request == 'compare_inf':
    #    compare()
    #elif request == 'add_ok_list':
    #    update_ok_traj_list()
    #else:
    #    print("wrong request")
