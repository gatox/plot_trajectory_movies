#Script for checking missing computed trajectories
#And submitting the missed trajectories 
import os
from subprocess import run, CalledProcessError

def all_traj():
    total_traj = []
    for i in range(200):
        total_traj.append(int(i))
    return total_traj

def read_files():
    add_traj = []
    file_types = ["cancelled_time_limit_iterations", "no_conv_scf_100_iterations", "trajectories_ok_list"]
    for i in file_types:
        file = open(i,'r+')
        for line in file:
            numb = line.split('./prop/traj_')[1]
            add_traj.append(int(numb))
    return sorted(add_traj)

def compare():
    comp = read_files()
    total_traj = all_traj()
    no_sub = no_submitted_traj()
    with open(f'trajectories_submitted.out', 'w') as f:
        f.write('--------------------------------------------------------\n')
        f.write(f'Information of Trajectories:\n')
        f.write(f'The number of trajectories submitted is {len(total_traj)}:\n')
        f.write(f'{total_traj}\n')
        f.write(f'The number of trajectories executed is {len(comp)}:\n')
        f.write(f'{comp}\n')
        f.write(f'The number of trajectories running is {len((set(total_traj)-set(comp))-set(no_sub))}:\n')
        f.write(f'{sorted(set(total_traj)-set(comp)-set(no_sub))}\n')
        f.write(f' The number of trajectories cancelled is {len(no_sub)}:\n')
        f.write(f'{sorted(set(no_sub))}\n')
        f.write('--------------------------------------------------------')
        f.close()
    return sorted(set(comp))

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

# Example usage
compare()
#submit_traj_missed()
