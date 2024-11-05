#Script for checking missing computed trajectories

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
    no_comp = read_files()
    total_traj = all_traj()
    with open(f'Trajs_info.out', 'w') as f:
        f.write('--------------------------------------------------------\n')
        f.write(f'Information of Trajectories:\n')
        f.write(f'All trajectories with a len of {len(total_traj)}:\n')
        f.write(f'{total_traj}\n')
        f.write(f'All trajectories submitted an started with a len of {len(no_comp)}:\n')
        f.write(f'{no_comp}\n')
        f.write(f'All trajectories no submitted  with a len of {len(set(total_traj) - set(no_comp))}:\n')
        f.write(f'{sorted(set(total_traj) - set(no_comp))}\n')
        f.write('--------------------------------------------------------')
        f.close()
# Example usage
compare()
