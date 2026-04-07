"""
Load the molecule data
"""


def read_mol(path_file):
    with open(path_file, "r") as file:
        point = file.readline().strip()
        cluster = []
        while point:
            point = list(map(float, point.split()))
            cluster.append(point)
            point = file.readline().strip() 
    return cluster


def read_molecules_from_dir(dir):
    clusters = []
    cluster_num = 1
    import os
    while os.path.isfile(f"{dir}{cluster_num}"):
        clusters.append(read_mol(f"{dir}{cluster_num}"))
        cluster_num += 1
    return clusters


def center_mol(mol):
    import numpy as np
    m1 = np.array(mol)
    m1_center = m1 - np.mean(m1, axis=0)
    #m1_dists = np.linalg.norm(m1_center, axis=1)
    #max_atom_dist = m1_dists.max()
    #return m1_center / max_atom_dist
    return m1_center


if __name__=="__main__":

    minima = read_molecules_from_dir("molecules/mol.")
    print(len(minima))
    print(minima[0])

    import numpy as np
    m1_normed = center_mol(minima[0])

    print("------------------------------------------------------------")
    print("- center of gravity", np.mean(m1_normed, axis=0))
    print("- dist_from_origin", (np.linalg.norm(m1_normed, axis=1)))


    #saddles = read_mols_in_dir("./LJ7_ts/points.")
