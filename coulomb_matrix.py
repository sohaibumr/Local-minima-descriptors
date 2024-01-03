from ase.io import read
from ase.neighborlist import NeighborList
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_neighbors(atoms, i, cutoff):
    natoms = atoms.get_global_number_of_atoms()
    nl = NeighborList(natoms * [cutoff / 2], skin=0.0,
                      self_interaction=False,
                      bothways=True)
    nl.update(atoms)
    j = nl.get_neighbors(i)[0]
    return j

atoms_list =   {'H': 1.,
    "B": 5,
    "C": 6,
    "N": 7,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79
}


def get_features(path, n=5, cutoff=4, use_upper=True, charges=atoms_list):
    
    # n defines the no of neighbors, default is n=5.
    
    atoms = read(path)
    base_atom = 110 # The index or symbol of atom from where you want to measure neighbors.
    print(base_atom, atoms[base_atom])
    neighbor = get_neighbors(atoms, base_atom, cutoff)
    distances = atoms.get_all_distances(mic=True)
    con_base_neighbor = np.concatenate([[base_atom], neighbor])
    m = con_base_neighbor.shape[0]
    R = distances[con_base_neighbor][:, con_base_neighbor]

    print('number of neighbors', m)
    distance = atoms.get_distances(base_atom, neighbor)


    C = np.zeros(shape=(m, m))
    for k in range(m):
        i1 = con_base_neighbor[k]
        a1 = atoms[i1].symbol
        c1 = charges[a1]
        for l in range(m):
            i2 = con_base_neighbor[l]
            a2 = atoms[i2].symbol
            c2 = charges[a2]
            if k != l:
                C[k, l] = c1 * c2 / R[k, l]
            else:
                C[k, l] = 0.5 * (c1 * c2) ** 2.4

    if use_upper:
        # upper diagonal
        upper_diag = []
        for k in range(m - 1):
            for l in range(k + 1, m):
                upper_diag.append(C[k, l])
        ud = np.stack(upper_diag).reshape(-1)
        return np.sort(ud)[-n:], distance
    else:
        # eigvals
        eig = np.linalg.eig(C)
        features = np.sort(eig[0])[-n:]
        return features, distance


def make_all():
    paths = Path("Path of the file or a folder").rglob('*CON*') #To read one or all files in a folder.
    data = []
    for p in paths:
        c, r = get_features(p, 10, use_upper=True) #here no of neighbors is n=10
        data.append(c)
    data = np.stack(data)
    np.sort(data)
    plt.imshow(data)
    np.savetxt("columb_matrix.csv", np.c_[data], delimiter=",")

make_all()
