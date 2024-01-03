import glob
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

atoms_list = {
    'H': 1.,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
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

    # get distances matrix
    i = 110 # The index or symbol of atom from where you want to measure neighbors.
    print(i, atoms[i])
    j = get_neighbors(atoms, i, cutoff)
    _R = atoms.get_all_distances(mic=True)
    jj = np.concatenate([[i], j])
    m = jj.shape[0]
    R = _R[jj][:, jj]  # <- distances matrix

    print('number of neighbors', m)
    rnn = atoms.get_distances(i, j)  ##  rnn defines the distance bw atom i & j

    C = np.zeros(shape=(m, m))
    for k in range(m):
        i1 = jj[k]
        a1 = atoms[i1].symbol
        c1 = charges[a1]
        for l in range(m):
            i2 = jj[l]
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
        return np.sort(ud)[-n:], rnn
    else:
        # eigvals
        eig = np.linalg.eig(C)
        features = np.sort(eig[0])[-n:]
        return features, rnn


def make_all():
    paths = Path("Path of the file or a folder").rglob('*CON*') #To read one or all files in a folder.
    data_r = []
    for p in paths:
        print(p)
        c, r = get_features(p, 10, use_upper=True)
        data_r.append(sorted(r)[:5])  ##  distances for 5 nearest neighbours
    data_r = np.stack(data_r)
    np.savetxt("distance.csv", np.c_[ data_r], delimiter=",")


make_all()
