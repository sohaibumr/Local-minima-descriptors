from ase.io import read
from ase.neighborlist import NeighborList
import numpy as np
from pathlib import Path



TM =["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo",
     "Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au"] # List of atoms in your system or systems

def get_neighbors(atoms, i, cutoff):
    natoms = atoms.get_global_number_of_atoms()
    nl = NeighborList(natoms * [cutoff / 2], skin=0.0,
                      self_interaction=False,
                      bothways=True)
    nl.update(atoms)
    j = nl.get_neighbors(i)[0]
    return j


def get_angles(atoms):
    atoms = read(atoms)
    index_1 = [a.index for a in atoms if a.number == 1][0] # This statement is for getting H index, Instead you can give an integer or symbol of the atom.
    index_2 = [a.index for a in atoms if a.symbol in TM][0] # This statement is for getting transistion metal index present in the given list of TM, Instead you can give an integer or symbol of the atom.
    nei_TM = get_neighbors(atoms, index_1, 6) # No of neighbors = 6
    nei_TM_filtered = np.array([n for n in nei_TM if atoms[n].symbol in  ['B', 'C']])
    distances = atoms.get_distances(index_2, nei_TM_filtered, mic=True)
    indices_nn = nei_TM_filtered[np.argsort(distances)][:6]
    print(indices_nn)
    angles = np.array([atoms.get_angle(index_1, index_2, index_3) for index_3 in indices_nn])
    return angles


def make_all():
    path = Path("Path of the file or a folder").rglob('*CON*')  # To read one or all files in a folder.
    data_angles = []
    for p in path:
        data_angles.append(get_angles(p))

    data_angles = np.stack(data_angles)
    np.sort(data_angles)
    np.savetxt("angles_H.csv", np.c_[data_angles], delimiter=",")


make_all()

# It will return angles between index_1, index_2, and index_3
