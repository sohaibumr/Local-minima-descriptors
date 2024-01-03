from dscribe.descriptors import EwaldSumMatrix
from ase.io import read
from pathlib import Path
import numpy as np



species = [ 
    'H', 'O', "B", "C", "N",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo",
     "Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au"
            ]

rcut = 6
gcut = 6

# Calculate Ewald sum matrix with DScribe (https://singroup.github.io/dscribe/latest/tutorials/descriptors/ewald_sum_matrix.html)

ems = EwaldSumMatrix(
    n_atoms_max=112, #maximum no of atoms in the system
    permutation="none",
    flatten=False
)

paths = Path("Path of the file or a folder").rglob('*CON*')  # To read one or all files in a folder.
def Ewaldmatrix():
    data = []
    for pt in paths:
        atom = read(pt)
        s = ems.create(atom)
        ss = np.array(s.sum(axis=1))
        p = ss.reshape(1, len(ss))
        f = list(np.squeeze(p))
        data.append(f)
        np.stack(data)
        np.sort(data)
        np.savetxt("Ewaldmatrix_M3N.csv", data, fmt='%4f', delimiter=",")

Ewaldmatrix()
