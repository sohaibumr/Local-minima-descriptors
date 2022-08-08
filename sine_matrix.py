from dscribe.descriptors import SineMatrix
from ase.io import read
from pathlib import Path
import numpy as np



# Setting up the Dscribe SineMatrix descriptor (https://singroup.github.io/dscribe/latest/tutorials/descriptors/sine_matrix.html#)

sm = SineMatrix(
    n_atoms_max=112, #maximum no of atoms in the system
    permutation="none",
    sparse=False,
    flatten=False)



paths = Path("Path of the file or a folder").rglob('*CON*') # To read one or all files in a folder.
def sine_matrix():
    data = []
    for pt in paths:
        atom = read(pt)
        print(pt)
        s = sm.create(atom)
        ss = np.array(s.sum(axis=1))
        p = ss.reshape(1, len(ss))
        f = list(np.squeeze(p))
        data.append(f)
        np.stack(data)
        np.sort(data)
        np.savetxt("sin_matrix_O.csv", data, fmt='%5f', delimiter=",")


sine_matrix()