from dscribe.descriptors import ACSF
from ase.io import read
from pathlib import Path
import numpy as np


# List of atom symbols included in your list
species = [
    'H',
    'O',
    "B",
    "C",
    "N",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au"
]


# Setting up the DScribe ACSF descriptor (https://singroup.github.io/dscribe/latest/tutorials/descriptors/acsf.html)

acsf = ACSF(
    species=species,
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)


paths = Path("Path of the file or a folder").rglob('*CON*') #To read one or all files in a folder.
def Acsf():
    data = []
    for pt in paths:
        atom = read(pt)
        s = acsf.create(atom)
        p = np.array(s.sum(axis=1)).reshape(1, len(atom))
        f = list(np.squeeze(p))
        print(f)
        data.append(f)
        np.stack(data)
        np.sort(data)
        np.savetxt("ACSF_M3N.csv", data, fmt='%4f',
               delimiter=",")
Acsf()


