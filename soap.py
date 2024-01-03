from dscribe.descriptors import SOAP
from ase.io import read
from pathlib import Path
import numpy as np


species = [
    H', 'O', "B", "C", "N",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo",
     "Tc","Ru","Rh","Pd","Ag","Cd","Hf","Ta","W","Re","Os","Ir","Pt","Au"
]
rcut = 6
nmax = 2
lmax = 2

# Setting up the SOAP descriptor (https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)
soap = SOAP(
    species=species,
    periodic=True,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax
)


paths = Path("Path of the file or a folder").rglob('*CON*')
def Soap():
    data = []
    for pt in paths:
        print(pt)
        atom = read(pt)
        s = soap.create(atom, positions=["indices of atoms for example = 109, 110, 111"])  #for a specified position, if position is not specified it will calculate for every atom)
        ss = np.array(s.sum(axis=1))
        p  =  ss.reshape(1, len(ss))
        f = list(np.squeeze(p))
        data.append(f)
        np.stack(data)
        np.sort(data)
        np.savetxt("soap_data.csv", data, fmt='%5f',
               delimiter=",")

Soap()


