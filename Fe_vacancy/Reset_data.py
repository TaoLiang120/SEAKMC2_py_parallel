from mylammps.inputs.data import lmpData
import numpy as np
a = 2.855312
fname = "SIA_Tet.dat"
atom_style = "atomic"
data = lmpData.from_file(fname, atom_style=atom_style)
data.reset_atom_ids()
data.scale_data(1/3.0 )
data.to_file(fname)

