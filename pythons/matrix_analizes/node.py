import numpy as np

class Node:
    def __init__(self, coord=(0,0), eqs=(0,0), is_fixed=(False, False), force=(0, 0), presc_disp=(0, 0)):
        self.eqs = np.array(eqs, dtype=int)
        self.coord = np.array(coord, dtype=float)
        self.is_fixed = is_fixed
        self.force = np.array(force, dtype=float)
        self.presc_disp = np.array(presc_disp, dtype=float)
