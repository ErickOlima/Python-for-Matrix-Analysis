import numpy as np


class SpringElement:
    def __init__(self, elnodes=(0, 0), spring_value=(0, 0, 0), nodes=None, cargas = (0, 0, 0, 0, 0, 0), normal = (1.0)):
        self.elnodes = np.array(elnodes, dtype=int)
        self.values = np.array(spring_value, dtype=float)
        self.nodes = nodes
        self.normal = normal
        self.cargas = np.array(cargas, dtype=float)
        
    
    
    def local_stiffness_matrix(self):
        s1, s2, s3 = self.values
       
        k = np.array([
            [ s1,  0,   0, -s1,  0,   0],
            [  0, s2,   0,   0, -s2,  0],
            [  0,  0,  s3,   0,   0, -s3],
            [-s1,  0,   0,  s1,  0,   0],
            [  0, -s2,  0,   0, s2,   0],
            [  0,  0, -s3,   0,  0,  s3]
        ])
        
   
        #cargas = np.zeros(6)
        return k, self.cargas

    def global_dof_indices(self):
        nodes = self.nodes
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], i[2], j[0], j[1], j[2]]

    def element_forces(self, displacement):
        nodes = self.nodes
        eqs = self.global_dof_indices()
        disploc = displacement[eqs]
        k_local, f_local = self.local_stiffness_matrix()
        return k_local @ disploc, f_local

    def reaction_forces(self, displacement, nodes):
        r, f = self.element_forces(displacement, nodes)
        return r - f

    def axial_reaction_forces(self, displacement, nodes):
        return self.reaction_forces(displacement, nodes)

    def local_geometric_stiffness_matrix(self):
        return np.zeros((6, 6))