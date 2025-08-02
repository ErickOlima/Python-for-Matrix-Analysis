import numpy as np

class BeamElement:
    def __init__(self, elnodes=(0, 0), section_id=0, cargas=(0, 0, 0, 0, 0, 0), section_dict=None, nodes= None):
        self.elnodes = np.array(elnodes, dtype=int)
        self.section_id = section_id
        self.cargas = np.array(cargas, dtype=float)
        self.section_dict = section_dict
        self.nodes = nodes
        #self.normal=normal
        self.length, self.cos, self.sin = self.calculate_geometry()

    def calculate_geometry(self):
        node0 = self.nodes[self.elnodes[0]]
        node1 = self.nodes[self.elnodes[1]]
        dx = node1.coord[0] - node0.coord[0]
        dy = node1.coord[1] - node0.coord[1]
        self.length = np.sqrt(dx**2 + dy**2)
        self.cos = dx / self.length
        self.sin = dy / self.length
        
        return self.length, self.cos, self.sin

    def axial_stiffness_matrix(self):
        section = self.section_dict[self.section_id]
                    
        E = section.E
        A = section.A
        I = section.I
        L = self.length
        k = E * A / L
        k_bending = E * I / (L**3)

        K_local = np.array([
            [k, 0, 0, -k, 0, 0],
            [0, 12*k_bending, 6*k_bending*L, 0, -12*k_bending, 6*k_bending*L],
            [0, 6*k_bending*L, 4*k_bending*L**2, 0, -6*k_bending*L, 2*k_bending*L**2],
            [-k, 0, 0, k, 0, 0],
            [0, -12*k_bending, -6*k_bending*L, 0, 12*k_bending, -6*k_bending*L],
            [0, 6*k_bending*L, 2*k_bending*L**2, 0, -6*k_bending*L, 4*k_bending*L**2]
        ])
        return K_local

    def rotation_matrix(self):
        c = self.cos
        s = self.sin
        R = np.zeros((6, 6))
        R[0:2, 0:2] = [[c, s], [-s, c]]
        R[3:5, 3:5] = [[c, s], [-s, c]]
        R[2, 2] = 1
        R[5, 5] = 1
        return R

    def local_stiffness_matrix(self):
        K_local = self.axial_stiffness_matrix()
        R = self.rotation_matrix()
        K_global = R.T @ K_local @ R
        f_global = self.cargas  # if already in global coords
        return K_global, f_global

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], i[2], j[0], j[1], j[2]]

    def element_forces(self, displacement, nodes):
        dofs = self.global_dof_indices(nodes)
        u_elem = displacement[dofs]
        K, f = self.local_stiffness_matrix()
        return K @ u_elem, f

    def reaction_forces(self, displacement, nodes):
        r, f = self.element_forces(displacement, nodes)
        return r - f

    def axial_reaction_forces(self, displacement, nodes):
        r = self.reaction_forces(displacement, nodes)
        R = self.rotation_matrix()
        return R @ r

    def axial_geometric_stiffness_matrix(self):
        
        l= self.length
        k_geo = np.array([
            [ 0, 0, 0, 0, 0, 0],
            [ 0, 6./(5.*l), 1./10., 0, -6./(5.*l), 1./10.],
            [ 0, 1./10., 2.*l/15., 0, -1./10., -1./30.],
            [0, 0, 0, 0, 0, 0],
            [ 0, -6./(5.*l), -1./10., 0, 6./(5.*l), -1./10.],
            [ 0, 1./10., -l/30., 0, -1/10., 2.*l/15.]
        ] ) * self.normal
        return k_geo 
    
    def local_geometric_stiffness_matrix(self):
        k_geo = self.axial_geometric_stiffness_matrix()
        rotation = self.rotation_matrix()
        k_rotated = rotation.T @ k_geo @ rotation
        
        return k_rotated