import numpy as np

class TrussElement:
    def __init__(self, elnodes=(0,0), section_id=0, cargas=(0, 0, 0, 0, 0, 0), section_dict=None, nodes=None, normal =(0)):
        self.elnodes = np.array(elnodes, dtype=int)
        self.section_id = section_id
        self.cargas = np.array(cargas, dtype=float)
        self.section_dict = section_dict
        self.nodes = nodes
        self.normal = normal
        self.length, self.cos, self.sin = self._calculate_geometry()

    def _calculate_geometry(self):
        id0 = self.elnodes[0]
        id1 = self.elnodes[1]
        node0 = self.nodes[id0]
        node1 = self.nodes[id1]
        dx = node1.coord[0] - node0.coord[0]
        dy = node1.coord[1] - node0.coord[1]
        length = np.sqrt(dx**2 + dy**2)
        cos = dx / length
        sin = dy / length
        return length, cos, sin


    def local_stiffness_matrix(self):
        section = self.section_dict[self.section_id]
        E, A = section.E, section.A
        k = E * A / self.length
        c, s = self.cos, self.sin
        K_local = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        
        #if c not in (0,1):
        #    print(f'T =  {self.RotationMatrix}:\n\n{K_local}\n\n')
        #    return self.RotationMatrix().T @ K_local @ self.RotationMatrix(), self.cargas 
        #else: 
        #    print(f'T = {self.RotationMatrix}:\n\n{K_local}\n\n')
        return  K_local  , self.cargas

    def global_dof_indices(self):
        nodes = self.nodes
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], j[0], j[1]]

    def element_forces(self, displacement):
        eqs = self.global_dof_indices()
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        return elstiff @ disploc, elforce

    def reaction_forces(self, displacement):
        reaction, elforce = self.element_forces(displacement)
        return reaction - elforce

    def RotationMatrix(self):
        c, s = self.cos, self.sin
        T= np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c]
        ])
        return T
    
    def axial_reaction_forces(self, displacement):
        reaction = self.reaction_forces(displacement)
        return self.RotationMatrix() @ reaction

    

    

    def local_geometric_stiffness_matrix(self):
        """
        x1, y1: coordenadas do nó inicial
        x2, y2: coordenadas do nó final
        axial_force: esforço normal na barra (positivo tração, negativo compressão)
        """
        # Comprimento e cosseno/seno
        L = self.length
        c = self.cos
        s = self.sin

        # Fator de multiplicação
        factor = self.normal / L

        # Matriz de estabilidade geométrica local (4x4)
        K_geo = factor * np.array([
            [ c**2,  c*s,   -c**2, -c*s ],
            [ c*s,   s**2,  -c*s,  -s**2],
            [-c**2, -c*s,   c**2,  c*s  ],
            [-c*s, -s**2,   c*s,   s**2 ]
        ])

        return K_geo
  
        