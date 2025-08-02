import numpy as np
import json

section_dict = {}
nodes = []
elements = []
rigid_body_modes = []
# Define classes for SectionProperties, Node, and Element

class SectionProperties:
    def __init__(self, id, E, A, I):
        self.id = id
        self.E = E  # Young's modulus
        self.A = A  # Cross-sectional area
        self.I = I  # Moment of inertia

    def __repr__(self):
        return f"SectionProperties(id={self.id}, E={self.E}, A={self.A})"

class Node:
    def __init__(self, coord=(0,0), eqs=(0,0), is_fixed=(False, False), force=(0, 0), presc_disp=(0, 0)):
        self.eqs = np.array(eqs, dtype=int)
        self.coord = np.array(coord, dtype=float)
        self.is_fixed = is_fixed  # (fix_x, fix_y)
        self.force = np.array(force, dtype=float)
        self.presc_disp = np.array(presc_disp, dtype=float)

class TrussElement:
    def __init__(self, elnodes=(0,0), section_id= 0,cargas=(0,0,0,0)):
        self.cargas = np.array(cargas, dtype=float)
        self.elnodes = np.array(elnodes, dtype=int)
        self.section_id = section_id
        self.length, self.cos, self.sin = self._calculate_geometry()

    def _calculate_geometry(self):
        id0 = self.elnodes[0]
        id1 = self.elnodes[1]
        node0 = nodes[id0]
        node1 = nodes[id1]
        dx = node1.coord[0] - node0.coord[0]
        dy = node1.coord[1] - node0.coord[1]
        length = np.sqrt(dx**2 + dy**2)
        cos = dx / length
        sin = dy / length
        return length, cos, sin

    def local_stiffness_matrix(self):
        c = self.cos
        s = self.sin
        section = section_dict[self.section_id]
        E = section.E
        A = section.A
        k = E * A / self.length
        K_local= k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ] )
        return K_local, self.cargas
        

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], j[0], j[1]]
    
    def reaction_forces(self, displacement):
        eqs = self.global_dof_indices(nodes)
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        # Calculate reaction forces
        reaction = elstiff @ disploc
        # Add local forces
        reaction -= elforce
        return reaction
    
    def RotationMatrix(self):
        c = self.cos
        s = self.sin
        # Rotation matrix for 2D truss element
        rotation_matrix = np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c]
        ])
        return rotation_matrix
    
    def axial_reaction_forces(self, displacement):
        reaction = self.reaction_forces(displacement)
        rotation_matrix = self.RotationMatrix()
        axial_force = rotation_matrix @ reaction
        return axial_force
    
class BeamElement:
    def __init__(self, elnodes=(0,0), section_id= 0,cargas=(0,0,0,0,0,0)):
        self.cargas = np.array(cargas, dtype=float)
        self.elnodes = np.array(elnodes, dtype=int)
        self.section_id = section_id
        self.length, self.cos, self.sin = self._calculate_geometry()

    def _calculate_geometry(self):
        id0 = self.elnodes[0]
        id1 = self.elnodes[1]
        node0 = nodes[id0]
        node1 = nodes[id1]
        dx = node1.coord[0] - node0.coord[0]
        dy = node1.coord[1] - node0.coord[1]
        length = np.sqrt(dx**2 + dy**2)
        cos = dx / length
        sin = dy / length
        return length, cos, sin
    # Local stiffness matrix for beam element
    # Assuming a 2D beam element with axial and bending stiffness
    # The local stiffness matrix is derived from the beam theory
    # and is a 6x6 matrix for axial and bending stiffness
    def axial_stiffness_matrix(self):
        section = section_dict[self.section_id]
        E = section.E
        A = section.A
        I = section.I
        k = E * A / self.length
        k_bending = E * I / (self.length**3)
        K_local= np.array([
            [ k, 0, 0, -k, 0, 0],
            [ 0, 12*k_bending, 6*k_bending*self.length, 0, -12*k_bending, 6*k_bending*self.length],
            [ 0, 6*k_bending*self.length, 4*k_bending*self.length**2, 0, -6*k_bending*self.length, 2*k_bending*self.length**2],
            [-k, 0, 0, k, 0, 0],
            [ 0, -12*k_bending, -6*k_bending*self.length, 0, 12*k_bending, -6*k_bending*self.length],
            [ 0, 6*k_bending*self.length, 2*k_bending*self.length**2, 0, -6*k_bending*self.length, 4*k_bending*self.length**2]
        ] )
        return K_local
    
    def rotation_matrix(self):
        # Rotating the axial stiffness matrix to global coordinates
        rotation = np.zeros((6, 6))
        c = self.cos
        s = self.sin
        rotation[0:2, 0:2] = np.array([[c, s], [-s, c]])
        rotation[3:5, 3:5] = np.array([[c, s], [-s, c]])
        rotation[2, 2] = 1
        rotation[5, 5] = 1
        return rotation

    def local_stiffness_matrix(self):
        c = self.cos
        s = self.sin
        K_axial = self.axial_stiffness_matrix()
        rotation = self.rotation_matrix()
        K_rotated = rotation.T @ K_axial @ rotation
        # Adding the local loads to the global loads
        # Rotating the local loads to global coordinates
        f_rotated = self.cargas
        
        return K_rotated, f_rotated
        

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], i[2], j[0], j[1], j[2]]
    
    def reaction_forces(self, displacement):
        eqs = self.global_dof_indices(nodes)
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        # Calculate reaction forces
        reaction = elstiff @ disploc
        # Add local forces
        reaction -= elforce
        return reaction
    
    def axial_reaction_forces(self,displacement):
        reaction = self.reaction_forces(displacement)
        rotation = self.rotation_matrix()
        axial_reaction = rotation @ reaction
        return axial_reaction


def assemble_global_stiffness(elements, nodes, total_dofs):
    F = np.zeros(total_dofs)
    K = np.zeros((total_dofs, total_dofs))
    for elem in elements:
        k_local,f_local = elem.local_stiffness_matrix()
        dof_indices = elem.global_dof_indices(nodes)
        neqs = len(dof_indices)
        for i in range(neqs):
            F[dof_indices[i]] += f_local[i]
            for j in range(neqs):
                K[dof_indices[i], dof_indices[j]] += k_local[i, j]
    return K, F

def apply_boundary_conditions(K, F, displacements, nfree):
    # Apply boundary conditions
    K_bc = K[:nfree,:nfree]
    F_bc = F[:nfree]- K[:nfree,nfree:] @ displacements[nfree:]
    # Apply prescribed displacements
    return K_bc, F_bc

def equation_count(nodes):
    free = 0
    restrained = 0
    for node in nodes:
        ndof = len(node.eqs)
        if ndof != 2 and ndof != 3:
            raise ValueError("Node must have 2 degrees of freedom")
        for i in range(ndof):
            if(node.is_fixed[i]) : restrained = restrained+1
            else : free = free+1
    return free, restrained
    
def solve_structure(nodes, elements):
    
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

    # Assemble global stiffness matrix
    K,F = assemble_global_stiffness(elements, nodes, total_dofs)


    # Assemble global force vector
    displacements = np.zeros(total_dofs)
    for node in nodes:
        for idf in range(len(node.eqs)):
            if node.is_fixed[idf]:
                displacements[node.eqs[idf]] = node.presc_disp[idf]
            F[node.eqs[idf]] += node.force[idf]


    # Print global stiffness matrix
    print("Global Stiffness Matrix (K):")
    print(K)
    
    print("Load vector (F):")
    print(F)
    # Multiply global stiffness matrix K with the transpose of the rigid body mode matrix
    # K_rbm = K @ rigid_body_mode_matrix.T
    # print("K multiplied by the transpose of the Rigid Body Mode Matrix:")
    # print(K_rbm.T)

    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy(), displacements, nfree)
    print("Force vector after restraints:")
    print(F_bc/8000)
    # Solve system
    freedisp = np.linalg.solve(K_bc, F_bc)
    # Assign displacements to global vector
    displacements[:nfree] = freedisp

    # axial_forces = []
    # for elem in elements:
    #     dof = elem.global_dof_indices(nodes)
    #     u_elem = displacements[dof]
    #     c = elem.cos
    #     s = elem.sin
    #     prop = section_dict[elem.section_id]
    #     E = prop.E
    #     A = prop.A
    #     T = np.array([-c, -s, c, s])
    #     force = E * A / elem.length * T @ u_elem
    #     axial_forces.append(force)

    return displacements

def ReadJSON(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Parse section properties
    for prop in data['section_properties']:
        section = SectionProperties(
            id=prop['id'],
            E=prop['E'],
            A=prop['A'],
            I=prop['I'] 
        )
        section_dict[prop['id']] = section
    
    # Parse nodes
    for node_data in data['nodes']:
        presc_disp = tuple(node_data['presc_disp']) if "presc_disp" in node_data else (0, 0)
        nodes.append(Node(
            coord=tuple(node_data['coord']),
            eqs=tuple(node_data['eqs']),
            is_fixed=tuple(node_data['is_fixed']),
            presc_disp=presc_disp,
            force=tuple(node_data['force'])
        ))

    # Parse elements
    for element_data in data['elements']:
        typeelement = element_data['type']
        if typeelement == "truss":
            cargas = tuple(element_data['cargas']) if "cargas" in element_data else (0,0,0,0)        
            elements.append(TrussElement(
                elnodes=tuple(element_data['nodes']),
                section_id=element_data['section_id'],
                cargas=cargas
            ))
        elif typeelement == "beam":
            cargas = tuple(element_data['cargas']) if "cargas" in element_data else (0,0,0,0,0,0)        
            elements.append(BeamElement(
                elnodes=tuple(element_data['nodes']),
                section_id=element_data['section_id'],
                cargas=cargas
            ))
        else:
            raise ValueError(f"Unknown element type: {typeelement}")
    # Define nodes
    rigid_body_modes = tuple(data['rigid_body_modes']) if "rigid_body_modes" in data else (0,0)
    return data
# Example usage


if __name__ == "__main__":
    ReadJSON(r'C:\Users\User\OneDrive - dac.unicamp.br\Cv\Cv 712\Python_files\jsons\tt.json')


    displacements = solve_structure(nodes, elements)

    print("Nodal Displacements (m):")
    for i, d in enumerate(nodes):
        ndof = len(d.eqs)
        disp = [displacements[d.eqs]]
        print(f"Node {i}: disp = {disp}")

    for i, el in enumerate(elements):
        reaction = el.reaction_forces(displacements)
        print(f"Element {i}: global forces = {reaction}")
        
    for i, el in enumerate(elements):
        reaction = el.axial_reaction_forces(displacements)
        print(f"Element {i}: axial forces = {reaction}")