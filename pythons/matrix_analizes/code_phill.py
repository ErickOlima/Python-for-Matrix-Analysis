import numpy as np
import scipy.linalg as la
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
        
        print(f'Local stiffness matrix for element {self.elnodes}:\n\n{K_local}\n\n')   
        return K_local, self.cargas
        

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], j[0], j[1]]
    
    def element_forces(self, displacement):
        eqs = self.global_dof_indices(nodes)
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        # Calculate element forces
        element_force = elstiff @ disploc
        return element_force, elforce
    
    def reaction_forces(self, displacement):
        # Calculate reaction forces
        reaction, elforce = self.element_forces(displacement)
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


class SpringElement:
    def __init__(self, elnodes=(0,0), spring_value=(0,0,0)):
        self.values = np.array(spring_value, dtype=float)
        self.elnodes = np.array(elnodes, dtype=int)
        
    def local_stiffness_matrix(self):
        s1= self.values[0]
        s2= self.values[1]
        s3= self.values[2]
        k = np.array([
            [ s1, 0, 0, -s1, 0, 0],
            [ 0, s2, 0, 0, -s2, 0],
            [ 0, 0, s3, 0, 0, -s3],
            [-s1, 0, 0, s1, 0, 0],
            [ 0, -s2, 0, 0, s2, 0],
            [ 0, 0, -s3, 0, 0, s3]
        ] )
        cargas=(0,0,0,0,0,0)
        return k, cargas
        

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1],i[2], j[0], j[1],j[2]]
    
    
    def element_forces(self, displacement):
        eqs = self.global_dof_indices(nodes)
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        # Calculate element forces
        element_force = elstiff @ disploc
        return element_force, elforce
    
    def reaction_forces(self, displacement):
        # Calculate reaction forces
        reaction, elforce = self.element_forces(displacement)
        # Add local forces
        reaction -= elforce
        return reaction
    
    def axial_reaction_forces(self,displacement):
        reaction = self.reaction_forces(displacement)
        # rotation = self.rotation_matrix()
        # axial_reaction = rotation @ reaction
        return reaction
    
class BeamElement:
    def __init__(self, elnodes=(0,0), section_id= 0,cargas=(0,0,0,0,0,0),normal = (0)):
        self.cargas = np.array(cargas, dtype=float)
        self.elnodes = np.array(elnodes, dtype=int)
        self.section_id = section_id
        self.length, self.cos, self.sin = self._calculate_geometry()
        self.normal = normal

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
    
    def axial_geometric_stiffness_matrix(self):
        l = self.length
        K_geo = np.array([
            [ 0, 0, 0, 0, 0, 0],
            [ 0, 6./(5.*l), 1./10., 0, -6./(5.*l), 1./10.],
            [ 0, 1./10., 2.*l/15., 0, -1./10., -1./30.],
            [0, 0, 0, 0, 0, 0],
            [ 0, -6./(5.*l), -1./10., 0, 6./(5.*l), -1./10.],
            [ 0, 1./10., -l/30., 0, -1/10., 2.*l/15.]
        ] ) * self.normal
        return K_geo
    
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
    
    def local_geometric_stiffness_matrix(self):
        K_geo = self.axial_geometric_stiffness_matrix()
        rotation = self.rotation_matrix()
        K_rotated = rotation.T @ K_geo @ rotation
        # Adding the local loads to the global loads
        # Rotating the local loads to global coordinates
        
        return K_rotated

    def global_dof_indices(self, nodes):
        i = nodes[self.elnodes[0]].eqs
        j = nodes[self.elnodes[1]].eqs
        return [i[0], i[1], i[2], j[0], j[1], j[2]]
    
    def element_forces(self, displacement):
        eqs = self.global_dof_indices(nodes)
        disploc = displacement[eqs]
        elstiff, elforce = self.local_stiffness_matrix()
        # Calculate element forces
        element_force = elstiff @ disploc
        return element_force, elforce
    
    def reaction_forces(self, displacement):
        reaction, elforce = self.element_forces(displacement)
        reaction -= elforce
        return reaction
    
    def axial_reaction_forces(self,displacement):
        reaction = self.reaction_forces(displacement)
        rotation = self.rotation_matrix()
        axial_reaction = rotation @ reaction
        return axial_reaction


def assemble_global_stiffness(elements, nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

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
    
    for node in nodes:
        for idf in range(len(node.eqs)):
            F[node.eqs[idf]] += node.force[idf]

    return K, F

def assemble_global_geometric_stiffness(elements, nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

    K = np.zeros((total_dofs, total_dofs))
    for elem in elements:
        k_local,f_local = elem.local_stiffness_matrix()
        dof_indices = elem.global_dof_indices(nodes)
        neqs = len(dof_indices)
        if(neqs != 6):
            continue
        k_geo = elem.local_geometric_stiffness_matrix()
        for i in range(neqs):
            for j in range(neqs):
                K[dof_indices[i], dof_indices[j]] += k_geo[i, j]
    
    return K


def initialize_displacements(nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained
    displacements = np.zeros(total_dofs)
    for node in nodes:
        for idf in range(len(node.eqs)):
            if node.is_fixed[idf]:
                displacements[node.eqs[idf]] = node.presc_disp[idf]
    return displacements

def apply_boundary_conditions(K, F):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained
    displacements = initialize_displacements(nodes)
        # Assemble global force vector

    # Apply boundary conditions
    K_bc = K[:nfree,:nfree]
    F_bc = F[:nfree]- K[:nfree,nfree:] @ displacements[nfree:]
    # Apply prescribed displacements
    
    print(f"\n\nApplying boundary conditions:\n {nfree},\n free DOFs to K_bc \n({K_bc}),\n {nrestrained} restrained DOFs to F_bc\n{F_bc}..\nBut K is {K}")    
    
    return K_bc, F_bc

def equation_count(nodes):
    free = set()
    restrained = set()
    for node in nodes:
        ndof = len(node.eqs)
        if ndof != 2 and ndof != 3:
            raise ValueError("Node must have 2 degrees of freedom")
        for i in range(ndof):
            if(node.is_fixed[i]) : restrained.add(node.eqs[i])
            else : free.add(node.eqs[i])
    ordered_free = sorted(free)
    ordered_restrained = sorted(restrained)
    for i, eq in enumerate(ordered_free):
        if eq != i:
            raise ValueError(f"Free degree of freedom {eq} is not in the expected order.")
    for i, eq in enumerate(ordered_restrained):
        if eq != i + len(free):
            raise ValueError(f"Restrained degree of freedom {eq} is not in the expected order.")
    
    return len(free), len(restrained)

def verify_singular_modes(nodes,elements):
    
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

    # Print the number of free and restrained degrees of freedom
    print(f"Number of free degrees of freedom: {nfree}")
    print(f"Number of restrained degrees of freedom: {nrestrained}")
    # Assemble global stiffness matrix
    K,F = assemble_global_stiffness(elements, nodes)
    
    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy())
    # Check if the system is singular
    if np.linalg.matrix_rank(K_bc) == nfree:
        print("The system is not singular. The number of free degrees of freedom is equal to the number of equations.")
        return
    
    eigval, eigvec = np.linalg.eig(K_bc)
    singular_modes = eigvec[:, np.isclose(eigval, 0)]
    
    print(f'SIMILAR MODES: {singular_modes}\n\n')
    print(f'eigvalues of the system: {eigval}\n\n')
    print(f'eigvectors of the system: {eigvec}\n\n')
    
    if singular_modes.size == 0:
        print("The system is not singular. No singular modes found.")
        return False
    else:
        print("The system is singular. Singular modes found:")
        for i, mode in enumerate(singular_modes.T):
            print(f"Mode {i+1}: {mode}")
            displacements = np.zeros(total_dofs)
            displacements[:nfree] = mode
            post_process_results(displacements, nodes, elements, withForce=False)

        
        print("The number of singular modes is:", singular_modes.shape[1])
        return True
    
    
def solve_structure(nodes, elements):
    
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

    # Print the number of free and restrained degrees of freedom
    print(f"Number of free degrees of freedom: {nfree}")
    print(f"Number of restrained degrees of freedom: {nrestrained}")
    # Assemble global stiffness matrix
    K,F = assemble_global_stiffness(elements, nodes)
    
    # Print global stiffness matrix
    print("Global Stiffness Matrix (K):")
    print(K)
    
    print("Load vector (F):")
    print(F)
    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy())
    print("Force vector after restraints:")
    print(F_bc)
    # Solve system
    freedisp = np.linalg.solve(K_bc, F_bc)
    # Assign displacements to global vector
    displacements = initialize_displacements(nodes)
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

def stability_structure(nodes, elements):
    
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree+nrestrained

    # Print the number of free and restrained degrees of freedom
    print(f"Number of free degrees of freedom: {nfree}")
    print(f"Number of restrained degrees of freedom: {nrestrained}")
    # Assemble global stiffness matrix
    K,F = assemble_global_stiffness(elements, nodes)
    KG = assemble_global_geometric_stiffness(elements, nodes)
    
    # Print global stiffness matrix
    print("Global Stiffness Matrix (K):")
    print(K)
    
    print("Load vector (F):")
    print(F)
    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy())
    KG_bc, F_bc = apply_boundary_conditions(KG.copy(), F.copy())
    # Solve system
   
    
    print(f'\n\nnodes {nodes}')
    print(f'Elements {elements}\n\n')
   
    print(f'\n\nKG_bc:\n {KG_bc}')
    print(f'\n\nK_bc: \n{K_bc}\n')
    
   
    eigenvalues = la.eigh(KG_bc, K_bc, eigvals_only=True)
    
    print(f"Eigenvalues of the system:\n{eigenvalues}\n\n")
    
    return 1./eigenvalues


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
            if "normal" in element_data:
                normal = element_data['normal']
            else:
                normal = (0)
            elements.append(BeamElement(
                elnodes=tuple(element_data['nodes']),
                section_id=element_data['section_id'],
                cargas=cargas,
                normal=normal
            ))
        elif typeelement == "spring":
            elements.append(SpringElement(
                elnodes=tuple(element_data['nodes']),
                spring_value=element_data['spring_values']
            ))
        else:
            raise ValueError(f"Unknown element type: {typeelement}")
    # Define nodes
    rigid_body_modes = tuple(data['rigid_body_modes']) if "rigid_body_modes" in data else (0,0)
    return data
# Example usage

def post_process_results(displacements, nodes, elements, withForce):
    for i, node in enumerate(nodes):
        ndof = len(node.eqs)
        disp = [displacements[node.eqs[j]] for j in range(ndof)]
        print(f"Node {i}: disp = {disp}")

    for i, el in enumerate(elements):
        if(withForce == False):
            reaction, elforce = el.element_forces(displacements)
        else:
            reaction = el.reaction_forces(displacements)
            elforce = (0,0,0,0)
        print(f"Element {i}: global forces = {reaction}")
    
        if(withForce == True):    
           for i, el in enumerate(elements):
                reaction = el.axial_reaction_forces(displacements)
                print(f"Element {i}: axial forces = {reaction}")

if __name__ == "__main__":
    ReadJSON(r'C:\Users\User\OneDrive - dac.unicamp.br\Cv\IC_Phill\Python_files\jsons\2705.json')

    eigenvalues = stability_structure(nodes, elements)
    print("Eigenvalues of the system:")
    print(eigenvalues)
    print("Stability analysis completed.")
    exit (0)
    print("Global Geometric Stiffness Matrix (KG):")
    print(KG)
    if(verify_singular_modes(nodes,elements) == True):
        print("The structure has singular modes. Please check the model.")
        exit(0)
    
    displacements = solve_structure(nodes, elements)

    post_process_results(displacements, nodes, elements, withForce=True)