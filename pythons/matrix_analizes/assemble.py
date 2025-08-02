import numpy as np
from beam_element import BeamElement
from truss_element import TrussElement
from spring_element import SpringElement

def equation_count(nodes):
    free = set()
    restrained = set()
    for node in nodes:
        ndof = len(node.eqs)
        if ndof not in [2, 3]:
            raise ValueError("Node must have 2 or 3 degrees of freedom.")
        for i in range(ndof):
            if node.is_fixed[i]:
                restrained.add(node.eqs[i])
            else:
                free.add(node.eqs[i])

    ordered_free = sorted(free)
    ordered_restrained = sorted(restrained)

    for i, eq in enumerate(ordered_free):
        if eq != i:
            raise ValueError(f"Free degrees of freedom {eq} is not ordered correctly.")
    for i, eq in enumerate(ordered_restrained):
        if eq != i + len(free):
            raise ValueError(f"Restrained degrees of freedom {eq} is not ordered correctly.")

    return len(free), len(restrained)

def assemble_global_stiffness(elements, nodes):
    
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained
    K = np.zeros((total_dofs, total_dofs))
    F = np.zeros(total_dofs)

    for elem in elements:
        k_local, f_local = elem.local_stiffness_matrix()
        dof_indices = elem.global_dof_indices()
        for i in range(len(dof_indices)):
            F[dof_indices[i]] += f_local[i]
            for j in range(len(dof_indices)):
                K[dof_indices[i], dof_indices[j]] += k_local[i, j]

    for node in nodes:
        for i in range(len(node.eqs)):
            F[node.eqs[i]] += node.force[i]

    return K, F

def initialize_displacements(nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained
    displacements = np.zeros(total_dofs)
    for node in nodes:
        for i in range(len(node.eqs)):
            if node.is_fixed[i]:
                displacements[node.eqs[i]] = node.presc_disp[i]
    return displacements

def apply_boundary_conditions(K, F, nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained
    displacements = initialize_displacements(nodes)
    K_bc = K[:nfree, :nfree]
    F_bc = F[:nfree] - K[:nfree, nfree:] @ displacements[nfree:]
    
    #print(f"\n\nApplying boundary conditions:\n {nfree},\n free DOFs to K_bc \n({K_bc}),\n {nrestrained} restrained DOFs to F_bc\n{F_bc}.\nBut K is {K}")
    
    return K_bc, F_bc

def local_geometric_stiffness_matrix(self):
        k_geo = self.axial_geometric_stiffness_matrix()
        rotation = self.rotation_matrix()
        k_rotated = rotation.T @ k_geo @ rotation
        
        return k_rotated, self.cargas  # if already in global coords

def assemble_global_geometric_stifness(elements, nodes):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained
    K_g = np.zeros((total_dofs, total_dofs))

    for elem in elements:
        k_local, f_local = elem.local_stiffness_matrix()
        dof_indices = elem.global_dof_indices()
        neqs = len(dof_indices)
       
               
        k_geo = elem.local_geometric_stiffness_matrix()#[0:neqs, 0:neqs]  
        dof_indices = elem.global_dof_indices()
        dof_indices[0:4] = np.array(dof_indices[0:4], dtype=int)
        for i in range (len(dof_indices)):
            for j in range(len(dof_indices)):
                #d,b =dof_indices[i['T']], dof_indices[j['T']]
                K_g[dof_indices[i], dof_indices[j]] += k_geo[i, j]
        
        if neqs != 6 and neqs != 4:
            continue
        if elem.normal >=0:
            continue
        
    return K_g