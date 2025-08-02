import numpy as np
from assemble import (
    equation_count,
    assemble_global_stiffness,
    apply_boundary_conditions,
    initialize_displacements, assemble_global_geometric_stifness
)
from postprocess import post_process_results
import scipy.linalg as la

def verify_singular_modes(nodes,elements):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained

    #print(f"Number of free DOFs: {nfree}")
    #print(f"Number of restrained DOFs: {nrestrained}")
    

    K, F = assemble_global_stiffness(elements, nodes)
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy(), nodes)

    if np.linalg.matrix_rank(K_bc) == nfree:
        print("System is not singular.")
        return 

    eigval, eigvec = np.linalg.eig(K_bc)
    singular_modes = eigvec[:, np.isclose(eigval, 0)]
    
    print(f'SIMILAR MODES FOUND: {singular_modes}\n\n')
    print(f'eigvalues of the system: {eigval}\n\n')
    print(f'eigvectors of the system: {eigvec}\n\n')

    if singular_modes.size == 0:
        print("System is not singular. No singular modes.")
        return False
    else:
        print("System is singular. Singular modes found:")
        for i, mode in enumerate(singular_modes.T):
            print(f"Mode {i+1}: {mode}")
            displacements = np.zeros(total_dofs)
            displacements[:nfree] = mode
            post_process_results(displacements, nodes, elements, withForce=False)
        print("Number of singular modes:", singular_modes.shape[1])
        return True

def solve_structure(nodes, elements):
    nfree, nrestrained = equation_count(nodes)
    print(f"Number of free DOFs: {nfree}")
    print(f"Number of restrained DOFs: {nrestrained}")

    K, F = assemble_global_stiffness(elements, nodes)
    #print("Global Stiffness Matrix (K):")
    #print(K)
    #print("Load vector (F):")
    #print(F)

    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy(), nodes)
    #print("Force vector after applying boundary conditions:")
    #print(F_bc)

    freedisp = np.linalg.solve(K_bc, F_bc)
    displacements = initialize_displacements(nodes)
    displacements[:nfree] = freedisp

    return displacements

def stability_structure (nodes, elements):
    nfree, nrestrained = equation_count(nodes)
    total_dofs = nfree + nrestrained
    
    #print(f"Number of degrees of freedom: {nfree}.")
    #print(f"Number of restrained degrees of freedom: {nrestrained}.")
    
    K,F = assemble_global_stiffness(elements, nodes)
    KG = assemble_global_geometric_stifness(elements, nodes)
          
    #print(f'Global stiffness matrix (K):\n{K}\n\n')
    #print(f'Load vector (F):\n{F}\n\n')
    #print(f'Geometric stiffness matrix (KG):\n{KG}\n\n')
    
    K_bc, F_bc = apply_boundary_conditions(K.copy(), F.copy(), nodes)
    KG_bc, F_bc = apply_boundary_conditions(KG.copy(), F.copy(), nodes)
    
    #print(f'\nnodes {nodes}')
    #print(f'Elements {elements}')
   
    #print(f'\n\nKG_bc:\n {KG_bc}')
    #print(f'\n\nK_bc: \n{K_bc}\n')
    
    try:
        eigenvalues = la.eigh(KG_bc, K_bc, eigvals_only=True)
        
    except np.linalg.LinAlgError as e:
        print(f"Error in eigenvalue computation: {e}")
        return None
    
    print(f"Eigenvalues of the system:\n{eigenvalues}\n\n")
      
    return 1./eigenvalues 