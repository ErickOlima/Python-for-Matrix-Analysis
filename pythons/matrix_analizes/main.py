from reader import read_json
from solver import solve_structure, verify_singular_modes, stability_structure
from assemble import assemble_global_geometric_stifness, assemble_global_stiffness
from postprocess import post_process_results

if __name__ == "__main__":
    # Leitura de entrada
    section_dict, nodes, elements, rigid_body_modes = read_json(r'C:\Users\User\OneDrive - dac.unicamp.br\Cv\IC_Phill\Python_files\jsons\stb.json')

    eigenvalues = stability_structure(nodes, elements)
    KG = assemble_global_geometric_stifness(elements, nodes)
    K,F = assemble_global_stiffness(elements, nodes)
    
    print(f'Eigvaliues of the system: \n{eigenvalues}')
    print(f'\nStability analysis completed.')
    
    print(f'\nGlobal Geometric Stiffness Matrix:\n{KG}')
    print(f'\n The Global stiffnes matrix is:\n{K}')
   
    # Verificação de estabilidade
    if verify_singular_modes(nodes, elements):
        print("The structure has singular modes. Please check the model.")
        exit(0)
        

    # Resolução do sistema
    displacements = solve_structure(nodes, elements)

    # Pós-processamento com forças internas
    post_process_results(displacements, nodes, elements, withForce=True)
