

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