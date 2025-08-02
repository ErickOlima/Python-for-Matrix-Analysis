import json
from section_properties import SectionProperties
from node import Node
from truss_element import TrussElement
from beam_element import BeamElement
from spring_element import SpringElement

def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    section_dict = {}
    for prop in data['section_properties']:
        section = SectionProperties(
            id=prop['id'],
            E=prop['E'],
            A=prop['A'],
            I=prop['I']
        )
        section_dict[prop['id']] = section

    nodes = []
    for node_data in data['nodes']:
        presc_disp = tuple(node_data['presc_disp']) if "presc_disp" in node_data else (0, 0)
        nodes.append(Node(
            coord=tuple(node_data['coord']),
            eqs=tuple(node_data['eqs']),
            is_fixed=tuple(node_data['is_fixed']),
            presc_disp=presc_disp,
            force=tuple(node_data['force'])
        ))

    elements = []
    for element_data in data['elements']:
        eltype = element_data['type']
        if eltype == "truss":
            cargas = tuple(element_data.get('cargas', (0,0,0,0)))
            if "normal" in element_data:
                normal = element_data['normal']
            else:
                normal = (0, 0, 0)
            
            elements.append(TrussElement(
                elnodes=tuple(element_data['nodes']),
                section_id=element_data['section_id'],
                cargas=cargas,
                section_dict=section_dict,
                nodes=nodes,
                normal=normal
            ))

        elif eltype == "beam":
            cargas = tuple(element_data.get('cargas', (0,0,0,0,0,0)))
            
            if "normal" in element_data:
                normal = element_data['normal']
            else:
                normal = (0, 0, 0)
            
            elements.append(BeamElement(
                elnodes=tuple(element_data['nodes']),
                section_id=element_data['section_id'],
                cargas=cargas,
                section_dict=section_dict,
                nodes=nodes,
                normal=normal
            ))
        elif eltype == "spring":
            elements.append(SpringElement(
                elnodes=tuple(element_data['nodes']),
                spring_value=element_data['spring_values'],
                nodes=nodes
            ))
        else:
            raise ValueError(f"Unknown element type: {eltype}")

    rigid_body_modes = tuple(data['rigid_body_modes']) if "rigid_body_modes" in data else (0, 0)

    return section_dict, nodes, elements, rigid_body_modes
