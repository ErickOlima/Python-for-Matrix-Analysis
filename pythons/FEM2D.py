import sympy as sp
from sympy import symbols
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#FEM 2D START

def input_data(l_x = 10, l_y =10, ne_x = 3, ne_y = 3):
    
    h_x=l_x/ne_x
    h_y=l_y/ne_y
    
    return l_x, l_y, ne_x, ne_y, h_x, h_y

def local_points():
    
    l_x, l_y, ne_x, ne_y, h_x, h_y = input_data()
    
    n_el = ne_x * ne_y, type (float)
    
    pts = [(i*h_x, j*h_y) for j in range(ne_x+1) for i in range(ne_y+1)]
    
    return n_el, pts
 
def mesh_connection():
    
    l_x, l_y, ne_x, ne_y, h_x, h_y = input_data()
    
    conect = []            
    for i in range(1,ne_x+1):
        for j in range(ne_x):
            n1= i+j*(ne_x+1)
            n2= n1+1
            n3=n2+ (ne_x+1)
            n4= n1 + (ne_x+1)  
            conect.append([n1,n2,n3,n4])
    return conect    

def shape_function(n=5):
    
    x, y = sp.symbols('x y')
    pts = local_points()
    conect = mesh_connection()
    conect = np.array(conect[n])
    pts = np.array(pts[1])
    
    print(f'conect: {conect}')
    cords = pts[conect]
    
    area = cords[1,0]-cords[0,0] * cords[3,1]-cords[0,1]
    
    phi_1 = (1/area) * (x - cords[2,0]) * (y - cords[2,1])
    phi_2 = (1/area) * (x - cords[3,0]) * (y - cords[3,1])
    psi_3 = (1/area) * (x - cords[0,0]) * (y - cords[0,1])
    psi_4 = (1/area) * (x - cords[1,0]) * (y - cords[1,1])
    
    fun = [phi_1, phi_2, psi_3, psi_4]
    
    return fun

def plot_shape_function(n=5):
    x, y = symbols('x y')
    conect = np.array(mesh_connection())
    pts = np.array(local_points()[1])
    fun = shape_function()  # ou qualquer função de forma simbólica
    
    conect = conect[n]  # seleciona o elemento que você quer plotar
    element_coords = pts[conect]  # coordenadas dos 4 nós do elemento
    
    
    x0 = element_coords[0, 0]
    x1 = element_coords[1, 0]
    y0 = element_coords[0, 1]
    y1 = element_coords[3, 1]

    # Cria malha de pontos (grid)
    X = np.linspace(x0, x1, 50)
    Y = np.linspace(y0, y1, 50)
    X, Y = np.meshgrid(X, Y)
    
    for i, phi in enumerate(fun):
        phi_func = sp.lambdify((x, y), phi, 'numpy')
        Z = np.array(phi_func(X, Y))
        
    # Plot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('phi')
    plt.title('Função de forma no elemento {}'.format(n))
    plt.show()
 
plot_shape_function()
    
    
