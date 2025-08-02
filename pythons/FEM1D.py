import numpy as np
import sympy as sp
from sympy import lambdify
import matplotlib.pyplot as plt




def create_mesh(x_0=0, x_1=1, n_el=4):
    
    return x_0, x_1, n_el

def coordenates():
    
    x_0, x_1, n_el = create_mesh()
    h = (x_1 - x_0) / n_el
    
    pts = np.arange(x_0, x_1 + h, h)
    
    #print(f"Mesh created from {x_0} to {x_1} with {n_el} elements.")
    #print("Coordinates of mesh points:\n", pts)
    
    return pts       
  
def elements():
   
    elem= []
    x_0, x_1, n_el = create_mesh() 
    h = (x_1 - x_0) / n_el
    pts = coordenates()
    
    for i in range(x_0, n_el):
        for j in range(2):
            el_step = (pts[i], pts[i+1])
        elem.append(el_step)
    
    #print("\nElements of the mesh:\n", elem)
    return elem  

def basin_functions(x=None):
    x = sp.Symbol('x') if x is None else x

    pts = coordenates()
    x_0, x_1, el = create_mesh()  # define os limites e número de elementos
    h = (x_1 - x_0) / el
    Points = [x_0 + i*h for i in range(el + 1)]

    xi = [(x - Points[i]) 
          for i in range(el)]
    
    psi_a = [
    sp.Piecewise(
        (1 - (xi[i] / h), ((i) * h <= x) & (x < (i + 1) * h)),
        (0, x <= (i - 1) * h),
        (0, x >= (i + 3) * h)
    )
    for i in range(el)  
    ]

    psi_b = [
        sp.Piecewise(
            ((xi[i] / h), ((i) * h <= x) & (x < (i + 1) * h)),
            (0, x <= (i - 1) * h),
            (0, x >= (i + 3) * h)
        )
        for i in range(el)
    ]

    #print('\nBasin functions created:\n')
    #print(f'Psi_a: {psi_a}\n')
    #print(f'Psi_b: {psi_b}\n')
    
    return psi_a, psi_b, xi

def hat_functions():
    
    psi_a, psi_b = basin_functions()
   
    return psi_a + psi_b
        
def plot_hat_functions(num_points=500):
    x = sp.Symbol('x')
    plt.figure(figsize=(10, 6))
    
    psi_a, psi_b = basin_functions()
    x_vqals = np.linspace(0, 1, num_points)
    psi_a_funcs = [lambdify(x, f, 'numpy') for f in psi_a]
    psi_b_funcs = [lambdify(x, f, 'numpy') for f in psi_b]
    
    plt.figure(figsize=(10, 6))
    x_vals = np.linspace(0, 1, num_points)

    for i, f in enumerate(psi_a_funcs + psi_b_funcs):
        plt.plot(x_vals, f(x_vals), label=f'psia[{i}]')

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("hat functions")
    plt.grid(True)
    plt.legend()
    plt.show()
       
def derivatives():
    x = sp.Symbol('x')
    
    psi_a, psi_b, _ = basin_functions()
        
    d_a = [sp.diff(psi_a, x) for psi_a in psi_a]
    d_b = [sp.diff(psi_b, x) for psi_b in psi_b]     
    
    return d_a, d_b 

def integrate():
    
    psi_a, psi_b, xi = basin_functions()
    pts = coordenates()
    #print(f'\nCoordinates of mesh points:\n{pts}\n')
    
    d_a, d_b = derivatives()
    
    x = sp.Symbol('x')
    x_0, x_1, el = create_mesh()
    h = (x_1 - x_0) / el

    
    k_aa = [sp.integrate(d_a[i]**2 + psi_a[i]**2, (x, 0, h))for i in range(el)]
    k_ab = [sp.integrate(psi_a[i] * psi_b[i] + d_a[i] * d_b[i], (x, 0, h))for i in range(el)]
    k_ba = k_ab
    k_bb = [sp.integrate(d_b[i]**2 + psi_b[i]**2, (x, 0, h)) for i in range(el)]
    
    
    f_a = [sp.integrate((pts[i]+xi[i])*(1-xi[i]/h),(x, 0, h)) for i in range(el)]
    f_b = [sp.integrate((pts[i]+xi[i])*(xi[i]/h),(x, pts[i], pts[i+1])) for i in range(el)]
    
    #print(f'\nLocal stiffness matrix components:\n')
    #print(f'k_aa: {k_aa}\n')
    #print(f'k_ab: {k_ab}\n')
    #print(f'k_ba: {k_ba}\n')
    #print(f'k_bb: {k_bb}\n')
    #print(f'Local force vector components:\n')
    #print(f'f_a: {f_a}\n')
    #print(f'f_b: {f_b}\n')
        
    
    return k_aa, k_ab, k_ba, k_bb, f_a, f_b

def local_matixes():
    _, _, el = create_mesh()
    k_aa, k_ab, k_ba, k_bb, f_a, f_b = integrate()
    
    k_local = np.zeros((2, 2))
    f_local = np.zeros(2)
    
    k_local = [(
        k_aa[0],k_ab[0]),(
        k_ba[0],k_bb[0])
    ]  
    
    f_local = [(
        f_a[i], f_b[i])
        for i in range(el)
    ]
    
    #print(f'\nLocal stiffness matrix:\n{k_local}\n')
    #print(f'Local force vector:\n{f_local}\n')
    return k_local, f_local

def global_matrix():
    _, _, el = create_mesh()
    dim = el + 1
    k_global = np.zeros((dim, dim), dtype=float)   
    k_local, f_local = local_matixes()
    f_global = np.zeros(dim, dtype=float)

    # Montagem da matriz de rigidez
    for i in range(el):
            
        k_block = np.array(k_local, dtype=float)
        k_global[i:i+2, i:i+2] += k_block
        k_global[0,0]=k_global[el,el]= int(1*10**9)

    # Montagem do vetor de forças
    for i in range(el-1):
        f_global[i+1] += f_local[i][1] + f_local[i+1][0]

    f_global[0] = f_local[0][0]
    f_global[-1] = f_local[-1][1]

    #print(f'\nGlobal stiffness matrix:\n{k_global}\n')
    #print(f'Global force vector:\n{f_global}\n')

    return k_global, f_global

def solve_system():
    
    k_global, f_global = global_matrix()
   
    displacements = np.linalg.solve(k_global, f_global)
    #print(f'Displacements:\n{displacements}\n')
    #print("\nSystem solved successfully.")
   
    u_= displacements
    
    return u_   

def final_solution():
    _, _, el = create_mesh()
    k_global, f_global = global_matrix()
    u_ = solve_system()
    u_ =np.array(u_, dtype=float)
    
    psi_a, psi_b, xi = basin_functions()
    #psi_=psi_a[0][0]
    
   
    psi_= [psi_b[i]+ psi_a[i+1] 
        for i in range(len(psi_a)-1)]
   
    for fun in psi_:
        print(f'\nHat functions:\n{fun}\n')
    
    Fs =[psi_[i]*u_[i+1] for i in range(len(psi_)-1)]
    
    print(f'\nFinal solution:\n{Fs}\n')
    
      
    return Fs, u_

def plot_solution():
    
    x = sp.Symbol('x')
    Fs, u_ = final_solution()

    # Corrigido: só extrai o primeiro item se for tupla/lista
    Fs = [f[0] if isinstance(f, (tuple, list)) else f for f in Fs]

    x_vals = np.linspace(0, 1, 500)
    Fs_funcs = [lambdify(x, f, 'numpy') for f in Fs]

    plt.figure(figsize=(10, 6))
    for i, f in enumerate(Fs_funcs):
        plt.plot(x_vals, f(x_vals), label=f'Fs[{i}]')

    plt.xlabel("x")
    plt.ylabel("Fs(x)")
    plt.title("Final Solution")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #create_mesh()
    #coordenates()
    #elements()
    #basin_functions()
    #hat_functions()
    #plot_hat_functions()
    #derivatives()
    #integrate()
    #local_matixes()
    #global_matrix()
    #solve_system()
    #final_solution()
    #plot_solution()
    exit(0)