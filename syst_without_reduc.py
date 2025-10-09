import numpy as np
import json
import matplotlib.pyplot as plt
from numpy import sin, cos
from system_dynamic import xy_dyn, num_integration, draw_graph, draw_start_end, mod_2pi


def create_H_func(eps1, alp1, eps2, alp2):
    
    def f_H(ksi):
        return eps1*sin(ksi - alp1) + eps2*sin(2*ksi - alp2)
    
    return f_H


def create_dH_func(eps1, alp1, eps2, alp2):
    
    def f_dH(ksi):
        return eps1*cos(ksi - alp1) + 2*eps2*cos(2*ksi - alp2)
    
    return f_dH


def create_full_syst_func(N, mu, eps1, alp1, eps2, alp2):
    
    def f_syst_dyn(t, Vec):
        res = np.array([0.] * 2*N)
        H = create_H_func(eps1, alp1, eps2, alp2)
        
        for i in range(N):
            sum_N = 0.
            for n in range(N):
                sum_N += H(Vec[2*n] - Vec[2*i])
                
            res[2*i] = Vec[2*i+1]
            res[2*i+1] = (sum_N/N - Vec[2*i+1]) / mu
            
        return res
    
    return f_syst_dyn


def calc_A(u, v, k, dH):
    return dH(-u) + k*( dH(0) + dH(v-u) )


def calc_B(u, dH):
    return dH(0) - dH(u)


def calc_C(u, v, dH):
    return dH(v-u) - dH(v)


def calc_D(x, y, k, H):
    return H(0) + k*( H(x) + H(y) )


def create_delta_xy_dyn_func(N, mu, eps1, alp1, eps2, alp2):
    k = (N-1) // 2
    H = create_H_func(eps1, alp1, eps2, alp2)
    dH = create_dH_func(eps1, alp1, eps2, alp2)
    
    f_xy_dyn = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
    
    def delta_xy_dyn_func(t, Vec):
        xy_vec, delta_xy_vec = Vec[:4], Vec[4:]
        
        # delta_xy consts
        x, y = xy_vec[0], xy_vec[2]
        
        Ax = calc_A(x, y, k, dH)
        Ay = calc_A(y, x, k, dH)
        
        Bx = calc_B(x, dH)
        By = calc_B(y, dH)
        
        Cx = calc_C(x, y, dH)
        Cy = calc_C(y, x, dH)
        
        D = calc_D(x, y, k, H)
        dH_x = dH(x)
        dH_y = dH(y)
        
        # Variables
        # theta1 = Vec[0]   # unused
        d_theta1 = delta_xy_vec[1]
        
        delta_x = delta_xy_vec[2 : 2*k+2 : 2]
        d_delta_x = delta_xy_vec[3 : 2*k+3 : 2]
        
        delta_y = delta_xy_vec[2*k+2 :: 2]
        d_delta_y = delta_xy_vec[2*k+3 :: 2]
        
        # Integrate step
        new_xy_vec = f_xy_dyn(t, xy_vec)
        res = np.array([0.] * 2*N)
        
        res[0] = d_theta1
        res[1] = ( ( D + dH_x*sum(delta_x) + dH_y*sum(delta_y) )/N - d_theta1 ) / mu
        
        for j in range(k):            
            res[2*j + 2] = d_delta_x[j]
            res[2*j + 3] = ( (-Ax*delta_x[j] + Bx*sum(delta_x) + Cx*sum(delta_y)) / N 
                            - d_delta_x[j] ) / mu
            
            res[2*(j+k) + 2] = d_delta_y[j]
            res[2*(j+k) + 3] = ( (-Ay*delta_y[j] + By*sum(delta_y) + Cy*sum(delta_x)) / N
                                - d_delta_y[j] ) / mu
        
        return np.array(new_xy_vec.tolist() + res.tolist())
    
    return delta_xy_dyn_func


def make_matrix_D(x, y, N, mu, dH):
    k = (N-1) // 2
    
    Ax = calc_A(x, y, k, dH)
    Ay = calc_A(y, x, k, dH)
    
    Bx = calc_B(x, dH)
    By = calc_B(y, dH)
    
    Cx = calc_C(x, y, dH)
    Cy = calc_C(y, x, dH)
    
    dH_x = dH(x)
    dH_y = dH(y)
    
    # D matrix
    D = np.array([[0.]*2*N] * 2*N)
    for i in range(k+1):
        D[2*i][2*i+1] = 1.
        D[2*(i+k)][2*(i+k)+1] = 1.
        D[2*i+1][2*i+1] = -1/mu
        D[2*(i+k)+1][2*(i+k)+1] = -1/mu
        for j in range(k):
            if i == 0:
                D[2*i+1][2*j+2] = dH_x/(mu*N)
                D[2*i+1][2*(j+k)+2] = dH_y/(mu*N)
            
            else:
                if i == j+1:
                    D[2*i+1][2*j+2] = (Bx - Ax)/(mu*N)
                    D[2*(i+k)+1][2*(j+k)+2] = (By - Ay)/(mu*N)
                    
                else:
                    D[2*i+1][2*j+2] = Bx/(mu*N)
                    D[2*(i+k)+1][2*(j+k)+2] = By/(mu*N)
                
                D[2*i+1][2*(j+k)+2] = Cx/(mu*N)
                D[2*(i+k)+1][2*j+2] = Cy/(mu*N)
    
    return D


def psi_dyn(N, mu, eps1, alp1, eps2, alp2):
    # H = create_H_func(eps1, alp1, eps2, alp2)
    dH = create_dH_func(eps1, alp1, eps2, alp2)
    
    f_xy_dyn = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
    
    def RHS(t, Vec_xy_psi):
        xy_vec, psi = Vec_xy_psi[:4], Vec_xy_psi[4:].reshape(2*N, 2*N)
        
        # delta_xy consts
        x, y = xy_vec[0], xy_vec[2]
        D = make_matrix_D(x, y, N, mu, dH)
        
        new_xy_vec = f_xy_dyn(t, xy_vec)
        new_psi = D @ psi
        # print(D @ psi)
        return new_xy_vec.tolist() + new_psi.ravel().tolist()
    
    return RHS


def full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2, x0=0.):
    
    def f_stab_det(initial_vec):
        fundament_matrix = np.zeros((2*N, 2*N))
        
        # 2N unit vectors
        arr_psi0 = np.array([[0.]*2*N]*2*N)
        for i in range(2*N): arr_psi0[i][i] = 1.
        
        F_psi = psi_dyn(N, mu, eps1, alp1, eps2, alp2)
        
        vec_for_int = [x0, initial_vec[0], initial_vec[1], initial_vec[2]]
        Vec0 = vec_for_int + arr_psi0.ravel().tolist()
        arr_sol, arr_t = num_integration(F_psi, Vec0, initial_vec[3])
        arr_psi = np.transpose(arr_sol)[-1]
        arr_psi = arr_psi[4:]
        arr_psi = arr_psi.reshape(2*N, 2*N)
        
        for i in range(2*N):
            fundament_matrix[i] = arr_psi[i]
            
        eigvals = np.linalg.eigvals(fundament_matrix)
        
        # print(eigvals, '\n')
        # print()
        # print(fundament_matrix)
        # print()
        
        stability_err = 2*1e-6
        if np.all(np.abs([eigvals[1]] + np.abs(eigvals[3:]) < 1)) and np.all(np.abs([eigvals[0], eigvals[2]]) <= 1+stability_err):
            return True, eigvals
        else:
            return False, eigvals
        # inds = np.where(np.abs(eigvals) >= 1.)
        # return inds
        
    return f_stab_det


def integrate_full_syst(N, mu, eps1, alp1, eps2, alp2, init_xy_vec, T, noise=False):
    k = (N-1)//2
    init_vec = np.array([0, 0] +
                        [init_xy_vec[0], init_xy_vec[1]]*k + 
                        [init_xy_vec[2], init_xy_vec[3]]*k)
    
    if noise: init_vec = init_vec + np.random.uniform(0, 1, 2*N)
    
    # Integration
    rhs = create_full_syst_func(N, mu, eps1, alp1, eps2, alp2)
    arr_sol, arr_t = num_integration(rhs, init_vec, T)
    
    return arr_sol, arr_t


def draw_full_syst(arr_sol, arr_t, T, start=True, end=True, T_inter=200, relative0=True):
    arr_sol, arr_t = np.array(arr_sol), np.array(arr_t)
    
    
    # Result arrays
    if relative0:
        arr_thetas = np.array([arr_sol[i] for i in range(2, 2*N, 2)]) - arr_sol[0]
        arr_d_thetas = np.array([arr_sol[i] for i in range(3, 2*N, 2)]) - arr_sol[1]
    else:
        arr_thetas = np.array([arr_sol[i] for i in range(0, 2*N, 2)])
        arr_d_thetas = np.array([arr_sol[i] for i in range(1, 2*N, 2)])
    
    
    # Drawing
    # draw graph for thetas by T
    draw_start_end(mod_2pi(arr_thetas), arr_t, r'$\theta_k$', T,
                    draw_start=start, draw_end=end, T_inter=T_inter)
    
    # draw graph for d_thetas by T
    draw_start_end(arr_d_thetas, arr_t, r'$\dot{\theta}_k$', T,
                    draw_start=start, draw_end=end, T_inter=T_inter, 
                    ylims=(np.min(arr_d_thetas)*1.1-0.05, np.max(arr_d_thetas)*1.1+0.05))
    
    
def find_init_rb_state(arr_sol, arr_t):
    for i in range(len(arr_t)):
        if arr_t[i] < 100: continue
        
        if np.abs(mod_2pi(arr_sol[0][i])) < 1e-2 and np.abs(mod_2pi(arr_sol[2][i])) < 1e-2:
            print(i)
            print([0, arr_sol[3][i], np.mod(arr_sol[-2][i] + np.pi, 2*np.pi) - np.pi, arr_sol[-1][i]])
            break
        

def find_init_br_state(arr_sol, arr_t):
    Found = False
    for i in range(len(arr_t)):
        if arr_t[i] < 100: continue
        
        if not Found and np.abs(mod_2pi(arr_sol[0][i])) < 1e-3:
            init_state = np.array([mod_2pi(arr_sol[2][i]), arr_sol[3][i], 
                                   mod_2pi(arr_sol[-2][i]), arr_sol[-1][i]])
            init_i = i
            
            print(init_i)
            print(init_state.tolist(), '\n')
            Found = True
            continue
        
        elif Found and i-init_i > 1000 and np.abs(init_state[1] - arr_sol[3][i]) < 1e-3:
            end_state = np.array([mod_2pi(arr_sol[2][i]), arr_sol[3][i], 
                                  mod_2pi(arr_sol[-2][i]), arr_sol[-1][i]])
            print(i)
            print(end_state.tolist(), '\n')
            print((i - init_i)/100)
            break


if __name__ == '__main__':
    # Font settings
    rc_fonts = {
        'font.size': 20,
        "text.usetex": True,
        'mathtext.default': 'regular',
        'text.latex.preamble': r"\usepackage{bm}",
        "font.family": "serif",
        "font.serif": "computer modern roman",
    }
    plt.rcParams.update(rc_fonts)
    
    
    # Always parameters
    dir_name = 'Syst_without_reduc_results/'
    mu = 1.0
    epsilon1 = 1.0
    
    
    # Changing parameters
    # N = 11
    # alpha1 = 1.7
    # epsilon2 = 0.08
    # alpha2 = -2.0
    # initial_xy_vec = np.array([0, -0.05921021, 2.64676814, 0.14678535])
    # T = 41.02969191
    
    # N = 11
    # alpha1 = 1.6
    # epsilon2 = 0.1378
    # alpha2 = 2.123
    # initial_xy_vec = np.array([0, 0.007466, 2.133077, 0.124330])
    # periodT = 63.700816
    
    N = 11
    alpha1 = 1.7
    epsilon2 = 0.075
    alpha2 = 0
    # initial_xy_vec = np.array([0, -0.05921021, 2.64676814, 0.14678535])
    initial_xy_vec = np.array([1.9812276696916573, 0.16464216503908244, -1.2098409186762211, 0.09448776865900128])
    periodT = 28.89

    
    # Stable determination
    # f_stab = full_syst_stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    # isStable, eigvs = f_stab(np.array([initial_xy_vec[1], initial_xy_vec[2], initial_xy_vec[3], periodT]))
    # print(isStable, max(eigvs))
    
    
    # Integrate
    T = 1500
    isNoise = False
    try:
        # 1/0
        with open(dir_name + f'results_noise={isNoise}_N={N}_alp1={alpha1:.3f}_eps1={epsilon1:.3f}_alp2={alpha2:.3f}_eps2={epsilon2:.3f}.txt', 'r') as fr:
            initial_xy_vec, arr_sol, arr_t = json.load(fr)
    
    except:
        print('New calculating')
        arr_sol, arr_t = integrate_full_syst(N, mu, epsilon1, alpha1, epsilon2, alpha2, initial_xy_vec, T, noise=isNoise)
        with open(dir_name + f'results_noise={isNoise}_N={N}_alp1={alpha1:.3f}_eps1={epsilon1:.3f}_alp2={alpha2:.3f}_eps2={epsilon2:.3f}.txt', 'w') as fw:
            json.dump([initial_xy_vec.tolist(), arr_sol.tolist(), arr_t.tolist()], fw)
            
    draw_full_syst(arr_sol, arr_t, arr_t[-1], T_inter=500, start=True, relative0=True)
    
    # find_init_rb_state(arr_sol, arr_t)
    find_init_br_state(arr_sol, arr_t)
    
    # arr = np.array([-87.97266460914177, -0.29475409343962183, 
    #                 -75.39903405905316, -0.2925483817605887,
    #                 -71.14440795778007, -0.1542180828494336])
    # for i in range(3):
    #     arr[i] -= arr
    