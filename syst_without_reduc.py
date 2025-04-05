import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp


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


def calc_A(u, v, k, H):
    return H(-u) + k*( H(0) - H(u) + H(v-u) - H(v) )


def calc_B(u, v, k, dH):
    return dH(-u) + k*( dH(0) + dH(v-u) )


def calc_C(u, dH):
    return dH(0) - dH(u)


def calc_D(u, v, dH):
    return dH(v-u) - dH(v)


def calc_E(x, y, k, H):
    return H(0) + k*( H(x) + H(y) )


def create_delta_xy_dyn_func(N, mu, eps1, alp1, eps2, alp2, x, y):
    k = int((N - 1) / 2)
    
    H = create_H_func(eps1, alp1, eps2, alp2)
    dH = create_dH_func(eps1, alp1, eps2, alp2)
    
    Ax = calc_A(x, y, k, H)
    Ay = calc_A(y, x, k, H)
    
    Bx = calc_B(x, y, k, dH)
    By = calc_B(y, x, k, dH)
    
    Cx = calc_C(x, dH)
    Cy = calc_C(y, dH)
    
    Dx = calc_D(x, y, dH)
    Dy = calc_D(y, x, dH)
    
    E = calc_E(x, y, k, H)
    dH_x = dH(x)
    dH_y = dH(y)
    
    def delta_xy_dyn_func(t, Vec):
        theta1 = Vec[0]
        d_theta1 = Vec[1]
        
        delta_x = Vec[2 : 2*k+2 : 2]
        d_delta_x = Vec[3 : 2*k+3 : 2]
        
        delta_y = Vec[2*k+2 :: 2]
        d_delta_y = Vec[2*k+3 :: 2]
        
        # print(theta1, d_theta1)
        # print(delta_x, d_delta_x)
        # print(delta_y, d_delta_y)
        
        res = np.array([0.] * 2*N)
        
        res[0] = Vec[1]
        res[1] = ( ( E + dH_x*sum(delta_x) + dH_y*sum(delta_y) )/N ) / mu
        
        for j in range(k):            
            res[2*j + 2] = d_delta_x[j]
            res[2*j + 3] = ( (Ax - Bx*delta_x[j] + Cx*sum(delta_x) + Dx*sum(delta_y)) / N 
                            - d_delta_x[j] ) / mu
            
            res[2*(j+k) + 2] = d_delta_y[j]
            res[2*(j+k) + 3] = ( (Ay - By*delta_y[j] + Cy*sum(delta_y) + Dy*sum(delta_x)) / N
                                - d_delta_y[j] ) / mu
        
        return res
    
    return delta_xy_dyn_func


def find_Theta0(gamma, N):
    M = int((N + 1) / 2)
    res = np.array([-gamma] * (M - 1) + [0.] + [gamma] * (M - 1))
    # res += np.random.uniform(-0.1, 0.1, 11)
    return res
    
    
def num_integration(rhs, y0, T, step_size):
    sol = solve_ivp(rhs, [0, T], y0, max_step=0.01)
    # arr_sol, arr_t = np.transpose(sol.y), sol.t
    arr_sol, arr_t = sol.y, sol.t
        
    return arr_sol, arr_t


if __name__ == '__main__':
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.
    
    x0 = 0.
    y0 = 2.646
    rhs = create_delta_xy_dyn_func(N, mu, epsilon1, alpha1, epsilon2, alpha2, x0, y0)
    
    Vec0 = np.array([0.] * 2*N)
    
    T = 10.
    step_size = 0.01
    arr_sol, arr_t = num_integration(rhs, Vec0, T, step_size)
    
    # gamma = np.arccos(1 / (1 - N))
    
    # # initial values
    # Theta0 = find_Theta0(gamma, N)
    # Y0 = np.array([1.] * N)
    # print(Theta0)
    
    # vec0 = np.array([0.] * 2*N)
    # for i in range(N):
    #     vec0[2*i] = float(Theta0[i])
    #     vec0[2*i + 1] = Y0[i]
        
    # # numerical integration
    # rhs = create_full_syst_func(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    
    # # step_n = 12000   # 120000
    # T = 120
    # step_size = 0.01
    
    # arr_sol, arr_t = num_integration(rhs, vec0, T, step_size)
        