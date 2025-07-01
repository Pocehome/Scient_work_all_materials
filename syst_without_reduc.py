import numpy as np
from numpy import sin, cos
from system_dynamic import xy_dyn, num_integration, draw_graph


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
    
    f_xy_dyn = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    
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


def full_syst_stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    
    def f_stab_det(initial_vec):
        fundament_matrix = np.zeros((2*N, 2*N))
        
        # 2N unit vectors
        arr_psi0 = np.array([[0.]*2*N]*2*N)
        for i in range(2*N): arr_psi0[i][i] = 1.
        
        F_psi = psi_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
        
        vec_for_int = [0., initial_vec[0], initial_vec[1], initial_vec[2]]
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
        
        stability_err = 2*1e-3
        if np.all(np.abs([eigvals[1]] + np.abs(eigvals[3:]) < 1)) and np.all(np.abs([eigvals[0], eigvals[2]]) <= 1+stability_err):
            return True, eigvals
        else:
            return False, eigvals
        # inds = np.where(np.abs(eigvals) >= 1.)
        # return inds
        
    return f_stab_det


def draw_start_end(arr_sol, arr_t, _y_name, T, ex_legend=[]):
    draw_graph([arr_t], arr_sol,
               [(0, 100), (-np.pi-0.5, np.pi+0.5)],
               x_name='t', y_name=_y_name,
               colors=['black'], legend=ex_legend)

    draw_graph([arr_t], arr_sol,
               [(T-150, T), (-np.pi-0.5, np.pi+0.5)],
               x_name='t', y_name=_y_name,
               colors=['black'], legend=ex_legend)


def draw_examples_with_theta1(arr_sol, arr_t, N, T):
    arr_theta_1 = [np.mod(arr_sol[0], 2*np.pi) - np.pi]
    arr_d_theta_1 = [arr_sol[1]]
    arr_theta_x = [np.mod(arr_sol[i], 2*np.pi) - np.pi for i in range(2, N+1, 2)]
    arr_d_theta_x = [arr_sol[i] for i in range(3, N+2, 2)]
    arr_theta_y = [np.mod(arr_sol[i], 2*np.pi) - np.pi for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    # draw graph for theta_x by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_x, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_x$', T, ex_legend=[r'$\theta_1$'])

    # draw graph for theta_y by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_y, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_y$', T, ex_legend=[r'$\theta_1$'])
    
    # draw graph for d_theta_x by T
    draw_start_end(arr_d_theta_1+arr_d_theta_x, arr_t, r'$\dot{\theta}_x$', 
                  T, ex_legend=[r'$\dot{\theta}_1$'])
    
    # draw graph for d_theta_y by T
    draw_start_end(arr_d_theta_1+arr_d_theta_y, arr_t, r'$\dot{\theta}_y$', 
                  T, ex_legend=[r'$\dot{\theta}_1$'])


if __name__ == '__main__':
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    k = (N-1)//2
    
    # f_stab_det = full_syst_stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    # isStable, eigv = f_stab_det(initial_vec)
    # print(f_stab_det(initial_vec))
    
    alpha2 = 2.523893
    epsilon2 = 0.08
    initial_xy_vec = np.array([0, -0.052839, 2.459533, 0.134690])
    T = 1000
    
    initial_vec = np.array([0, 0] + 
                           [initial_xy_vec[0], initial_xy_vec[1]]*k + 
                           [initial_xy_vec[2], initial_xy_vec[3]]*k)
    initial_vec = initial_vec + np.random.uniform(0, 1, 2*N)
    
    rhs = create_full_syst_func(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    arr_sol, arr_t = num_integration(rhs, initial_vec, T)
    
    # Drawing
    draw_examples_with_theta1(arr_sol, arr_t, N, T)