import numpy as np
from numpy import cos
from system_dynamic import xy_dyn


def X_psi_RK4_step(X, arr_psi, F_X, F_psi, dt):
    # Вычисляем промежуточные значения для X
    k1_X = dt * F_X(dt, X)
    k2_X = dt * F_X(dt, X + 0.5*k1_X)
    k3_X = dt * F_X(dt, X + 0.5*k2_X)
    k4_X = dt * F_X(dt, X + k3_X)
    X_new = X + (k1_X + 2*k2_X + 2*k3_X + k4_X)/6
    
    # Вычисляем промежуточные значения для psi, используя X_new
    psi1, psi2, psi3, psi4 = arr_psi
    
    k1_psi1 = dt * F_psi(X_new, psi1)
    k2_psi1 = dt * F_psi(X_new, psi1 + 0.5*k1_psi1)
    k3_psi1 = dt * F_psi(X_new, psi1 + 0.5*k2_psi1)
    k4_psi1 = dt * F_psi(X_new, psi1 + k3_psi1)
    psi1_new = psi1 + (k1_psi1 + 2*k2_psi1 + 2*k3_psi1 + k4_psi1)/6
    
    k1_psi2 = dt * F_psi(X_new, psi2)
    k2_psi2 = dt * F_psi(X_new, psi2 + 0.5*k1_psi2)
    k3_psi2 = dt * F_psi(X_new, psi2 + 0.5*k2_psi2)
    k4_psi2 = dt * F_psi(X_new, psi2 + k3_psi2)
    psi2_new = psi2 + (k1_psi2 + 2*k2_psi2 + 2*k3_psi2 + k4_psi2)/6
    
    k1_psi3 = dt * F_psi(X_new, psi3)
    k2_psi3 = dt * F_psi(X_new, psi3 + 0.5*k1_psi3)
    k3_psi3 = dt * F_psi(X_new, psi3 + 0.5*k2_psi3)
    k4_psi3 = dt * F_psi(X_new, psi3 + k3_psi3)
    psi3_new = psi3 + (k1_psi3 + 2*k2_psi3 + 2*k3_psi3 + k4_psi3)/6
    
    k1_psi4 = dt * F_psi(X_new, psi4)
    k2_psi4 = dt * F_psi(X_new, psi4 + 0.5*k1_psi4)
    k3_psi4 = dt * F_psi(X_new, psi4 + 0.5*k2_psi4)
    k4_psi4 = dt * F_psi(X_new, psi4 + k3_psi4)
    psi4_new = psi4 + (k1_psi4 + 2*k2_psi4 + 2*k3_psi4 + k4_psi4)/6
    
    return X_new, [psi1_new, psi2_new, psi3_new, psi4_new]


def X_psi_num_integration(X0, F_X, arr0_psi, F_psi, T, step_size):
    X = X0
    # psi01, psi02, psi03, psi04 = arr0_psi
    arr_psi = arr0_psi
    step_n = int(T / step_size)
    
    arr_X = np.array([[0.] * 4] * (step_n+1))
    arr_X[0] = X0
    
    # arr_psi = np.array([[0.] * 4] * (step_n+1))
    # arr_psi[0] = psi0
    
    arr_t = np.array([0.] * (step_n+1))
    
    for k in range(1, step_n+1):
        X, arr_psi = X_psi_RK4_step(X, arr_psi, F_X, F_psi, step_size)
        arr_X[k] = X
        # arr_psi[k] = psi
        arr_t[k] = k * step_size
        
    return arr_X, arr_psi, arr_t


def da_dx(x, y, N, epsilon1, alpha1, epsilon2, alpha2):
    alpha = [0, alpha1, alpha2]
    epsilon = [0, epsilon1, epsilon2]
    _da_dx = 0
    for q in [1, 2]:
        _da_dx += epsilon[q] * (-q*cos(q*x+alpha[q]) - (N-1)/2 * q *
                                (cos(q*x-alpha[q]) + cos(q*(x-y)+alpha[q])))
    return _da_dx / N


def da_dy(x, y, N, epsilon1, alpha1, epsilon2, alpha2):
    alpha = [0, alpha1, alpha2]
    epsilon = [0, epsilon1, epsilon2]
    _da_dy = 0
    for q in [1, 2]:
        _da_dy += epsilon[q] * (-(N-1)/2 * q * (cos(q*y-alpha[q]) -
                                                cos(q*(x-y)+alpha[q])))
    return _da_dy / N


def db_dx(x, y, N, epsilon1, alpha1, epsilon2, alpha2):
    alpha = [0, alpha1, alpha2]
    epsilon = [0, epsilon1, epsilon2]
    _db_dx = 0
    for q in [1, 2]:
        _db_dx += epsilon[q] * (-(N-1)/2 * q * (cos(q*x-alpha[q]) -
                                                cos(q*(y-x)+alpha[q])))
    return _db_dx / N


def db_dy(x, y, N, epsilon1, alpha1, epsilon2, alpha2):
    alpha = [0, alpha1, alpha2]
    epsilon = [0, epsilon1, epsilon2]
    _db_dy = 0
    for q in [1, 2]:
        _db_dy += epsilon[q] * (-q*cos(q*y+alpha[q]) - (N-1)/2 * q *
                                (cos(q*y-alpha[q]) + cos(q*(y-x)+alpha[q])))
    return _db_dy / N

    
def psi_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    
    def RHS(X, psi):
        # res = np.array([0.] * 4)
        x, y = X[0], X[2]
        
        D = np.array([[0., 0., 0., 0.]] * 4)
        D[0] = np.array([0., 1., 0., 0.])
        D[1] = np.array([da_dx(x, y, N, epsilon1, alpha1, epsilon2, alpha2),
                         -1.,
                         da_dy(x, y, N, epsilon1, alpha1, epsilon2, alpha2),
                         0.]) / mu
        D[2] = np.array([0., 0., 0., 1.])
        D[3] = np.array([db_dx(x, y, N, epsilon1, alpha1, epsilon2, alpha2),
                         0.,
                         db_dy(x, y, N, epsilon1, alpha1, epsilon2, alpha2),
                         -1.]) / mu
        # print(D @ psi)
        return D @ psi
    
    return RHS


def stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    
    def f_stab_det(initial_vec):
        fundament_matrix = np.zeros((4, 4))
        arr_psi0 = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
        
        F_X = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
        F_psi = psi_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
        
        vec_for_int = np.array([0., initial_vec[0], initial_vec[1], initial_vec[2]])
        arr_X, arr_psi, arr_t = X_psi_num_integration(vec_for_int, F_X, arr_psi0, F_psi, initial_vec[3], 0.01)
        
        for i in range(4):
            fundament_matrix[:, i] = arr_psi[i]
            
        eigvals = np.linalg.eigvals(fundament_matrix)
        
        print(eigvals, '\n')
        # print()
        # print(fundament_matrix)
        # print()
        
        stability_err = 2*1e-3
        if np.all(abs(eigvals[1:]) < 1) and abs(eigvals[0]) < 1+stability_err:
            return True
        else:
            return False
        
    return f_stab_det


if __name__ == '__main__':
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    epsilon2 = 0.08
    alpha2 = -2.0
    initial_vec = np.array([-0.05921021, 2.64676814, 0.14678535, 41.02969191])
    
    # alpha2 = -np.pi
    # epsilon2 = 0.1
    # initial_vec = np.array([-0.0751, 2.2966, 0.0833, 42.0465])
    
    # alpha2 = -np.pi+0.1
    # epsilon2 = 0.1
    # initial_vec = np.array([-0.0782949, 2.29223005, 0.08001052, 41.93263796])
    
    # alpha2 = -3.141592653589793
    # epsilon2 = 0.19800000000000004
    # initial_vec = [-0.06868220471368051, 2.120648900481641, 0.02303488601578402, 54.98259987860226]
    
    f_stab_det = stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    print(f_stab_det(initial_vec))
    