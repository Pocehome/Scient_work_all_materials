import numpy as np
import json


def xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    def RHS(Vec):
        res = np.array([0.] * 4)

        x = Vec[0]
        y = Vec[2]

        for_x_q1 = epsilon1 * (np.sin(alpha1) - np.sin(x + alpha1) -
                               (N - 1)/2 * (np.sin(x - alpha1) + np.sin(y - alpha1) +
                                            np.sin(alpha1) + np.sin(x - y + alpha1)))
        for_x_q2 = epsilon2 * (np.sin(alpha2) - np.sin(2*x + alpha2) -
                               (N - 1)/2 * (np.sin(2*x - alpha2) + np.sin(2*y - alpha2) +
                                            np.sin(alpha2) + np.sin(2*(x - y) + alpha2)))

        for_y_q1 = epsilon1 * (np.sin(alpha1) - np.sin(y + alpha1) -
                               (N - 1)/2 * (np.sin(x - alpha1) + np.sin(y - alpha1) +
                                            np.sin(alpha1) + np.sin(y - x + alpha1)))
        for_y_q2 = epsilon2 * (np.sin(alpha2) - np.sin(2*y + alpha2) -
                               (N - 1)/2 * (np.sin(2*x - alpha2) + np.sin(2*y - alpha2) +
                                            np.sin(alpha2) + np.sin(2*(y - x) + alpha2)))

        res[0] = Vec[1]
        res[1] = ((for_x_q1 + for_x_q2) / N - Vec[1]) / mu
        res[2] = Vec[3]
        res[3] = ((for_y_q1 + for_y_q2) / N - Vec[3]) / mu

        return res

    return RHS


def find_R1(N, x, y):
    k = (N - 1) / 2
    res = (k * np.exp(complex(0, 1) * x) +
           k * np.exp(complex(0, 1) * y) + 1) / N
    return res


def find_R2(N, x, y):
    k = (N - 1) / 2
    res = (k * np.exp(2 * complex(0, 1) * x) +
           k * np.exp(2 * complex(0, 1) * y) + 1) / N
    return res


def RK4_step(f, X, dt):
    
    global arr_R1, arr_R2, R_n
    x, y = X[::2]
    R1 = find_R1(N, x, y)
    R2 = find_R2(N, x, y)
    arr_R1[R_n] = [R1.real, R1.imag]
    arr_R2[R_n] = [R2.real, R2.imag]
    R_n += 1
    
    k1 = dt * f(X)
    k2 = dt * f(X + k1 / 2)
    k3 = dt * f(X + k2 / 2)
    k4 = dt * f(X + k3)

    X_next = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X_next


def num_integration(rhs, sol0, step_n, step_size):
    sol = sol0
    arr_sol = np.array([[0.] * 4] * step_n)
    arr_t = np.array([0.] * step_n)
    
    for k in range(step_n):
        sol = RK4_step(rhs, sol, step_size)
        arr_sol[k] = sol
        arr_t[k] = k * step_size
        
    return arr_sol, arr_t


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.0

    # initial values [x, x_der, y, y_der]
    vec0 = np.array([0.0712587 - 0.28363, -0.169298, -2.14216 - 0.28363, -0.0698518])

    # numerical integration
    rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)

    step_n = 120000
    step_size = 0.01
    
    arr_R1 = np.array([[0., 0.]] * step_n)
    arr_R2 = np.array([[0., 0.]] * step_n)
    R_n = 0
    
    arr_sol, arr_t = num_integration(rhs, vec0, step_n, step_size)
        
    # writing to file
    with open(f'Results/Reduced_N={N}_mu={mu:.2f}_'\
              f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_'\
              f'eps2={epsilon2:.5f}_alpha2={alpha2:.5f}.txt', 'w') as fw:
        json.dump([[N, mu, epsilon1, alpha1, epsilon2, alpha2, step_n, step_size],
                   arr_sol.tolist(), arr_t.tolist(),
                   arr_R1.tolist(), arr_R2.tolist()], fw)
