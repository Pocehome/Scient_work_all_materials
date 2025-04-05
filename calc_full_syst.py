import numpy as np
import json


def find_R1(N, Theta):
    res = np.exp(complex(0, 1) * Theta)
    return sum(res) / N


def find_R2(N, Theta):
    res = np.exp(2 * complex(0, 1) * Theta)
    return sum(res) / N


def find_Theta0(gamma, N):
    M = int((N + 1) / 2)
    res = np.array([-gamma] * (M - 1) + [0.] + [gamma] * (M - 1))
    # res += np.random.uniform(-0.1, 0.1, 11)
    return res


def f1(Theta_k, epsilon1, alpha1, epsilon2, alpha2, R1, R2):
    res = (epsilon1 * R1 * np.exp(complex(0, 1) * (-Theta_k - alpha1)) +
           epsilon2 * R2 * np.exp(complex(0, 1) * (-2 * Theta_k - alpha2))).imag
    return res


def f2(N, mu, omega, epsilon1, alpha1, epsilon2, alpha2):
    
    def RHS(X):
        res = np.array([0.] * 2*N)

        Theta = X[::2]
        # Y = X[1::2]

        R1 = find_R1(N, Theta)
        R2 = find_R2(N, Theta)

        for i in range(N):
            res[2*i] = X[2*i + 1]
            res[2*i + 1] = (omega - X[2*i + 1] + 
                            f1(X[2*i], epsilon1, alpha1, epsilon2, alpha2, R1, R2)) / mu 

        return res

    return RHS


def RK4_step(f, X, dt):
    
    global arr_R1, arr_R2, R_n
    Theta = X[::2]
    R1 = find_R1(N, Theta)
    R2 = find_R2(N, Theta)
    arr_R1[R_n] = [R1.real, R1.imag]
    arr_R2[R_n] = [R2.real, R2.imag]
    R_n += 1
    
    k1 = dt * f(X)
    k2 = dt * f(X + k1/2)
    k3 = dt * f(X + k2/2)
    k4 = dt * f(X + k3)

    X_next = X + (k1 + 2*k2 + 2*k3 + k4)/6
    return X_next


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    omega = 1.7
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.
    gamma = np.arccos(1 / (1 - N))
    
    # initial values
    Theta0 = find_Theta0(gamma, N)
    Y0 = np.array([1.] * N)
    print(Theta0)
    
    vec0 = np.array([0.] * 2*N)
    for i in range(N):
        vec0[2*i] = float(Theta0[i])
        vec0[2*i + 1] = Y0[i]
    
    # numerical integration
    rhs = f2(N, mu, omega, epsilon1, alpha1, epsilon2, alpha2)
    
    step_n = 12000   # 120000
    step_size = 0.01
    sol = vec0
    arr_sol = np.array([[0.] * 2*N] * step_n)
    arr_t = np.array([0.] * step_n)
    
    arr_R1 = np.array([[0., 0.]] * step_n)
    arr_R2 = np.array([[0., 0.]] * step_n)
    R_n = 0
    
    for k in range(step_n):
        sol = RK4_step(rhs, sol, step_size)
        arr_sol[k] = sol
        arr_t[k] = k * step_size
    
    # writing to file
    with open(f'Results/Full_N={N}_mu={mu:.2f}_omega={omega:.2f}_'\
              f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_'\
              f'eps2={epsilon2:.5f}_alpha2={alpha2:.5f}_'\
              f'stepn={step_n}.txt', 'w') as fw:
        json.dump([[N, mu, omega, epsilon1, alpha1, epsilon2, alpha2, gamma, step_n, step_size],
                   arr_sol.tolist(), arr_t.tolist(),
                   arr_R1.tolist(), arr_R2.tolist()], fw)
