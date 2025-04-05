import numpy as np


def RK4_step(f, X, dt):
    k1 = dt * f(0, X)
    k2 = dt * f(0, X + k1 / 2)
    k3 = dt * f(0, X + k2 / 2)
    k4 = dt * f(0, X + k3)

    X_next = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X_next


def num_integration(rhs, sol0, T, step_size):
    sol = sol0
    step_n = int(T / step_size)
    arr_sol = np.array([[0.] * 4] * step_n)
    arr_t = np.array([0.] * step_n)
    
    for k in range(step_n):
        sol = RK4_step(rhs, sol, step_size)
        arr_sol[k] = sol
        arr_t[k] = k * step_size
        
    return arr_sol, arr_t