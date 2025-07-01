import numpy as np
# from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import acos


def find_R1(N, Theta):
    res = np.exp(complex(0, 1) * Theta)
    return 1 / N * sum(res)


def find_R2(N, Theta):
    res = np.exp(2 * complex(0, 1) * Theta)
    return 1 / N * sum(res)


def find_Theta0(gamma, N):
    M = int((N + 1) / 2)
    res = np.array([-gamma] * (M - 1) + [0.] + [gamma] * (M - 1))
    res += np.random.uniform(-0.1, 0.1, 11)
    return res


def f1(Theta_k, epsilon1, alpha1, epsilon2, alpha2, R1, R2):
    res = (epsilon1 * R1 * np.exp(complex(0, 1) * (-Theta_k - alpha1)) +
           epsilon2 * R2 * np.exp(complex(0, 1) * (-2 * Theta_k - alpha2))).imag
    return res


def f2(N, mu, omega, epsilon1, alpha1, epsilon2, alpha2):
    def RHS(X):
        res = np.array([0.] * 2 * N)

        Theta = X[::2]
        # Y = X[1::2]

        R1 = find_R1(N, Theta)
        R2 = find_R2(N, Theta)

        for i in range(N):
            res[2 * i] = X[2 * i + 1]
            res[2 * i + 1] = 1 / mu * (omega - X[2 * i + 1] +
                                       f1(X[2 * i], epsilon1, alpha1, epsilon2, alpha2, R1, R2))

        return res

    return RHS


def RK4_step(f, X, dt):
    k1 = dt * f(X)
    k2 = dt * f(X + k1 / 2)
    k3 = dt * f(X + k2 / 2)
    k4 = dt * f(X + k3)

    X_next = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X_next


def eq_quiver(rhs, limits, N=16):
    fi_lims, y_lims = limits
    fis = np.linspace(fi_lims[0], fi_lims[1], N)
    ys = np.linspace(y_lims[0], y_lims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, fi in enumerate(fis):
            vfield = rhs(np.array([fi, y]))
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return fis, ys, U, V


def plot_plane(rhs, limits):
    fi_lims, y_lims = limits
    plt.xlim(fi_lims[0], fi_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    fi_vec, y_vec, U, V = eq_quiver(rhs, limits)
    plt.quiver(fi_vec, y_vec, U, V, alpha=0.8)
    
    
def draw_graph(X, N, t, limits):
    x_lims, t_lims = limits
    plt.ylim(x_lims[0], x_lims[1])
    plt.xlim(t_lims[0], t_lims[1])
    
    for el in X:
        plt.plot(t, el, 'blue')
    
    plt.xlabel('time', fontsize=10, color='black')
    # plt.ylabel('', fontsize=10, color='black')
    # plt.legend()
    plt.grid(True)
    
    plt.show()
    

def find_cyclop_i(last_Thetas, N):
    group_1 = [0., 0, 0]
    group_2 = [0., 0, 0]
    group_3 = [0., 0, 0]
    
    for i, theta in enumerate(last_Thetas):
        rd_theta = round(theta, 2)
        
        if group_1[1] == 0 or group_1[0] == rd_theta:
            group_1[0] = rd_theta
            group_1[1] += 1
            group_1[2] = i
            
        elif group_2[1] == 0 or group_2[0] == rd_theta:
            group_2[0] = rd_theta
            group_2[1] += 1
            group_2[2] = i
            
        elif group_3[1] == 0 or group_3[0] == rd_theta:
            group_3[0] = rd_theta
            group_3[1] += 1
            group_3[2] = i
        
    if group_1[1] == 1:
        cyclop_i = group_1[2]
    if group_2[1] == 1:
        cyclop_i = group_2[2]
    if group_3[1] == 1:
        cyclop_i = group_3[2]
    
    print(group_1, group_2, group_3)
    return cyclop_i


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    omega = 1.7
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.
    gamma = acos(1 / (1 - N))
    
    # initial values
    Theta0 = find_Theta0(gamma, N)
    Y0 = np.array([1.] * N)
    
    vec0 = np.array([0.] * 2 * N)
    for i in range(N):
        vec0[2 * i] = float(Theta0[i])
        vec0[2 * i + 1] = Y0[i]
    
    # numerical integration
    rhs = f2(N, mu, omega, epsilon1, alpha1, epsilon2, alpha2)
    
    step_n = 20000   # 200000
    step_size = 0.01
    sol = vec0
    arr_sol = np.array([[0.] * 2 * N] * step_n)
    arr_t = np.array([0.] * step_n)
    
    for k in range(step_n):
        sol = RK4_step(rhs, sol, step_size)
        arr_sol[k] = sol
        arr_t[k] = k * step_size
    
    # drawing graphs
    tr_arr_sol = np.transpose(arr_sol)
    
    # for Theta derivates
    last_Thetas = arr_sol[-1][::2]
    last_Thetas = np.mod(last_Thetas, 2 * np.pi) - np.pi
    cyclop_i = find_cyclop_i(last_Thetas, N)
    
    Theta_ders = tr_arr_sol[1::2]
    Theta_ders = Theta_ders - Theta_ders[cyclop_i]
    
    draw_graph(Theta_ders, N, arr_t, 
               [(-1.5, 1.5), (step_n*step_size-200., step_n*step_size)])
