import numpy as np
import matplotlib.pyplot as plt


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


def RK4_step(f, X, dt):
    k1 = dt * f(X)
    k2 = dt * f(X + k1 / 2)
    k3 = dt * f(X + k2 / 2)
    k4 = dt * f(X + k3)

    X_next = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X_next


def draw_graph(x, y, t, limits, der=False):
    xy_lims, t_lims = limits
    plt.ylim(xy_lims[0], xy_lims[1])
    plt.xlim(t_lims[0], t_lims[1])
    
    
    if der:
        plt.plot(t, x, 'blue', label='\u1E8B')
        plt.plot(t, y, 'red', label='\u1E8F')
    else:
        x = np.mod(x, 2*np.pi)
        y = np.mod(y, 2*np.pi)
        plt.plot(t, x, 'blue', label='x')
        plt.plot(t, y, 'red', label='y')
    
    plt.xlabel('time', fontsize=10, color='black')
    # plt.ylabel('', fontsize=10, color='black')
    plt.legend()
    plt.grid(True)
    
    plt.show()


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.0

    # initial values
    vec0 = np.array([0.0712587 - 0.28363, -0.169298, -2.14216 - 0.28363, -0.0698518])

    # numerical integration
    rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)

    step_n = 200000
    step_size = 0.01
    sol = vec0
    arr_sol = np.array([[0.] * 4] * step_n)
    arr_t = np.array([0.] * step_n)
    for k in range(step_n):
        sol = RK4_step(rhs, sol, step_size)
        for i in range(4):
            arr_sol[k][i] = sol[i]
        arr_t[k] = k * step_size
    
    # drawing graphs
    arr_sol = np.transpose(arr_sol)
    
    max_xy = max(max(arr_sol[0]), max(arr_sol[2]))
    min_xy = min(min(arr_sol[0]), min(arr_sol[2]))
    draw_graph(arr_sol[0], arr_sol[2], arr_t, 
               [(-0.5, 2*np.pi + 0.5), (0., step_n*step_size)])
    
    max_xy_der = max(max(arr_sol[1]), max(arr_sol[3]))
    min_xy_der = min(min(arr_sol[1]), min(arr_sol[3]))
    draw_graph(arr_sol[1], arr_sol[3], arr_t, 
               [(min_xy_der-0.5, max_xy_der+0.5), (step_n*step_size - 200, step_n*step_size)],
               der=True)
