import numpy as np
from numpy import sin
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    
    def RHS(t, Vec):
        res = np.array([0.] * 4)

        x = Vec[0]
        y = Vec[2]
        
        alpha = [0, alpha1, alpha2]
        epsilon = [0, epsilon1, epsilon2]
        
        a = 0
        b = 0
        for q in [1, 2]:
            a += epsilon[q] * (sin(alpha[q]) - sin(q*x+alpha[q]) - (N-1)/2 *
                               (sin(q*x-alpha[q]) + sin(q*y-alpha[q]) +
                                sin(alpha[q]) + sin(q*(x-y)+alpha[q])))
            
            b += epsilon[q] * (sin(alpha[q]) - sin(q*y+alpha[q]) - (N-1)/2 *
                               (sin(q*x-alpha[q]) + sin(q*y-alpha[q]) +
                                sin(alpha[q]) + sin(q*(y-x)+alpha[q])))

        res[0] = Vec[1]
        res[1] = (a/N - Vec[1]) / mu
        res[2] = Vec[3]
        res[3] = (b/N - Vec[3]) / mu

        return res

    return RHS


# !!!
def num_integration(rhs, sol0, T):
    sol = solve_ivp(rhs, [0, T], sol0, max_step=0.01)
    # arr_sol, arr_t = np.transpose(sol.y), sol.t
    arr_sol, arr_t = sol.y, sol.t
        
    return arr_sol, arr_t
# !!!


def draw_graph(X, Y, limits, x_name='', y_name='', visible=1, 
               colors=[], legend=[], xs=[], ys=[], grid=True):
    
    x_lims, y_lims = limits
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    
    if len(X) == 1:
        for i in range(len(Y)):
            try:
                plt.plot(X[0], Y[i], colors[i], alpha=visible)
            except:
                plt.plot(X[0], Y[i], alpha=visible)
                
    elif len(X) == len(Y):
        for i in range(len(X)):
            try:
                plt.plot(X[i], Y[i], colors[i], alpha=visible)
            except:
                plt.plot(X[i], Y[i], alpha=visible)
    
    for x in xs:
        plt.axvline(x=x, color='black', linestyle='--')
    for y in ys:
        plt.axhline(y=y, color='black', linestyle='--')
    
    plt.xlabel(x_name, fontsize=10, color='black')
    plt.ylabel(y_name, fontsize=10, color='black')    
    plt.legend(legend, loc='upper right')
    if grid: plt.grid(True)
    
    plt.show()


if __name__ == '__main__':
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    # alpha2 = -np.pi
    # epsilon2 = 0.1
    # initial_vec = np.array([0., -0.0751, 2.2966, 0.0833])
    # T = 10.
    
    epsilon2 = 0.08
    alpha2 = -2.0
    initial_vec = np.array([0., -0.05921021, 2.64676814, 0.14678535])
    # T = 41.02969191
    T = 100.
    
    # numerical integration
    
    rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    
    arr_sol, arr_t = num_integration(rhs, initial_vec, T)
    # tr_arr_sol = np.transpose(arr_sol)

    # draw graph for x and y by time
    draw_graph([arr_t], [np.mod(arr_sol[0], 2*np.pi) - np.pi,
                         np.mod(arr_sol[2], 2*np.pi) - np.pi],
               [(T-200, T), (-np.pi-0.5, np.pi+0.5)],
               x_name='t', colors=['blue', 'red'], legend=['x(t)', 'y(t)'])
    
    # darw graph for x and y derivatives by time
    max_xy_der = max(max(arr_sol[1]), max(arr_sol[3]))
    min_xy_der = min(min(arr_sol[1]), min(arr_sol[3]))
    
    draw_graph([arr_t], [arr_sol[1], arr_sol[3]],
               [(T-200, T), (min_xy_der-0.5, max_xy_der+0.5)],
               x_name='t', colors=['blue', 'red'], legend=['\u1E8B(t)', '\u1E8F(t)'])