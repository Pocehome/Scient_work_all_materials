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


def num_integration(rhs, sol0, T):
    sol = solve_ivp(rhs, [0, T], sol0, max_step=0.01)
    # arr_sol, arr_t = np.transpose(sol.y), sol.t
    arr_sol, arr_t = sol.y, sol.t
        
    return arr_sol, arr_t


def mod_2pi(arr):
    return np.mod(np.array(arr) - np.pi, 2*np.pi) - np.pi


def draw_graph(X, Y, limits, x_name='', y_name='', visible=1, 
               colors=[], legend=[], xs=[], ys=[], grid=False, linewidth=5.0, draw_theta=False, do_y_lims=False):
    # Figure size
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Limits
    x_lims, y_lims = limits
    ax.set_xlim(x_lims[0], x_lims[1])
    
    if draw_theta:
        ax.set_ylim(y_lims[0], y_lims[1])
        
        # Tick labels for theta_k
        y_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        y_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)
    # else:
    #     y_ticks = np.arange(np.ceil(y_lims[0]/0.1)*0.1, y_lims[1], 0.1)
    #     ax.set_yticks(y_ticks)
    #     ax.set_ylabel('t', fontsize=30)
    
    if do_y_lims:
        ax.set_ylim(y_lims[0], y_lims[1])
        
    
    # Tick labels for t - every 50
    x_ticks = np.arange(np.ceil(x_lims[0]/50)*50, x_lims[1]+50, 50)
    ax.set_xticks(x_ticks)
    ax.set_xlabel('t', fontsize=30)

    # Tick label's fontsize
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    ax.grid(False)
        
    # alp2, eps2 fontsize
    ax.set_xlabel(x_name, color='black', fontsize=30)
    ax.set_ylabel(y_name, color='black', fontsize=30)
    
    if len(X) == 1:
        for i in range(len(Y)):
            try:
                ax.plot(X[0], Y[i], colors[i], alpha=visible, linewidth=linewidth)
            except:
                ax.plot(X[0], Y[i], alpha=visible, linewidth=linewidth)
                
    elif len(X) == len(Y):
        for i in range(len(X)):
            try:
                ax.plot(X[i], Y[i], colors[i], alpha=visible, linewidth=linewidth)
            except:
                ax.plot(X[i], Y[i], alpha=visible, linewidth=linewidth)
    
    for x in xs:
        ax.axvline(x=x, color='black', linestyle='--')
    for y in ys:
        ax.axhline(y=y, color='black', linestyle='--')
    
    ax.set_xlabel(x_name, color='black')
    ax.set_ylabel(y_name, color='black')
    
    if legend:
        ax.legend(legend, loc='upper right', fontsize=30)
    
    if grid:
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def draw_start_end(arr_sol, arr_t, _y_name, T,
                   ex_legend=[],
                   tspan=False, draw_start=False, draw_end=False,
                   T_inter=150, y_lims=(-np.pi-0.5, np.pi+0.5),
                   draw_theta=False, do_y_lims=False):
    if T <= T_inter:
        draw_graph([arr_t], arr_sol,
                   [(0, T), y_lims],
                   x_name='t', y_name=_y_name,
                   colors=['black'], legend=ex_legend,
                   draw_theta=draw_theta, do_y_lims=do_y_lims)
    
    else:
        if draw_start:
            draw_graph([arr_t], arr_sol,
                       [(0, 0+T_inter), y_lims],
                       x_name='t', y_name=_y_name,
                       colors=['black'], legend=ex_legend,
                       draw_theta=draw_theta, do_y_lims=do_y_lims)
    
        if draw_end:
            draw_graph([arr_t], arr_sol,
                       [(T-T_inter, T), y_lims],
                       x_name='t', y_name=_y_name,
                       colors=['black'], legend=ex_legend,
                       draw_theta=draw_theta, do_y_lims=do_y_lims)
            
        if tspan:
            draw_graph([arr_t], arr_sol,
                       [tspan, y_lims],
                       x_name='t', y_name=_y_name,
                       colors=['black'], legend=ex_legend,
                       draw_theta=draw_theta, do_y_lims=do_y_lims)


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