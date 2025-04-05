import numpy as np
import matplotlib.pyplot as plt
import json


def find_cyclop_i(Thetas, N):
    group_1 = [0., 0, 0]
    group_2 = [0., 0, 0]
    group_3 = [0., 0, 0]
    
    for i, theta in enumerate(Thetas):
        rd_theta = round(theta, 3)
        
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
    
    # print(group_1, group_2, group_3)
    return cyclop_i


def normalize_angle(angle):
    normalized = angle % (2 * np.pi)
    if normalized > np.pi:
        normalized -= 2 * np.pi
    return normalized


def draw_graph(X, N, t, limits, y_name='', colors=[], legend=[], ts=[], grid=True): 
    x_lims, t_lims = limits
    plt.ylim(x_lims[0], x_lims[1])
    plt.xlim(t_lims[0], t_lims[1])
    
    if colors:
       for i in range(N):
           plt.plot(t, X[i], colors[i], label=legend[i])
    else:
        for el in X:
            plt.plot(t, el, 'blue')
    
    if ts:
        for t in ts:
            plt.axvline(x=t, color='black', linestyle='--')
    
    plt.xlabel('time', fontsize=10, color='black')
    if y_name: plt.ylabel(y_name, fontsize=10, color='black')    
    if legend: plt.legend(legend, loc='upper right')
    if grid: plt.grid(True)
    
    plt.show()
    

def draw_mult(Thetas_t, Thetas_der_t, t, N):
    plt.title(f"t={t}")
    
    Thetas_t = np.mod(Thetas_t, 2 * np.pi) - np.pi
    
    plt.ylim(-1.25, 1.25)
    plt.xlim(-1.25, 1.25)
    circle = plt.Circle((0, 0), 1, color='black', linewidth=1, fill=False)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax=plt.gca()
    ax.add_patch(circle)
    
    cyclop_i = find_cyclop_i(Thetas_t, N)
    # print(cyclop_i, Thetas_t)
    
    r = 1
    for i in range(N):
        color='green'
        angle = normalize_angle(Thetas_t[i] - Thetas_t[cyclop_i])
        
        # color elems
        if angle == 0: # i == cyclop_i:
            color = '0.5'
        elif -np.pi < angle < 0:
            color = 'royalblue'
        elif 0 < angle < np.pi:
            color = 'indianred'
            
        plt.scatter(r*np.cos(angle + np.pi/2), r*np.sin(angle + np.pi/2),
                    s=300, c=color, marker='o', edgecolor='black', linewidth=0.5)
        r -= 0.02
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    
    plt.show()
    
    
if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    omega = 1.7
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.
    step_n = 120000
    
    with open(f'Results/Full_N={N}_mu={mu:.2f}_omega={omega:.2f}_'\
              f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_'\
              f'eps2={epsilon2:.5f}_alpha2={alpha2:.5f}_'\
              f'stepn={step_n}.txt', 'r') as fr:
        pars, arr_sol, arr_t, arr_R1, arr_R2 = json.load(fr)
        
    N, mu, omega, epsilon1, alpha1, epsilon2, alpha2, gamma, step_n, step_size = pars
    
    R1_for_draw = [0.] * step_n
    R2_for_draw = [0.] * step_n
    
    for i in range(step_n):
        R1_for_draw[i] = (arr_R1[i][0]**2 + arr_R1[i][1]**2)**0.5
        R2_for_draw[i] = (arr_R2[i][0]**2 + arr_R2[i][1]**2)**0.5

    tr_arr_sol = np.transpose(arr_sol)
    
    # last value of Thetas
    last_Thetas = arr_sol[-1][::2]
    last_Thetas = np.mod(last_Thetas, 2 * np.pi) - np.pi
    
    # cyclop element number
    cyclop_i = find_cyclop_i(last_Thetas, N)
    
    # Theta values
    Thetas_dyn = tr_arr_sol[::2]
    Thetas_dyn = np.mod(Thetas_dyn, 2*np.pi)
    
    # Theta derivative values
    Theta_ders_dyn = tr_arr_sol[1::2]
    Theta_ders_dyn = Theta_ders_dyn - Theta_ders_dyn[cyclop_i]
    
    # drawing graphs
    t1 = 1019
    t2 = 1032
    t3 = 1056
    
    draw_graph(Theta_ders_dyn, N, arr_t, 
               [(-1.5, 1.5), (step_n*step_size-200., step_n*step_size)],
               y_name='d\u03B8/dt')
    
    draw_graph([R1_for_draw, R2_for_draw], 2, arr_t,
               [(-0.5, 1.5), (step_n*step_size-200., step_n*step_size)],
               colors=['darkgreen', 'red'], legend=['R1', 'R2'], ts=[t1, t2, t3])
    
    # drawing mults
    draw_mult(arr_sol[int(t1/step_size)][::2], 
              arr_sol[int(t1/step_size)][1::2], t1, N)
    draw_mult(arr_sol[int(t2/step_size)][::2], 
              arr_sol[int(t2/step_size)][1::2], t2, N)
    draw_mult(arr_sol[int(t3/step_size)][::2],
              arr_sol[int(t3/step_size)][::2], t3, N)
