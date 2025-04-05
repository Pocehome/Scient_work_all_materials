import numpy as np
import matplotlib.pyplot as plt
import json


def point_okr(x, x0, err):
    return x0-err < x < x0+err


def draw_graph(X, Y, limits, x_name='', y_name='', 
               colors=[], legend=[], xs=[], ys=[], grid=True):
    
    x_lims, y_lims = limits
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    
    if len(X) == 1:
        for i in range(len(Y)):
            color = 'blue'
            if colors:
                color = colors[i]
            plt.plot(X[0], Y[i], color)
            
    elif len(X) == len(Y):
        for i in range(len(X)):
            color = 'blue'
            if colors:
                color = colors[i]
            plt.plot(X[i], Y[i], color)
    
    for x in xs:
        plt.axvline(x=x, color='black', linestyle='--')
    for y in ys:
        plt.axhline(y=y, color='black', linestyle='--')
    
    plt.xlabel(x_name, fontsize=10, color='black')
    plt.ylabel(y_name, fontsize=10, color='black')    
    plt.legend(legend, loc='upper right')
    if grid: plt.grid(True)
    
    plt.show()


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    epsilon2 = 0.08
    alpha2 = -2.0
    
    with open(f'Results/Reduced_N={N}_mu={mu:.2f}_'\
              f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_'\
              f'eps2={epsilon2:.5f}_alpha2={alpha2:.5f}.txt', 'r') as fr:
        params, arr_sol, arr_t, arr_R1, arr_R2 = json.load(fr)
        
    N, mu, epsilon1, alpha1, epsilon2, alpha2, step_n, step_size = params
    
    R1_for_draw = [0.] * step_n
    R2_for_draw = [0.] * step_n
    
    for i in range(step_n):
        R1_for_draw[i] = (arr_R1[i][0]**2 + arr_R1[i][1]**2)**0.5
        R2_for_draw[i] = (arr_R2[i][0]**2 + arr_R2[i][1]**2)**0.5
        
    tr_arr_sol = np.transpose(arr_sol)
    
    # graph for x and y by time
    # draw_graph([arr_t], [np.mod(tr_arr_sol[0], 2*np.pi), np.mod(tr_arr_sol[2], 2*np.pi)],
    #            [(step_n*step_size - 200, step_n*step_size), (-0.5, 2*np.pi + 0.5)],
    #            x_name='t', colors=['blue', 'red'], legend=['x(t)', 'y(t)'])
    draw_graph([arr_t], [np.mod(tr_arr_sol[0], 2*np.pi), np.mod(tr_arr_sol[2], 2*np.pi)],
               [(0, 200), (-0.5, 2*np.pi + 0.5)],
               x_name='t', colors=['blue', 'red'], legend=['x(t)', 'y(t)'])
    
    # graph for x and y derivatives by time
    max_xy_der = max(max(tr_arr_sol[1]), max(tr_arr_sol[3]))
    min_xy_der = min(min(tr_arr_sol[1]), min(tr_arr_sol[3]))
    # draw_graph([arr_t], [tr_arr_sol[1], tr_arr_sol[3]], 
    #            [(step_n*step_size - 200., step_n*step_size), (min_xy_der-0.5, max_xy_der+0.5)],
    #            x_name='t', colors=['blue', 'red'], legend=['\u1E8B(t)', '\u1E8F(t)'])
    draw_graph([arr_t], [tr_arr_sol[1], tr_arr_sol[3]], 
               [(0, 200), (min_xy_der-0.5, max_xy_der+0.5)],
               x_name='t', colors=['blue', 'red'], legend=['\u1E8B(t)', '\u1E8F(t)'])
    
    # graph for R1 and R2 by time
    draw_graph([arr_t], [R1_for_draw, R2_for_draw], 
               [(step_n*step_size-200., step_n*step_size), (-0.5, 1.5)],
               x_name='time', colors=['darkgreen', 'red'], legend=['R1', 'R2'])
    
    # graph for x and y derivatives by x and y
    # draw_graph([tr_arr_sol[0][100000:], tr_arr_sol[2][100000:]], 
    #            [tr_arr_sol[1][100000:], tr_arr_sol[3][100000:]],
    #            [(-170, -163), (-1, 0.5)], x_name='x, y', colors=['blue', 'red'],
    #            legend=['\u1E8B(x)', '\u1E8F(y)'])
    draw_graph([tr_arr_sol[0], tr_arr_sol[2]], 
               [tr_arr_sol[1], tr_arr_sol[3]],
               [(-20, 0), (-1, 0.5)], x_name='x, y', colors=['blue', 'red'],
               legend=['\u1E8B(x)', '\u1E8F(y)'])
    
    # i = 120000 - 1
    # i1 = 0
    # i2 = 0
    # while True:
    #     if point_okr(np.mod(arr_sol[i][0], 2*np.pi), 0, 0.001):
    #         print(np.mod(arr_sol[i][0], 2*np.pi), arr_sol[i][1],
    #               np.mod(arr_sol[i][2], 2*np.pi), arr_sol[i][3], '\n', i)
    #         if not(i1):
    #             i1 = i
    #         elif not(i2) and :
    #             i2 = i
    #             break
        
    #     i -= 1
    #     if i < 0:
    #         print('!!!')
    #         break
