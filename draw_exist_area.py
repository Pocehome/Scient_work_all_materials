import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json

from exist_br_area import write_to_file

# from syst_without_reduc import full_syst_stability_determination


def draw_points(points, limits, x_name='', y_name='', state_settings=False, extra_points=[]):
    # Plot settings
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True)
    
    x_lims, y_lims = limits
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    x_len = x_lims[1] - x_lims[0]
    y_len = y_lims[1] - y_lims[0]
    
    
    # Font settings
    if state_settings:
        # Tick labels for eps
        if y_lims[1] == 0.3:
            ax.set_yticks([0, 0.1, 0.2, 0.3])
        else:
            ax.set_yticks([0, 0.1, 0.2])
    
        # Tick labels for alp2
        x_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        x_ticklabels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)
    
        # Tick label's fontsize
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        ax.grid(False)
        
    # alp2, eps2 fontsize
    ax.set_xlabel(x_name, color='black', fontsize=30)
    ax.set_ylabel(y_name, color='black', fontsize=30)
        
    
    # Drawing
    for point in points:
        # if point[0] == 0. and point[1] == 0. and point[2] == 1:
        #     continue
        
        if point[2]:
            ax.plot(point[0], point[1], color='royalblue', marker='s', markersize=2.5)
            continue
        else:
            ax.plot(point[0], point[1], color='tomato', marker='s', markersize=2.5) 
            continue  
        
    numbers = ['A', 'B', 'C', 'D']
    for i, point in enumerate(extra_points):
        ax.plot(point[0], point[1], color='black', marker='s', markersize=4)
        ax.text(point[0] + 0.01*x_len, point[1] + 0.02*y_len, 
                r'\textbf{' + numbers[i] + '}', fontsize=24)
    
    plt.tight_layout()
    plt.show()


def read_file(dir_name, file_name):
    with open(dir_name + '/' + file_name, 'r') as fr:
        params, area_existence = json.load(fr)
        try:
            N, mu, epsilon1, alpha1 = params
        except:
            N, mu, epsilon1, alpha1, area_step_n = params
            
    return N, mu, epsilon1, alpha1, area_existence


def create_line(N, mu, eps1, alp1, step_n, area_exist, eps2):
    line_exist = 0
    for line in area_exist:
        for el in line:
            if el[1] == eps2:
                line_exist = [line]
                break
        if line_exist: break
    write_to_file('one_Full_line_test.txt', 'RotobreatherResults',  N, mu, eps1, alp1, step_n, line_exist)


if __name__ == "__main__":
    # Font settings
    rc_fonts = {
        'font.size': 22,
        "text.usetex": True,
        'mathtext.default': 'regular',
        'text.latex.preamble': r"\usepackage{bm}",
        "font.family": "serif",
        "font.serif": "computer modern roman",
    }
    plt.rcParams.update(rc_fonts)
    
    
    # dir_name = 'GoodResults'
    # file_name = 'Reduced_area_exist_N=11_mu=1.00_eps1=1.00000_alpha1=1.70000_stepN=250.txt'
    
    # dir_name = 'RotobreatherResults'
    # file_name = 'alpha1=1.7.txt'
    # file_name = 'one_Full_line_test.txt'
    # file_name = 'test.txt'
    # file_name = 'test2.txt'
    # file_name = 'comb_area_alpha1=1.7.txt'
    
    dir_name = 'BreatherResults'
    # file_name = 'area1.txt'
    # file_name = 'area2.txt'
    # file_name = 'area3.txt'
    file_name = 'br_alp1=1.7_all_areas.txt'
    
    # file_name = 'test.txt'
    
    N, mu, epsilon1, alpha1, area_existence = read_file(dir_name, file_name)
    # write_to_file('copy_ar2.txt', dir_name, [N, mu, epsilon1, alpha1], area_existence)
    # create_line(N, mu, epsilon1, alpha1, area_step_n, area_existence, 0.08)
    
    
    # Creation points arr
    points = []    
    flag = False
    for arr_eps in area_existence:
        for i, el in enumerate(arr_eps):
            # if el == [2.764601, 0.059, [0.268572, -0.458804, 0.125202, 34.751747], True]:
            if np.pi-0.5 < el[0] < np.pi-0.35 and 0.04 < el[1] < 0.05 and el[3]:
                points.append([el[0], el[1], el[3]])
                flag = True
                print(el)
            
            # points.append([el[0], el[1], el[3]])
    
    
    # with open('arr_for_C_eigv.txt', 'w') as fw: json.dump(arr_for_eigv, fw)
    
    # arr_eps = area_existence[4]
    # for el in arr_eps:
    #     points[n] = [el[0], el[1]]
    #     n += 1
    
    eps2_max = 0.2
    # eps2_max = 0.3
    
    examples = []
    examples = [(-0.377, 0.049), (-1.32, 0.1494), (-0.867, 0.13)]
    # examples = [(-2.704, 0.08), (2.65, 0.08), (-1.974867, 0.2)]
    
    draw_points(points, [(-np.pi, np.pi), (0, eps2_max)],
                x_name=r'$\alpha_2$', y_name=r'$\varepsilon_2$',
                state_settings=True, extra_points=examples)
                
    # draw_filled_areas(points, [(-np.pi, np.pi), (0, 0.2)],
    #                   x_name=r'$\alpha_2$', y_name=r'$\varepsilon_2$',
    #                   state_settings=True)