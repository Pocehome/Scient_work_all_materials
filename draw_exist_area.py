import numpy as np
import matplotlib.pyplot as plt
import json
from exist_rb_area import write_to_file

# from syst_without_reduc import full_syst_stability_determination


def draw_points(points, limits, x_name='', y_name='', grid=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_lims, y_lims = limits
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(y_lims[0], y_lims[1])
    
    for point in points:
        if point[0] == 0 and point[1] == 0 and point[2] == 1:
            continue
        
        if point[2]:
            ax.plot(point[0], point[1], color='blue', marker='s', markersize=2.5)
        else:
            ax.plot(point[0], point[1], color='red', marker='s', markersize=2.5)
    
    ax.set_xlabel(x_name, color='black')
    ax.set_ylabel(y_name, color='black')
    
    if grid:
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def read_file(dir_name, file_name):
    with open(dir_name + '/' + file_name, 'r') as fr:
        params, area_existence = json.load(fr)
        N, mu, epsilon1, alpha1, area_step_n = params
    return N, mu, epsilon1, alpha1, area_step_n, area_existence


def create_line(N, mu, eps1, alp1, step_n, area_exist, eps2):
    line_exist = 0
    for line in area_exist:
        for el in line:
            if el[1] == eps2:
                line_exist = [line]
                break
        if line_exist: break
    write_to_file('one_Full_line_test.txt', N, mu, eps1, alp1, step_n, line_exist)


if __name__ == "__main__":
    # Font settings
    rc_fonts = {
        'font.size': 20, # 14
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
    # file_name = 'Area_exist_for_scient_work.txt'
    # file_name = 'area_stability_for_scient_work.txt'
    # file_name = 'Full_test.txt'
    # file_name = 'Good results/Full_test2.txt'
    # file_name = 'one_line_test.txt'
    # file_name = 'one_Full_line_test.txt'
    
    dir_name = 'BreatherResults'
    file_name = 'area1.txt'
    # file_name = 'area2.txt'
    # file_name = 'area3.txt'
    
    # file_name = 'test.txt'
    
    N, mu, epsilon1, alpha1, area_step_n, area_existence = read_file(dir_name, file_name)
    # write_to_file(file_name, dir_name, N, mu, epsilon1, alpha1, area_step_n, area_existence)
    
    # create_line(N, mu, epsilon1, alpha1, area_step_n, area_existence, 0.08)
        
    points = np.array([[0., 0., True]] * area_step_n**2)
    n = 0
    
    unstable_not_find = True
    
    # arr_for_eigv = []
    
    flag = False
    for arr_eps in area_existence:
        for i, el in enumerate(arr_eps):
            if el != [0., 0., [0., 0., 0., 0.], True]: #and el[1] == 0.08:
                points[n] = [el[0], el[1], el[3]]
                
                # if 0.075 == el[1] and -1 < el[0] < -0.9 and not flag:
                #     flag = True
                #     print(el)
                
                n += 1
    
    # with open('arr_for_C_eigv.txt', 'w') as fw: json.dump(arr_for_eigv, fw)
    
    # arr_eps = area_existence[4]
    # for el in arr_eps:
    #     points[n] = [el[0], el[1]]
    #     n += 1
    
    draw_points(points, [(-np.pi, np.pi), (0, 0.2)],
                x_name=r'$\alpha_2$', y_name=r'$\varepsilon_2$')
