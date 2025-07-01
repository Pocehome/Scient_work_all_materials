import numpy as np
import matplotlib.pyplot as plt
import json
from area_of_existing import write_to_file

# from monodromia_matrix import stability_determination


def draw_points(points, limits, x_name='', y_name='', grid=True):
    x_lims, y_lims = limits
    plt.xlim(x_lims[0], x_lims[1])
    plt.ylim(y_lims[0], y_lims[1])
    
    for point in points:
        if point[0] == 0 and point[1] == 0 and point[2] == 1:
            continue
        
        # if point[2] == 0:
        #     print(point)
        
        if point[2]:
            plt.plot(point[0], point[1], color='blue', marker='o', markersize=1)
        else:
            plt.plot(point[0], point[1], color='red', marker='o', markersize=1)
    
    plt.xlabel(x_name, fontsize=10, color='black')
    plt.ylabel(y_name, fontsize=10, color='black')
    if grid: plt.grid(True)
    
    plt.show()


def read_file(file_name):
    with open(file_name, 'r') as fr:
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
    write_to_file('one_line_test.txt', N, mu, eps1, alp1, step_n, line_exist)


if __name__ == "__main__":    
    # file_name = 'Results/Full_N=11_mu=1.00_omega=1.70_eps1=1.00000_alpha1=1.70000_eps2=0.08000_alpha2=-2.00000_stepn=120000.txt'
    # file_name = 'Results/Reduced_N=11_mu=1.00_eps1=1.00000_alpha1=1.70000_eps2=0.08000_alpha2=-2.00000.txt'
    # file_name = 'Area_exist_for_scient_work.txt'
    # file_name = 'area_stability_for_scient_work.txt'
    # file_name = 'test.txt'
    # file_name = 'Full_test.txt'
    # file_name = 'Full_test2.txt'
    # file_name = 'one_line_test.txt'
    file_name = 'one_line_test2.txt'
    
    N, mu, epsilon1, alpha1, area_step_n, area_existence = read_file(file_name)
    create_line(N, mu, epsilon1, alpha1, area_step_n, area_existence, 0.08)
        
    points = np.array([[0., 0., True]] * area_step_n**2)
    n = 0
    
    unstable_not_find = True
    
    for arr_eps in area_existence:
        for i, el in enumerate(arr_eps):
            if el != [0., 0., [0., 0., 0., 0.], True] and el[1] == 0.08:
                points[n] = [el[0], el[1], el[3]]
                
                # if n != 0:
                #     if points[n-1][2] != points[n][2]:
                #         print(arr_eps[i-3])
                #         print(arr_eps[i+3], '\n')
                
                # if el[0] == -2 and el[1] == 0.08:
                #     print(el)
                
                n += 1
    
    # arr_eps = area_existence[4]
    # for el in arr_eps:
    #     points[n] = [el[0], el[1]]
    #     n += 1
    
    draw_points(points, [(-np.pi, np.pi), (0, 0.2)],
               x_name='\u03B1\u2082', y_name='\u03B5\u2082')
