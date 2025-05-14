import numpy as np
import matplotlib.pyplot as plt
import json

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
            plt.plot(point[0], point[1], color='blue', marker='s', markersize=2.5)
        else:
            plt.plot(point[0], point[1], color='red', marker='s', markersize=2.5)
    
    plt.xlabel(x_name, fontsize=10, color='black')
    plt.ylabel(y_name, fontsize=10, color='black')
    if grid: plt.grid(True)
    
    plt.show()


if __name__ == "__main__":    
    # file_name = 'Results/Full_N=11_mu=1.00_omega=1.70_eps1=1.00000_alpha1=1.70000_eps2=0.08000_alpha2=-2.00000_stepn=120000.txt'
    # file_name = 'Results/Reduced_N=11_mu=1.00_eps1=1.00000_alpha1=1.70000_eps2=0.08000_alpha2=-2.00000.txt'
    # file_name = 'Area_exist_for_scient_work.txt'
    # file_name = 'area_stability_for_scient_work.txt'
    # file_name = 'test.txt'
    file_name = 'Full_test.txt'
    # file_name = 'Full_test2.txt'
    # file_name = 'one_line_test.txt'
    
    with open(file_name, 'r') as fr:
        params, area_existence = json.load(fr)
        N, mu, epsilon1, alpha1, area_step_n = params
        
    points = np.array([[0., 0., True]] * area_step_n**2)
    n = 0
    
    unstable_not_find = True
    
    for arr_eps in area_existence:
        for el in arr_eps:
            if el != [0., 0., [0., 0., 0., 0.], True]:
                points[n] = [el[0], el[1], el[3]]
                
                # if el[3] and unstable_not_find:
                # print(el)
                    # unstable_not_find = False
                
                n += 1
    
    # arr_eps = area_existence[4]
    # for el in arr_eps:
    #     points[n] = [el[0], el[1]]
    #     n += 1
    
    draw_points(points, [(-np.pi, np.pi), (0, 0.2)],
               x_name='\u03B1\u2082', y_name='\u03B5\u2082')
