import numpy as np
import matplotlib.pyplot as plt
import json

from monodromia_matrix import stability_determination


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
    
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    area_step = 0.02
    
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
        # a = json.load(fr)
        
    points = np.array([[0., 0., True]] * ((int(2*np.pi/area_step) + 1) * (int(0.2/area_step) + 1)))
    n = 0
    
    for arr_eps in area_existence:
        for el in arr_eps:
            # if 0.008 < el[1] < 0.012:
            points[n] = [el[0], el[1], el[3]]
            
            # if el[3] == 0:
            #     st_det = stability_determination(N, mu, epsilon1, alpha1, el[1], el[0])
            #     print(st_det(el[2]))
                # print(el, '\n')
                
            if 0.157 < el[1] < 0.164 and -1.63 < el[0] < 0:
                print(el, '\n')
                
                st_det = stability_determination(N, mu, epsilon1, alpha1, el[1], el[0])
                st_det(el[2])
                
                # st_det = stability_determination(N, mu, epsilon1, alpha1, el[1], el[0]+0.01)
                # st_det(el[2])
                
            n += 1
    
    # arr_eps = area_existence[4]
    # for el in arr_eps:
    #     points[n] = [el[0], el[1]]
    #     n += 1
    
    draw_points(points, [(-np.pi, np.pi), (0, 0.2)],
               x_name='\u03B1\u2082', y_name='\u03B5\u2082')
