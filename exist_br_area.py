from find_br_traj import np, xy_dyn, get_xy_from_vec_i, find_br_init_vec
from syst_without_reduc import full_syst_stability_determination
import json


def br_stretching_by_alpha2(eps2, init_alp2, vec_in_init_alp2, alp2_step, area_step_n, init_x):
    # parameters
    N = 11
    mu = 1.0
    eps1 = 1.0
    alp1 = 1.7
    
    # area_existence = [alpha2, epsilon2, [x_der, y, y_der, T], is_stable]
    alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
    change_flag = False
    rotate_flag = False
    
    try:
        # right stretching by alpha2
        alp2 = init_alp2
        init_vec = vec_in_init_alp2
        while True:
            if alp2 > init_alp2 and rotate_flag:
                break
            if alp2 > np.pi:
                alp2 -= 2*np.pi
                rotate_flag = True
            
            # calculate functions
            rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
            calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2, init_x)
            f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2, init_x)
            
            # Newton's method
            init_vec, find_flag, is_stable, eigv = find_br_init_vec(init_vec, rhs, calc_xy, f_stab_det, init_x, True)
            
            if not find_flag:
                break
            
            # print([alp2, eps2, init_vec.tolist()], is_stable, '\n')
            print([round(alp2, 6), round(eps2, 6), 
                   [round(x, 6) for x in init_vec.tolist()]], is_stable, '\n')
            
            alpha2_area_existence[int(alp2/alp2_step)] = [alp2, eps2, init_vec.tolist(), is_stable]
            alp2 += alp2_step
            change_flag = True
        print("Right border\n")
        
        # left stretching by alpha2
        alp2 = init_alp2 - alp2_step
        init_vec = vec_in_init_alp2
        while True:
            if alp2 < -np.pi:
                if rotate_flag:
                    break
                else:
                    alp2 += 2*np.pi
                    rotate_flag = True
            
            # calculate functions
            rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
            calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2, init_x)
            f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2, init_x)
            
            # Newton's method
            init_vec, find_flag, is_stable, eigv = find_br_init_vec(init_vec, rhs, calc_xy, f_stab_det, init_x, True)
            
            if not find_flag:
                break
            
            # print([alp2, eps2, init_vec.tolist()], is_stable, '\n')
            print([round(alp2, 6), round(eps2, 6), 
                   [round(x, 6) for x in init_vec.tolist()]], is_stable, '\n')
            
            alpha2_area_existence[int(alp2/alp2_step)] = [alp2, eps2, init_vec.tolist(), is_stable]
            alp2 -= alp2_step
            change_flag = True
        print("Left border\n")    
    
        return alpha2_area_existence, change_flag
    
    except:
        print(f'\nError\neps2={eps2}\talpha2={alp2}\tinit_vec={init_vec}')
        return alpha2_area_existence, change_flag


def stretching_by_epsilon2(init_epsilon2, alpha2, vec_in_init_epsilon2, epsilon2_step, area_step_n):
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    # area_existence = [alpha2, epsilon2, [x_der, y, y_der, T], is_stable]
    epsilon2_area_existence = [0., 0., [0., 0., 0., 0.], True] * area_step_n
    find_flag = False
    
    try:
        # up streching by epsilon2
        epsilon2 = init_epsilon2
        init_vec = vec_in_init_epsilon2
        while True:
            if epsilon2 > 0.2:
                break
            
            # calculate functions
            rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            calc_xy = get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            
            # Newton's method
            init_vec, find_flag, is_stable = find_br_init_vec(init_vec, rhs, calc_xy)
            
            if not find_flag:
                break
            
            print([alpha2, epsilon2, init_vec.tolist()], is_stable, '\n')
            
            epsilon2_area_existence[int(epsilon2/epsilon2_step)] = [alpha2, epsilon2, init_vec.tolist(), is_stable]
            epsilon2 += epsilon2_step
            change_flag = True
        print("Up border\n")
        
        # down streching by epsilon2
        epsilon2 = init_epsilon2 - epsilon2_step
        init_vec = vec_in_init_epsilon2
        while True:
            if epsilon2 < 0:
                break
            
            # calculate functions
            rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            calc_xy = get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            
            # Newton's method
            init_vec, find_flag, is_stable = find_br_init_vec(init_vec, rhs, calc_xy)
            
            if not find_flag:
                break
            
            print([alpha2, epsilon2, init_vec.tolist()], is_stable, '\n')
            
            epsilon2_area_existence[int(epsilon2/epsilon2_step)] = [alpha2, epsilon2, init_vec.tolist(), is_stable]
            epsilon2 -= epsilon2_step
            change_flag = True
        print("Down border\n")
        
        return epsilon2_area_existence, change_flag
            
    except:
        print(f'\nError\neps2={epsilon2}\talpha2={alpha2}\tinit_vec={init_vec}')
        return epsilon2_area_existence, change_flag


def br_stretching_by_epsilon2_alpha2(init_eps2, alp2, vec_in_init_eps2, eps2_step, alp2_step, area_step_n, init_x):    
    # area_existence = [alpha2, epsilon2, [x_der, y, y_der, T], is_stable]
    alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
    area_existence = [alpha2_area_existence] * area_step_n
    
    try:
        # up streching by epsilon2
        eps2 = init_eps2
        init_vec = vec_in_init_eps2
        while True:
            if eps2 > 0.2:
                break
            
            alpha2_area_existence, change_flag = br_stretching_by_alpha2(eps2, alp2, init_vec, alp2_step, area_step_n, init_x)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[int(alp2/alp2_step)][2]
            area_existence[int(eps2/eps2_step)] = alpha2_area_existence
            eps2 += eps2_step
        print("Up border\n")
        
        # down streching by epsilon2
        eps2 = init_eps2 - eps2_step
        init_vec = vec_in_init_eps2
        while True:
            if eps2 < 0:
                break
            
            alpha2_area_existence, change_flag = br_stretching_by_alpha2(eps2, alp2, init_vec, alp2_step, area_step_n, init_x)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[int(alp2/alp2_step)][2]
            area_existence[int(eps2/eps2_step)] = alpha2_area_existence
            eps2 -= eps2_step
        print("Down border\n")
        
        return area_existence
            
    except:
        print(f'\nError\nepsilon2={eps2}\talpha2={alp2}\tinit_vec={init_vec}')
        return area_existence


def write_to_file(file_name, dir_name, N, mu, eps1, alp1, area_step_n, area_existence):
    
    def filter_area_arr(arr):
        def format_item(item):
            return [round(item[0], 6), round(item[1], 6),
                    [round(x, 6) for x in item[2]], item[3]]
        
        filtered_arr = []
        for subarr in arr:
            filtered_subarr = []
            for item in subarr:
                if item == [0., 0., [0., 0., 0., 0.], True]:
                    continue
                filtered_subarr.append(format_item(item))
                
            if filtered_subarr:
                filtered_arr.append(filtered_subarr)
        
        return filtered_arr
    
    with open(dir_name + '/' + file_name, 'w') as fw:
        json.dump([[N, mu, eps1, alp1, area_step_n], filter_area_arr(area_existence)], fw)


if __name__ == "__main__":
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    area_step_n = 100
    
    dir_name = 'BreatherResults'
    # file_name = 'test.txt'
    # file_name = 'area1.txt'
    
    file_name = 'area2.txt'
    # file_name = 'area3.txt'
    # file_name = 'area4.txt'
        
    try:
        # Area settings
        alpha2_step = 2*np.pi / area_step_n
        epsilon2_step = 0.2 / area_step_n
        alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
        epsilon2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
        area_existence = [alpha2_area_existence] * area_step_n
        
        
        # Initial conditions
        # init_epsilon2 = 0.075
        # init_alpha2 = 0.
        # initial_vec = np.array([0.164135, -1.199126, 0.093172, 28.888474])
        # initial_x = 2.
        
        init_epsilon2 = 0.075
        init_alpha2 = -0.942478
        initial_vec = np.array([0.34002, -0.947954, 0.224485, 27.140181])
        initial_x = 2.
        
        
        # Calc type
        # alpha2_area_existence, flag = br_stretching_by_alpha2(init_epsilon2, init_alpha2, initial_vec, alpha2_step, area_step_n, initial_x)
        # area_existence[0] = alpha2_area_existence
        
        # epsilon2_area_existence, flag = stretching_by_epsilon2(init_epsilon2, init_alpha2, initial_vec, epsilon2_step, area_step_n, initial_x)
        
        area_existence = br_stretching_by_epsilon2_alpha2(init_epsilon2, init_alpha2, initial_vec, epsilon2_step, alpha2_step, area_step_n, initial_x)
        
        
        # Writing to file        
        write_to_file(file_name, dir_name, N, mu, epsilon1, alpha1, area_step_n, area_existence)
    
    except:
        # Writing to file        
        write_to_file(file_name, dir_name, N, mu, epsilon1, alpha1, area_step_n, area_existence)
