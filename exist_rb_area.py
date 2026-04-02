import numpy as np
import json

from find_rb_traj import xy_dyn, get_xy_from_vec_i, find_rb_init_vec
from syst_without_reduc import full_syst_stability_determination


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
            init_vec, find_flag, is_stable = find_rb_init_vec(init_vec, rhs, calc_xy)
            
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
            init_vec, find_flag, is_stable = find_rb_init_vec(init_vec, rhs, calc_xy)
            
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


def rb_stretching_by_alpha2(params, eps2, init_alp2, vec_in_init_alp2, alp2_step, period_len):
    # Parameters
    N, mu, eps1, alp1 = params
    
    
    change_flag = False
    rotate_flag = False
    right_strech_end = -2*np.pi
    
    
    # Streching
    alpha2_area_existence = []
    try:
        # Right stretching by alpha2
        alp2 = init_alp2
        init_vec = vec_in_init_alp2
        while True:
            if alp2 > init_alp2 and rotate_flag:
                break
            
            elif alp2 > np.pi:
                alp2 -= 2*np.pi
                rotate_flag = True
                
            right_strech_end = alp2
            
            # Calculate functions
            rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
            calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2)
            f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
            
            # Newton's method
            init_vec, find_flag, is_stable, eigv = find_rb_init_vec(init_vec, rhs, calc_xy, f_stab_det, period_len, True)
            
            if not find_flag:
                break
            
            # print([alp2, eps2, init_vec.tolist()], is_stable, '\n')
            print([round(alp2, 6), round(eps2, 6), 
                   [round(x, 6) for x in init_vec.tolist()]], is_stable, '\n')
            
            alpha2_area_existence.append([alp2, eps2, init_vec.tolist(), is_stable])
            alp2 += alp2_step
            change_flag = True
        print("Right border\n")
        
        
        # Left stretching by alpha2
        alp2 = init_alp2 - alp2_step
        init_vec = vec_in_init_alp2
        while True:
            if init_alp2 < alp2 < right_strech_end:
                break
            
            elif alp2 < right_strech_end < init_alp2:
                break
            
            elif alp2 < -np.pi and not rotate_flag:
                alp2 += 2*np.pi
                rotate_flag = True
            
            # Calculate functions
            rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
            calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2)
            f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
            
            # Newton's method
            init_vec, find_flag, is_stable, eigv = find_rb_init_vec(init_vec, rhs, calc_xy, f_stab_det, period_len, True)
            
            if not find_flag:
                break
            
            # print([alp2, eps2, init_vec.tolist()], is_stable, '\n')
            print([round(alp2, 6), round(eps2, 6), 
                   [round(x, 6) for x in init_vec.tolist()]], is_stable, '\n')
            
            alpha2_area_existence.append([alp2, eps2, init_vec.tolist(), is_stable])
            alp2 -= alp2_step
            change_flag = True
        print("Left border\n")    
    
        return alpha2_area_existence, change_flag
    
    except:
        print(f'\nError\neps2={eps2}\talpha2={alp2}\tinit_vec={init_vec}')
        return alpha2_area_existence, change_flag


def rb_stretching_by_epsilon2_alpha2(params, init_eps2, alp2, vec_in_init_eps2, eps2_step, alp2_step, period_len, eps2_bord=(0., 0.2)):
    # Parameters
    N, mu, eps1, alp1 = params
    
    
    # Streching
    area_existence = []
    try:
        # Up streching by epsilon2
        eps2 = init_eps2
        init_vec = vec_in_init_eps2
        while True:
            if eps2 > eps2_bord[1]:
                break
            
            alpha2_area_existence, change_flag = rb_stretching_by_alpha2(params, eps2, alp2, init_vec, alp2_step, period_len)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[0][2]
            area_existence.append(alpha2_area_existence)
            eps2 += eps2_step
        print("Up border\n")
        
        
        # Down streching by epsilon2
        eps2 = init_eps2 - eps2_step
        init_vec = vec_in_init_eps2
        while True:
            if eps2 < eps2_bord[0]:
                break
            
            alpha2_area_existence, change_flag = rb_stretching_by_alpha2(params, eps2, alp2, init_vec, alp2_step, period_len)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[0][2]
            area_existence.append(alpha2_area_existence)
            eps2 -= eps2_step
        print("Down border\n")
        
        return area_existence
            
    except:
        print(f'\nError\nepsilon2={eps2}\talpha2={alp2}\tinit_vec={init_vec}')
        return area_existence


def write_to_file(file_name, dir_name, params, area_existence):
    
    def sorted_area_arr(arr):
        def format_item(item):
            return [
                round(item[0], 6), 
                round(item[1], 6),
                [round(x, 6) for x in item[2]], 
                item[3]
                ]
        
        # Sort arr
        arr.sort(key=lambda subarr: subarr[0][1] if subarr else 0)
        
        # Sort subarr
        for subarr in arr:
            subarr.sort(key=lambda element: element[0])
            for i in range(len(subarr)):
                subarr[i] = format_item(subarr[i])
        
        return arr
    
    with open(dir_name + '/' + file_name, 'w') as fw:
        json.dump([params, sorted_area_arr(area_existence)], fw)


if __name__ == "__main__":    
    # Const parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    const_pars = [N, mu, epsilon1, alpha1]
    
    # RB parameters
    period_len = 2*np.pi
    
    
    # File path
    dir_name = 'RotobreatherResults'
    
    # file_name = (f'Results/Reduced_area_exist_N={N}_mu={mu:.2f}_'\
    #              f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_stepN={area_step_n}.txt')
    # file_name = 'test.txt'
    file_name = 'test2.txt'
        
    
    # Area settings
    area_step_n = 25
    alpha2_step = 2*np.pi / area_step_n
    epsilon2_step = 0.2 / area_step_n
    area_existence = []
    
    
    # Streching
    try:
        # Initial conditions
        init_epsilon2 = 0.08
        init_alpha2 = -2.0
        initial_vec = np.array([-0.05921, 2.646768, 0.146785, 41.029692])
        eps2_border = (0.075, 0.085)
        
        
        # Calc type
        # alpha2_area_existence, flag = rb_stretching_by_alpha2(const_pars, init_epsilon2, init_alpha2, initial_vec, alpha2_step, period_len)
        # area_existence = [alpha2_area_existence]
        
        area_existence = rb_stretching_by_epsilon2_alpha2(const_pars, init_epsilon2, init_alpha2, initial_vec, 
                                                          epsilon2_step, alpha2_step, period_len, eps2_border)
        
        
        # Writing to file        
        write_to_file(file_name, dir_name, const_pars, area_existence)
    
    except:
        # Writing to file
        write_to_file(file_name, dir_name, const_pars, area_existence)
