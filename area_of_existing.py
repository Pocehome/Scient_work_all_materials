from find_reduc_rot import (np, xy_dyn, get_xy_from_vec_i, find_initial_vec, stability_determination)
import json


def stretching_by_alpha2(epsilon2, init_alpha2, vec_in_init_alpha2, alpha2_step, area_step_n):
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    # area_existence = [alpha2, epsilon2, [x_der, y, y_der, T], is_stable]
    alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
    change_flag = False
    rotate_flag = False
    
    try:
        # right stretching by alpha2
        alpha2 = init_alpha2
        init_vec = vec_in_init_alpha2
        while True:
            if alpha2 > init_alpha2 and rotate_flag:
                break
            if alpha2 > np.pi:
                alpha2 -= 2*np.pi
                rotate_flag = True
            
            # calculate functions
            rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            calc_xy = get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            f_stab_det = stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            
            # Newton's method
            init_vec, find_flag, is_stable = find_initial_vec(init_vec, rhs, calc_xy, f_stab_det)
            
            if not find_flag:
                break
            
            print([alpha2, epsilon2, init_vec.tolist()], is_stable, '\n')
            
            alpha2_area_existence[int(alpha2/alpha2_step)] = [alpha2, epsilon2, init_vec.tolist(), is_stable]
            alpha2 += alpha2_step
            change_flag = True
        print("Right border\n")
        
        # left stretching by alpha2
        alpha2 = init_alpha2 - alpha2_step
        init_vec = vec_in_init_alpha2
        while True:
            if alpha2 < -np.pi:
                if rotate_flag:
                    break
                else:
                    alpha2 += 2*np.pi
                    rotate_flag = True
            
            # calculate functions
            rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            calc_xy = get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            f_stab_det = stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
            
            # Newton's method
            init_vec, find_flag, is_stable = find_initial_vec(init_vec, rhs, calc_xy, f_stab_det)
            
            if not find_flag:
                break
            
            print([alpha2, epsilon2, init_vec.tolist()], is_stable, '\n')
            
            alpha2_area_existence[int(alpha2/alpha2_step)] = [alpha2, epsilon2, init_vec.tolist(), is_stable]
            alpha2 -= alpha2_step
            change_flag = True
        print("Left border\n")    
    
        return alpha2_area_existence, change_flag
    
    except:
        print(f'\nError\neps2={epsilon2}\talpha2={alpha2}\tinit_vec={init_vec}')
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
            init_vec, find_flag, is_stable = find_initial_vec(init_vec, rhs, calc_xy)
            
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
            init_vec, find_flag, is_stable = find_initial_vec(init_vec, rhs, calc_xy)
            
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


def stretching_by_epsilon2_alpha2(init_epsilon2, alpha2, vec_in_init_epsilon2, epsilon2_step, alpha2_step, area_step_n):    
    # area_existence = [alpha2, epsilon2, [x_der, y, y_der, T], is_stable]
    alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
    area_existence = [alpha2_area_existence] * area_step_n
    
    try:
        # up streching by epsilon2
        epsilon2 = init_epsilon2
        init_vec = vec_in_init_epsilon2
        while True:
            if epsilon2 > 0.2:
                break
            
            alpha2_area_existence, change_flag = stretching_by_alpha2(epsilon2, alpha2, init_vec, alpha2_step, area_step_n)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[int(alpha2/alpha2_step)][2]
            area_existence[int(epsilon2/epsilon2_step)] = alpha2_area_existence
            epsilon2 += epsilon2_step
        print("Up border\n")
        
        # down streching by epsilon2
        epsilon2 = init_epsilon2
        init_vec = vec_in_init_epsilon2
        while True:
            if epsilon2 < 0:
                break
            
            alpha2_area_existence, change_flag = stretching_by_alpha2(epsilon2, alpha2, init_vec, alpha2_step, area_step_n)
            
            if not change_flag:
                break
            
            init_vec = alpha2_area_existence[int(alpha2/alpha2_step)][2]
            area_existence[int(epsilon2/epsilon2_step)] = alpha2_area_existence
            epsilon2 -= epsilon2_step
        print("Down border\n")
        
        return area_existence
            
    except:
        print(f'\nError\nepsilon2={epsilon2}\talpha2={alpha2}\tinit_vec={init_vec}')
        return area_existence


def write_to_file(file_name, N, mu, epsilon1, alpha1, area_existence):
    with open(file_name, 'w') as fw:
        json.dump([[N, mu, epsilon1, alpha1], area_existence], fw)


if __name__ == "__main__":
    
    # file_name = (f'Results/Reduced_area_exist_N={N}_mu={mu:.2f}_'\
    #         f'eps1={epsilon1:.5f}_alpha1={alpha1:.5f}_stepN={area_step_n}.txt')
    
    file_name = 'Full_test2.txt'
    # file_name = 'one_line_test.txt'
    # file_name = 'test.txt'
    # file_name = 'Area_exist_for_scient_work.txt'
    # file_name = 'area_stability_for_scient_work.txt'
        
    try:
        # parameters
        N = 11
        mu = 1.0
        epsilon1 = 1.0
        alpha1 = 1.7
        
        # area settings
        area_step_n = 50
        alpha2_step = 2*np.pi / area_step_n
        epsilon2_step = 0.2 / area_step_n
        alpha2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
        epsilon2_area_existence = [[0., 0., [0., 0., 0., 0.], True]] * area_step_n
        area_existence = [alpha2_area_existence] * area_step_n
        
        init_epsilon2 = 0.08
        init_alpha2 = -2.0
        initial_vec = np.array([-0.05921021, 2.64676814, 0.14678535, 41.02969191])
        
        # alpha2_area_existence, flag = stretching_by_alpha2(init_epsilon2, init_alpha2, initial_vec, alpha2_step, area_step_n)
        # area_existence[0] = alpha2_area_existence
        
        # epsilon2_area_existence, flag = stretching_by_epsilon2(init_epsilon2, init_alpha2, initial_vec, epsilon2_step, area_step_n)
        
        area_existence = stretching_by_epsilon2_alpha2(init_epsilon2, init_alpha2, initial_vec, epsilon2_step, alpha2_step, area_step_n)
        
        # writing to file        
        write_to_file(file_name, N, mu, epsilon1, alpha1, area_existence)
    
    except:
        # writing to file        
        write_to_file(file_name, N, mu, epsilon1, alpha1, area_existence)
