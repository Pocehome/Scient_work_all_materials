import numpy as np
import json

from system_dynamic import num_integration, draw_graph
from find_reduc_rot import full_finding_func
from draw_area_of_exiisting import read_file


def integration_drawing(rhs, init_vec_opt, T):
    # Numerical integration and drawing
    arr_sol, arr_t = num_integration(rhs, init_vec_opt, T)
    
    draw_graph([arr_t], [np.mod(arr_sol[0], 2*np.pi) - np.pi],
                [(0, T), (-np.pi-0.5, np.pi+0.5)], x_name='t', y_name='x',
                ys=[(np.mod(arr_sol[0], 2*np.pi) - np.pi)[0]])
    draw_graph([arr_t], [np.mod(arr_sol[2], 2*np.pi) - np.pi],
                [(0, T), (-np.pi-0.5, np.pi+0.5)], x_name='t', y_name='y',
                ys=[(np.mod(arr_sol[2], 2*np.pi) - np.pi)[0]])
    
    return arr_sol, arr_t
    
    
def step_by_step_finding(targ_alp1, pars, init_vec, step):
    prev_pars = pars.copy()
    prev_init_vec = init_vec
    prev_find_flag = True
    prev_isStab = 0
    prev_eigv = 0
    
    if pars[3] > targ_alp1:
        while pars[3] > targ_alp1:      
            if pars[3] - step < targ_alp1:
                pars[3] = targ_alp1
            else:
                pars[3] -= step
            
            # print('alpha1 =', pars[3])
            init_vec, find_flag, isStab, eigv = full_finding_func(pars, prev_init_vec, printing=True)
            
            if find_flag:
                prev_pars = pars.copy()
                prev_init_vec = init_vec.copy()
                prev_find_flag = find_flag
                prev_isStab = isStab
                prev_eigv = eigv
            else:
                # print('alpha1 =', prev_pars[3])
                return prev_pars, prev_init_vec, prev_find_flag, prev_isStab, prev_eigv
        
    elif pars[3] < targ_alp1:
        while pars[3] < targ_alp1:      
            if pars[3] + step > targ_alp1:
                pars[3] = targ_alp1
            else:
                pars[3] += step
            
            print('alpha1 =', pars[3])
            init_vec, find_flag, isStab, eigv = full_finding_func(pars, prev_init_vec, printing=True)
            
            if find_flag:
                prev_pars = pars.copy()
                prev_init_vec = init_vec.copy()
                prev_find_flag = find_flag
                prev_isStab = isStab
                prev_eigv = eigv
            else:
                # print('alpha1 =', prev_pars[3])
                return prev_pars, prev_init_vec, prev_find_flag, prev_isStab, prev_eigv
    
    # print('alpha1 =', prev_pars[3])
    return prev_pars, prev_init_vec, prev_find_flag, prev_isStab, prev_eigv


def search_on_line(target_alp1, f_name):
    steps = [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 7.8125e-05, 3.90625e-05]
    # steps = [1.953125e-05, 9.765625e-06]
    res = []  # res struct: [[alp2, alp1, init_vec, isStab]]
    
    try:
        a = 1/0
        N, mu, eps1, eps2, target_alp1, line = read_res(f_name)
        
        for el in line:
            if el == [0.0, 0.0, [0.0, 0.0, 0.0, 0.0], True]: continue
        
            pars = [N, mu, eps1, el[1], eps2, el[0]]
            init_vec = np.array(el[2])
            
            # init_vec, find_flag, isStab, eigv = full_finding_func(pars, init_vec, printing=False)
            # print(el[0], el[1], find_flag)
            print(f'\nAlpha_2 = {el[0]:.8f}, \tAlpha_1 = {el[1]:.8f}\n')
            for step in steps:
                pars, init_vec, find_flag, isStab, eigv = step_by_step_finding(target_alp1, pars, init_vec, step)
                
            res.append([pars[5], pars[3], init_vec.tolist()])
    except:
        print('Calk from 0 !!!!!!!!!!!!!')
        N, mu, eps1, alp1, step_n, area_exist = read_file('one_line_test2.txt')
        line = area_exist[0]
        
        for el in line:
            if el == [0.0, 0.0, [0.0, 0.0, 0.0, 0.0], True]: continue
        
            pars = [N, mu, eps1, alp1, el[1], el[0]]
            init_vec = np.array(el[2])
            
            print(f'\nAlpha_2 = {el[0]:.8f}, \tAlpha_1 = {pars[3]:.8f}\n')
            for step in steps:
                # if int(abs(alp1 - target_alp1) / step) > 40:
                #     break
                print('Step =', step)
                pars, init_vec, find_flag, isStab, eigv = step_by_step_finding(target_alp1, pars, init_vec, step)
                
            res.append([pars[5], pars[3], init_vec.tolist()])
    
    # print(*line, sep='\n')
        
    return [pars[0], pars[1], pars[2], pars[4]], res


def search_for_one_el(target_alp1, pars, el):
    N, mu, eps1, eps2 = pars
    # steps = [0.01, 0.005, 0.0025, 0.00125]
    # steps = [0.00125, 0.000625, 0.0003125, 0.00015625]
    steps = [0.00015625, 7.8125e-05, 3.90625e-05]
    
    pars = [N, mu, eps1, el[1], eps2, el[0]]
    init_vec = np.array(el[2])
    
    print(f'\nAlpha_2 = {el[0]:.8f}, \tAlpha_1 = {el[1]:.8f}\n')
    
    # pars[3] = target_alp1
    init_vec, find_flag, isStab, eigv = full_finding_func(pars, init_vec, printing=True, do_stable_det=True)
    print(find_flag)
    return 1, 1
    
    for step in steps:
        pars, init_vec, find_flag, isStab, eigv = step_by_step_finding(target_alp1, pars, init_vec, step)
        
    return [pars[0], pars[1], pars[2], pars[4]], [pars[5], pars[3], init_vec.tolist()]

    
def read_res(file_name):
    with open(file_name, 'r') as fr:
        file_pars, tar_alp1, res = json.load(fr)
        N, mu, eps1, eps2 = file_pars
    return N, mu, eps1, eps2, tar_alp1, res


def write_results_for_line(file_pars, tar_alp1, res):
    with open(f'new_alpha1_results_alp1={tar_alp1}.txt', 'w') as fw:
        json.dump([file_pars, tar_alp1, res], fw)


if __name__ == "__main__":
    # Const params
    # N = 11
    # mu = 1.0
    # epsilon1 = 1.0
    
    
    # Initial params
    target_alpha1 = 1.8
    params, results = search_on_line(target_alpha1, f'new_alpha1_results_alp1={target_alpha1}.txt')
    write_results_for_line(params, target_alpha1, results)
    
    # params = [11, 1.0, 1.0, 0.08]
    # result_el = [2.6495571273128915, 1.6009375000000008, [-0.33733754955644696, 1.407603607372734, -0.19637161833482533, 87.02323355182159]]
    # params, result_el = search_for_one_el(target_alpha1, params, result_el)
    
    
    # Printing
    # print_alphas1(results, target_alpha1)
    