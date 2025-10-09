import numpy as np
from system_dynamic import xy_dyn, num_integration, draw_graph
# from monodromia_matrix import stability_determination
from syst_without_reduc import full_syst_stability_determination


def get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2, init_x=0.):
    
    def calc_xy(vec_i):
        res = np.array([0.] * 4)
    
        x = init_x
        y = vec_i[1]
    
        for_x_q1 = eps1 * (np.sin(alp1) - np.sin(x + alp1) -
                               (N - 1)/2 * (np.sin(x - alp1) + np.sin(y - alp1) +
                                            np.sin(alp1) + np.sin(x - y + alp1)))
        for_x_q2 = eps2 * (np.sin(alp2) - np.sin(2*x + alp2) -
                               (N - 1)/2 * (np.sin(2*x - alp2) + np.sin(2*y - alp2) +
                                            np.sin(alp2) + np.sin(2*(x - y) + alp2)))
    
        for_y_q1 = eps1 * (np.sin(alp1) - np.sin(y + alp1) -
                               (N - 1)/2 * (np.sin(x - alp1) + np.sin(y - alp1) +
                                            np.sin(alp1) + np.sin(y - x + alp1)))
        for_y_q2 = eps2 * (np.sin(alp2) - np.sin(2*y + alp2) -
                               (N - 1)/2 * (np.sin(2*x - alp2) + np.sin(2*y - alp2) +
                                            np.sin(alp2) + np.sin(2*(y - x) + alp2)))
    
        res[0] = vec_i[0]
        res[1] = ((for_x_q1 + for_x_q2) / N - vec_i[0]) / mu
        res[2] = vec_i[2]
        res[3] = ((for_y_q1 + for_y_q2) / N - vec_i[2]) / mu
    
        return res
    
    return calc_xy


def br_FG(vec_i, rhs, init_x):
    vec_for_int = np.array([init_x, vec_i[0], vec_i[1], vec_i[2]])
    T = vec_i[3]
    res = np.array([0., 0., 0., 0.])
    
    vT = num_integration(rhs, vec_for_int, T)[0][:, -1]
    
    res[0] = vT[0] - init_x
    res[1] = vT[1] - vec_i[0]
    res[2] = vT[2] - vec_i[1]
    res[3] = vT[3] - vec_i[2]
    
    return res


def br_FG_dxp(vec_i, fg, rhs, init_x):
    delta = 1e-6
    delta_vec = np.array([delta, 0., 0., 0.])
    # res = np.array([0., 0., 0., 0.])
    
    res = (br_FG(vec_i + delta_vec, rhs, init_x) - fg) / delta
    return res


def br_FG_dy(vec_i, fg, rhs, init_x):
    delta = 1e-6
    delta_vec = np.array([0, delta, 0, 0])
    
    res = (br_FG(vec_i + delta_vec, rhs, init_x) - fg) / delta
    return res


def br_FG_dyp(vec_i, fg, rhs, init_x):
    delta = 1e-6
    delta_vec = np.array([0, 0, delta, 0])
    
    res = (br_FG(vec_i + delta_vec, rhs, init_x) - fg) / delta
    return res


# finding the next vector using Newton's method
def find_br_next_vec(vec_i, fg, rhs, calc_br_xy, init_x):
    xy = calc_br_xy(vec_i)
    matrix = np.column_stack((br_FG_dxp(vec_i, fg, rhs, init_x),
                              br_FG_dy(vec_i, fg, rhs, init_x),
                              br_FG_dyp(vec_i, fg, rhs, init_x),
                              xy))
    inv_matrix = np.linalg.inv(matrix)
    
    return vec_i - np.dot(inv_matrix, fg)


# def make_func_find_initial_vec(N, mu, eps1, alp1, eps2, alp2):
#     rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
#     calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2)
#     f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)


def find_br_init_vec(vec_0, rhs, calc_xy, f_stab_det, init_x, do_stab_det=False):
    vec_i = vec_0
    find_flag = False
    is_stable = None
    eigv = None
    
    for _ in range(20):
        # print('\n', vec_i)
        if vec_i[3] < 0:
            break
        
        # Newton's method
        fg = br_FG(vec_i, rhs, init_x)
        
        if not np.all(np.abs(fg) < 20):
            # print("Too high FG values\n")
            break
        # print(f'F1={fg[0]:.9f}\tG1={fg[1]:.9f}\tF2={fg[2]:.9f}\tG2={fg[3]:.9f}\n')
        
        err = 1e-6
        if np.all(np.abs(fg) < err):
            find_flag = True
            break
        
        vec_i = find_br_next_vec(vec_i, fg, rhs, calc_xy, init_x)
    
    if find_flag and do_stab_det:
        is_stable, eigv = f_stab_det(vec_i)
    
    return vec_i, find_flag, is_stable, eigv


def turnover_made(t, y):
    return abs(y[0])
turnover_made.terminal = True


def full_br_finding_func(params, vec_0, init_x, do_stable_det=False, printing=False):
    N, mu, eps1, alp1, eps2, alp2 = params
    # Calc functions
    rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
    calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2, init_x)
    f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2, init_x)
        
    # Newton's method
    initial_vec, find_flag, isStable, eigv = find_br_init_vec(vec_0, rhs, calc_xy, f_stab_det, 
                                                              init_x, do_stable_det)
    if printing:
        if do_stable_det:
            print([alp2, eps2, initial_vec.tolist(), isStable],
                  '\nFind flag:', find_flag, '\nIs stable:', isStable, '\n')
        else:
            print([alp2, eps2, initial_vec.tolist(), isStable], '\n')
        
    return initial_vec, find_flag, isStable, eigv


if __name__ == "__main__":
    # Const parameters
    mu = 1.0
    epsilon1 = 1.0
    
    
    # Initial conditions    
    # N = 11
    # alpha1 = 1.7
    # epsilon2 = 0.075
    # alpha2 = 0.
    # vec0 = np.array([0.164135, -1.199126, 0.093172, 28.888474])
    # x0 = 2.
    
    N = 11
    alpha1 = 1.7
    epsilon2 = 0.101 
    alpha2 = -0.942478
    vec0 = np.array([0.329746, -1.048186, 0.240004, 25.621749])
    x0 = 2.
    
    
    # Finding
    params = [N, mu, epsilon1, alpha1, epsilon2, alpha2]
    initial_vec, find_flag, isStable, eigv = full_br_finding_func(params, vec0, x0, True)
    print([round(x, 6) for x in initial_vec.tolist()], find_flag, isStable)#, '\n', eigv)