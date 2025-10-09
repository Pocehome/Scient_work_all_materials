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


def rb_FG(vec_i, rhs, period_len):
    vec_for_int = np.array([0., vec_i[0], vec_i[1], vec_i[2]])
    T = vec_i[3]
    res = np.array([0., 0., 0., 0.])
    
    vT = num_integration(rhs, vec_for_int, T)[0][:, -1]
    
    res[0] = vT[0] + period_len
    res[1] = vT[1] - vec_i[0]
    res[2] = vT[2] - vec_i[1] + period_len
    res[3] = vT[3] - vec_i[2]
    
    return res


def rb_FG_dxp(vec_i, fg, rhs, period_len):
    delta = 1e-6
    delta_vec = np.array([delta, 0., 0., 0.])
    # res = np.array([0., 0., 0., 0.])
    
    res = (rb_FG(vec_i + delta_vec, rhs, period_len) - fg) / delta
    return res


def rb_FG_dy(vec_i, fg, rhs, period_len):
    delta = 0.000001
    delta_vec = np.array([0, delta, 0, 0])
    
    res = (rb_FG(vec_i + delta_vec, rhs, period_len) - fg) / delta
    return res


def rb_FG_dyp(vec_i, fg, rhs, period_len):
    delta = 0.000001
    delta_vec = np.array([0, 0, delta, 0])
    
    res = (rb_FG(vec_i + delta_vec, rhs, period_len) - fg) / delta
    return res


# finding the next vector using Newton's method
def find_rb_next_vec(vec_i, fg, rhs, calc_xy, period_len):
    xy = calc_xy(vec_i)
    matrix = np.column_stack((rb_FG_dxp(vec_i, fg, rhs, period_len),
                              rb_FG_dy(vec_i, fg, rhs, period_len),
                              rb_FG_dyp(vec_i, fg, rhs, period_len),
                              xy))
    inv_matrix = np.linalg.inv(matrix)
    
    return vec_i - np.dot(inv_matrix, fg)


# def make_func_find_initial_vec(N, mu, eps1, alp1, eps2, alp2):
#     rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
#     calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2)
#     f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)


def find_rb_init_vec(vec_0, rhs, calc_xy, f_stab_det, period_len, do_stab_det=False):
    vec_i = vec_0
    find_flag = False
    is_stable = None
    eigv = None
    
    for _ in range(20):
        # print('\n', vec_i)
        if vec_i[3] < 0:
            break
        
        # Newton's method
        fg = rb_FG(vec_i, rhs, period_len)
        
        if not np.all(np.abs(fg) < 20):
            # print("Too high FG values\n")
            break
        # print(f'F1={fg[0]:.9f}\tG1={fg[1]:.9f}\tF2={fg[2]:.9f}\tG2={fg[3]:.9f}\n')
        
        err = 1e-6
        if np.all(np.abs(fg) < err):
            find_flag = True
            break
        
        vec_i = find_rb_next_vec(vec_i, fg, rhs, calc_xy, period_len)
    
    if find_flag and do_stab_det:
        is_stable, eigv = f_stab_det(vec_i)
    
    return vec_i, find_flag, is_stable, eigv


def turnover_made(t, y):
    return abs(y[0]) - 2*np.pi
turnover_made.terminal = True


def full_rb_finding_func(params, vec_0, period_len, do_stab_det=False, printing=False):
    N, mu, eps1, alp1, eps2, alp2 = params
    # Calc functions
    rhs = xy_dyn(N, mu, eps1, alp1, eps2, alp2)
    calc_xy = get_xy_from_vec_i(N, mu, eps1, alp1, eps2, alp2)
    f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
        
    # Newton's method
    initial_vec, find_flag, isStable, eigv = find_rb_init_vec(vec_0, rhs, calc_xy, f_stab_det,
                                                              period_len, do_stab_det)
    if printing:
        if do_stab_det:
            print([alp2, eps2, initial_vec.tolist(), isStable],
                  '\nFind flag:', find_flag, '\nIs stable:', isStable, '\n')
        else:
            print([alp2, eps2, initial_vec.tolist(), isStable], '\n')
        
    return initial_vec, find_flag, isStable, eigv


if __name__ == "__main__":
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    
    # Initial conditions
    # epsilon2 = 0.08
    # alpha2 = -2.1
    # vec_0 = np.array([0.014374229240869783, 1.8119520391656536, 0.1509289175514676, 69.782630])
    
    # alpha2 = -np.pi
    # epsilon2 = 0.1
    # vec_0 = np.array([-0.07515058, 2.29669348, 0.08332114, 42.04651471])
    
    epsilon2 = 0.08
    alpha2 = -2.0
    vec_0 = np.array([-0.05921021, 2.64676814, 0.14678535, 41.02969191])
    
    # epsilon2 = 0.08
    # alpha2 = -2.0
    # vec_0 = np.array([-0.23284307726045125, 1.778821369446479, -0.07476857233464937, 56.803152136107485])
    
    
    # Finding
    params = [N, mu, epsilon1, alpha1, epsilon2, alpha2]
    initial_vec, find_flag, isStable, eigv = full_rb_finding_func(params, vec_0, 2*np.pi)
    print(initial_vec, find_flag, isStable, eigv)
    
    
    # # numerical integration
    # vec_for_int = np.array([0, initial_vec[0], initial_vec[1], initial_vec[2]])
    # T = initial_vec[3]
    
    # arr_sol, arr_t = num_integration(rhs, vec_for_int, T)
    # # tr_arr_sol = np.transpose(arr_sol)
    # # print(arr_sol[0] - arr_sol[-1] - np.array([2*np.pi, 0, 2*np.pi, 0]))
    
    # # draw graph for x and y derivatives by time
    # max_xy_der = max(max(arr_sol[1]), max(arr_sol[3]))
    # min_xy_der = min(min(arr_sol[1]), min(arr_sol[3]))
    
    # draw_graph([arr_t], [arr_sol[1], arr_sol[3]], 
    #             [(0, initial_vec[3]), (min_xy_der-0.5, max_xy_der+0.5)],
    #             x_name='t', colors=['blue', 'red'], legend=['\u1E8B(t)', '\u1E8F(t)'])
    
    # # draw graph for x and y by time
    # draw_graph([arr_t], [np.mod(arr_sol[0], 2*np.pi) - np.pi,
    #                       np.mod(arr_sol[2], 2*np.pi) - np.pi], 
    #             [(0, initial_vec[3]), (-np.pi, np.pi)],
    #             x_name='t', colors=['blue', 'red'], legend=['x(t)', 'y(t)'])
