import numpy as np
from system_dynamic import xy_dyn, num_integration, draw_graph
from monodromia_matrix import stability_determination


def get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2):
    
    def calc_xy(vec_i):
        res = np.array([0.] * 4)
    
        x = 0
        y = vec_i[1]
    
        for_x_q1 = epsilon1 * (np.sin(alpha1) - np.sin(x + alpha1) -
                               (N - 1)/2 * (np.sin(x - alpha1) + np.sin(y - alpha1) +
                                            np.sin(alpha1) + np.sin(x - y + alpha1)))
        for_x_q2 = epsilon2 * (np.sin(alpha2) - np.sin(2*x + alpha2) -
                               (N - 1)/2 * (np.sin(2*x - alpha2) + np.sin(2*y - alpha2) +
                                            np.sin(alpha2) + np.sin(2*(x - y) + alpha2)))
    
        for_y_q1 = epsilon1 * (np.sin(alpha1) - np.sin(y + alpha1) -
                               (N - 1)/2 * (np.sin(x - alpha1) + np.sin(y - alpha1) +
                                            np.sin(alpha1) + np.sin(y - x + alpha1)))
        for_y_q2 = epsilon2 * (np.sin(alpha2) - np.sin(2*y + alpha2) -
                               (N - 1)/2 * (np.sin(2*x - alpha2) + np.sin(2*y - alpha2) +
                                            np.sin(alpha2) + np.sin(2*(y - x) + alpha2)))
    
        res[0] = vec_i[0]
        res[1] = ((for_x_q1 + for_x_q2) / N - vec_i[0]) / mu
        res[2] = vec_i[2]
        res[3] = ((for_y_q1 + for_y_q2) / N - vec_i[2]) / mu
    
        return res
    
    return calc_xy


def FG(vec_i, rhs):
    vec_for_int = np.array([0., vec_i[0], vec_i[1], vec_i[2]])
    T = vec_i[3]
    res = np.array([0., 0., 0., 0.])
    
    vT = num_integration(rhs, vec_for_int, T, 0.001)[0][-1]
    
    res[0] = vT[0] + 2*np.pi
    res[1] = vT[1] - vec_i[0]
    res[2] = vT[2] - vec_i[1] + 2*np.pi
    res[3] = vT[3] - vec_i[2]
    
    return res


def FG_dxp(vec_i, fg, rhs):
    delta = 1e-6
    delta_vec = np.array([delta, 0., 0., 0.])
    # res = np.array([0., 0., 0., 0.])
    
    res = (FG(vec_i + delta_vec, rhs) - fg) / delta
    return res


def FG_dy(vec_i, fg, rhs):
    delta = 0.000001
    delta_vec = np.array([0, delta, 0, 0])
    
    res = (FG(vec_i + delta_vec, rhs) - fg) / delta
    return res


def FG_dyp(vec_i, fg, rhs):
    delta = 0.000001
    delta_vec = np.array([0, 0, delta, 0])
    
    res = (FG(vec_i + delta_vec, rhs) - fg) / delta
    return res


# finding the next vector using Newton's method
def find_next_vec(vec_i, fg, rhs, calc_xy):
    xy = calc_xy(vec_i)
    matrix = np.column_stack((FG_dxp(vec_i, fg, rhs), FG_dy(vec_i, fg, rhs),
                              FG_dyp(vec_i, fg, rhs), xy))
    inv_matrix = np.linalg.inv(matrix)
    
    return vec_i - np.dot(inv_matrix, fg)


def find_initial_vec(vec_0, rhs, calc_xy, f_stab_det):
    vec_i = vec_0
    find_flag = False
    is_stable = False
    
    for _ in range(20):
        # print('\n', vec_i)
        if vec_i[3] < 0:
            break
        
        # Newton's method
        fg = FG(vec_i, rhs)
        
        if not np.all(np.abs(fg) < 20):
            # print("Too high FG values\n")
            break
        # print(f'F1={fg[0]:.9f}\tG1={fg[1]:.9f}\tF2={fg[2]:.9f}\tG2={fg[3]:.9f}\n')
        
        err = 1e-6
        if np.all(np.abs(fg) < err):
            find_flag = True
            break
        
        vec_i = find_next_vec(vec_i, fg, rhs, calc_xy)
    
    if find_flag:
        is_stable = f_stab_det(vec_i)
    
    return vec_i, find_flag, is_stable


def turnover_made(t, y):
    return abs(y[0]) - 2*np.pi
turnover_made.terminal = True


if __name__ == "__main__":
    
    # parameters
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    
    # alpha2 = -np.pi
    # epsilon2 = 0.1
    # vec_0 = np.array([-0.07515058, 2.29669348, 0.08332114, 42.04651471])
    
    epsilon2 = 0.08
    alpha2 = -2.0
    vec_0 = np.array([-0.05921021, 2.64676814, 0.14678535, 41.02969191])
    
    # calculate functions
    rhs = xy_dyn(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    calc_xy = get_xy_from_vec_i(N, mu, epsilon1, alpha1, epsilon2, alpha2)
    f_stab_det = stability_determination(N, mu, epsilon1, alpha1, epsilon2, alpha2)
        
    # Newton's method
    initial_vec, find_flag, is_stable = find_initial_vec(vec_0, rhs, calc_xy, f_stab_det)
    print(initial_vec.tolist(), '\nFind flag:', find_flag, '\nIs stable:', is_stable)
    
    # numerical integration
    step_size = 0.01
    vec_for_int = np.array([0, initial_vec[0], initial_vec[1], initial_vec[2]])
    T = initial_vec[3]
    
    arr_sol, arr_t = num_integration(rhs, vec_for_int, T, step_size)
    tr_arr_sol = np.transpose(arr_sol)
    # print(arr_sol[0] - arr_sol[-1] - np.array([2*np.pi, 0, 2*np.pi, 0]))
    
    # darw graph for x and y derivatives by time
    max_xy_der = max(max(tr_arr_sol[1]), max(tr_arr_sol[3]))
    min_xy_der = min(min(tr_arr_sol[1]), min(tr_arr_sol[3]))
    
    draw_graph([arr_t], [tr_arr_sol[1], tr_arr_sol[3]], 
               [(0, initial_vec[3]), (min_xy_der-0.5, max_xy_der+0.5)],
               x_name='t', colors=['blue', 'red'], legend=['\u1E8B(t)', '\u1E8F(t)'])
    
    # draw graph for x and y by time
    draw_graph([arr_t], [np.mod(tr_arr_sol[0], 2*np.pi) - np.pi,
                         np.mod(tr_arr_sol[2], 2*np.pi) - np.pi], 
               [(0, initial_vec[3]), (-np.pi, np.pi)],
               x_name='t', colors=['blue', 'red'], legend=['x(t)', 'y(t)'])
