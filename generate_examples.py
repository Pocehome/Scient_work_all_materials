from syst_without_reduc import np, full_syst_stability_determination, draw_graph, create_full_syst_func, num_integration
import json


def draw_start_end(arr_sol, arr_t, _y_name, T, ex_legend=[]):
    # draw_graph([arr_t], arr_sol,
    #            [(0, 100), (-np.pi-0.5, np.pi+0.5)],
    #            x_name='t', y_name=_y_name,
    #            colors=['black'], legend=ex_legend)

    draw_graph([arr_t], arr_sol,
               [(T-150, T), (-np.pi-0.5, np.pi+0.5)],
               x_name='t', y_name=_y_name,
               colors=['black'], legend=ex_legend)


def draw_examples_with_theta1(arr_sol, arr_t, N, T):
    arr_theta_1 = [np.mod(arr_sol[0], 2*np.pi) - np.pi]
    arr_d_theta_1 = [arr_sol[1]]
    arr_theta_x = [np.mod(arr_sol[i], 2*np.pi) - np.pi for i in range(2, N+1, 2)]
    arr_d_theta_x = [arr_sol[i] for i in range(3, N+2, 2)]
    arr_theta_y = [np.mod(arr_sol[i], 2*np.pi) - np.pi for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    # draw graph for theta_x by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_x, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_x$', T, ex_legend=[r'$\theta_1$'])

    # draw graph for theta_y by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_y, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_y$', T, ex_legend=[r'$\theta_1$'])
    
    # draw graph for d_theta_x by T
    draw_start_end(arr_d_theta_1+arr_d_theta_x, arr_t, r'$\dot{\theta}_x$', 
                  T, ex_legend=[r'$\dot{\theta}_1$'])
    
    # draw graph for d_theta_y by T
    draw_start_end(arr_d_theta_1+arr_d_theta_y, arr_t, r'$\dot{\theta}_y$', 
                  T, ex_legend=[r'$\dot{\theta}_1$'])
    

def draw_examples_relative_theta1(arr_sol, arr_t, N, T, n_ciclop):
    k = int((N-1)/2)
    
    # arr_theta_1 = [arr_sol[0]]
    # arr_d_theta_1 = [arr_sol[1]]
    
    arr_theta_x  = [arr_sol[0]] * k
    for i in range(k):
        if i == n_ciclop-1: continue
        arr_theta_x[i] = arr_sol[2+2*i]
    
    arr_d_theta_x  = [arr_sol[1]] * k
    for i in range(k):
        if i == n_ciclop-1: continue
        arr_d_theta_x[i] = arr_sol[3+2*i]
    
    arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    arr_ciclop = np.array(arr_sol[2*n_ciclop])
    arr_d_ciclop = np.array(arr_sol[2*n_ciclop+1])
    
    arr_claster_1 = np.mod(np.array(arr_theta_x) - arr_ciclop, 2*np.pi) - np.pi
    arr_d_claster_1 = np.array(arr_d_theta_x) - arr_d_ciclop
    
    arr_claster_2 = np.mod(np.array(arr_theta_y) - arr_ciclop, 2*np.pi) - np.pi
    arr_d_claster_2 = np.array(arr_d_theta_y) - arr_d_ciclop
    
    # draw graph for theta_x by T
    draw_start_end(arr_claster_1, arr_t, r'$\delta_x$', T)
    
    # draw graph for theta_y by T
    draw_start_end(arr_claster_2, arr_t, r'$\delta_y$', T)

    # draw graph for d_theta_x by T
    draw_start_end(arr_d_claster_1, arr_t, r'$\dot{\delta}_x$', T)
    
    # draw graph for d_theta_y by T
    draw_start_end(arr_d_claster_2, arr_t, r'$\dot{\delta}_y$', T)


def save_integrate_results(arr_sol, arr_t, time, eigv, file_name):
    eigv_write = [str(num) for num in eigv]
    with open('Examples/'+file_name, 'w') as fw:
        json.dump([arr_sol, arr_t, time, eigv_write], fw)


def create_example(area_el, file_name, n_ciclop=False):
    # common variables
    N = 11
    mu = 1.0
    eps1 = 1.0
    alp1 = 1.7
    k = (N-1) // 2
    
    # variables
    alp2 = area_el[0]
    eps2 = area_el[1]
    init_vec = np.array(area_el[2])
    
    try:
        with open('Examples/'+file_name, 'r') as fr:
            arr_sol, arr_t, time, eigv = json.load(fr)
            eigv = np.array([complex(num) for num in eigv])
    except:
        # stable determination
        f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
        isStable, eigv = f_stab_det(init_vec)
        print(isStable)
        
        # numerical integration
        time = 1000
        init_xy_vec = np.array([0., init_vec[0], init_vec[1], init_vec[2]])
        init_full_syst_vec = np.array([0., 0.] + 
                               [init_xy_vec[0], init_xy_vec[1]]*k + 
                               [init_xy_vec[2], init_xy_vec[3]]*k)
        init_full_syst_vec += np.random.uniform(0., 1., 2*N)
        rhs = create_full_syst_func(N, mu, eps1, alp1, eps2, alp2)
        arr_sol, arr_t = num_integration(rhs, init_full_syst_vec, time)
        
        save_integrate_results(arr_sol.tolist(), arr_t.tolist(), time, eigv.tolist(), file_name)        
    
    # Drawing
    if n_ciclop:
        draw_examples_relative_theta1(arr_sol, arr_t, N, time, n_ciclop)
    else:
        draw_examples_with_theta1(arr_sol, arr_t, N, time)
    
    return eigv
    

if __name__ == '__main__':
    #  Stable example
    stable_ex = [2.272566, 0.08, [-0.044528, 2.541541, 0.154365, 44.659187], True]
    n_ciclop = 3
    # stable_ex = [2.398229, 0.08, [-0.048831, 2.496219, 0.144098, 43.932610], True]
    # n_ciclop = 3
    stable_eigv = create_example(stable_ex, f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_ciclop)
    
    # Unstable example
    # unstable_ex = [2.523893, 0.08, [-0.052839, 2.459533, 0.134690, 43.361433], False]
    # n_ciclop = 4
    unstable_ex = [2.649557, 0.08, [-0.056621, 2.430033, 0.126243, 42.912005], False]
    n_ciclop = 1
    unstable_eigv = create_example(unstable_ex, f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_ciclop)
    