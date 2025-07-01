from syst_without_reduc import np, full_syst_stability_determination, draw_graph, create_full_syst_func, num_integration
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def draw_start_end(arr_sol, arr_t, _y_name, T, ex_legend=[]):
    # draw_graph([arr_t], arr_sol,
    #             [(0, 200), (-np.pi-0.5, np.pi+0.5)],
    #             x_name='t', y_name=_y_name,
    #             colors=['blacK'], legend=ex_legend)

    draw_graph([arr_t], arr_sol,
               [(T-200, T), (-np.pi-0.5, np.pi+0.5)],
               x_name='t', y_name=_y_name,
               colors=['blacK'], legend=ex_legend)


def draw_examples_with_theta1(arr_sol, arr_t, N, T):
    arr_theta_1 = [arr_sol[0]]
    arr_d_theta_1 = [arr_sol[1]]
    arr_theta_x = [arr_sol[i] for i in range(2, N+1, 2)]
    arr_d_theta_x = [arr_sol[i] for i in range(3, N+2, 2)]
    arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    # draw graph for theta_x by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_x, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_x$', T, ex_legend=[r'$\theta_1$'])

    # draw graph for theta_y by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_y, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_y$', T, ex_legend=[r'$\theta_1$'])
    
    # draw graph for d_theta_x by T
    # draw_start_end(arr_d_theta_1+arr_d_theta_x, arr_t, r'$\dot{\theta}_x$', 
    #               T, ex_legend=[r'$\dot{\theta}_1$'])
    
    # draw graph for d_theta_y by T
    # draw_start_end(arr_d_theta_1+arr_d_theta_y, arr_t, r'$\dot{\theta}_y$', 
    #               T, ex_legend=[r'$\dot{\theta}_1$'])
    

def draw_examples_relative_theta1(arr_sol, arr_t, N, T, n_ciclop):
    # arr_theta_1 = [arr_sol[0]]
    # arr_d_theta_1 = [arr_sol[1]]
    
    arr_theta_x  = [arr_sol[0]] * K
    for i in range(K):
        if i == n_ciclop-1: continue
        arr_theta_x[i] = arr_sol[2+2*i]
    
    arr_d_theta_x  = [arr_sol[1]] * K
    for i in range(K):
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
    draw_start_end(arr_claster_1, arr_t, r'$\theta_x - \theta_1$', T)
    
    # draw graph for theta_y by T
    draw_start_end(arr_claster_2, arr_t, r'$\theta_y - \theta_1$', T)

    # draw graph for d_theta_x by T
    # draw_start_end(arr_d_claster_1, arr_t, r'$\dot{\theta}_x$', T)
    
    # draw graph for d_theta_y by T
    # draw_start_end(arr_d_claster_2, arr_t, r'$\dot{\theta}_y$', T)


def save_integrate_results(arr_sol, arr_t, time, eigv, file_name):
    eigv_write = [str(num) for num in eigv]
    with open('Examples/'+file_name, 'w') as fw:
        json.dump([arr_sol, arr_t, time, eigv_write], fw)


def create_example(params, area_el, file_name, n_ciclop=False):
    # print(file_name)
    # Variables
    N, mu, eps1, alp1 = params
    alp2 = area_el[0]
    eps2 = area_el[1]
    init_vec = np.array(area_el[2])
    
    try:
        # a = 1/0
        with open('Examples/'+file_name, 'r') as fr:
            arr_sol, arr_t, time, eigv = json.load(fr)
            eigv = np.array([complex(num) for num in eigv])
        
    except:
        a = 1/0
        # stable determination
        f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
        isStable, eigv = f_stab_det(init_vec)
        print(isStable)
        
        # numerical integration
        time = 200
        init_xy_vec = np.array([0., init_vec[0], init_vec[1], init_vec[2]])
        init_full_syst_vec = np.array([0., 0.] + 
                               [init_xy_vec[0], init_xy_vec[1]]*K + 
                               [init_xy_vec[2], init_xy_vec[3]]*K)
        init_full_syst_vec += np.random.uniform(0., 0.5, 2*N)
        rhs = create_full_syst_func(N, mu, eps1, alp1, eps2, alp2)
        arr_sol, arr_t = num_integration(rhs, init_full_syst_vec, time)
        
        save_integrate_results(arr_sol.tolist(), arr_t.tolist(), time, eigv.tolist(), file_name)        
    
    # Drawing
    if n_ciclop:
        draw_examples_relative_theta1(arr_sol, arr_t, N, time, n_ciclop)
    else:
        draw_examples_with_theta1(arr_sol, arr_t, N, time)
    
    return eigv


def shifted_hsv_cmap(shift=0.5):
    # Стандартная hsv палитра
    hsv = plt.get_cmap('hsv')
    
    # Создаем новые цвета со сдвигом
    x = np.linspace(0, 1, 256)
    new_colors = hsv((x + shift) % 1)
    
    return mcolors.ListedColormap(new_colors)


def draw_phase_diff(params, area_el, file_name, n_cyclop=0):
    # N, mu, eps1, alp1 = params
    # alp2 = area_el[0]
    # eps2 = area_el[1]
    # init_vec = np.array(area_el[2])
    
    with open('Examples/'+file_name, 'r') as fr:
        arr_sol, arr_t, T, eigv = json.load(fr)
        eigv = np.array([complex(num) for num in eigv])
    
    # Data arrs
    if n_cyclop:
        print(n_ciclop)
        arr_ciclop = [arr_sol[2*n_ciclop]]
        
        arr_theta_x  = [arr_sol[0]] * K
        for i in range(K):
            if i == n_ciclop-1: continue
            arr_theta_x[i] = arr_sol[2+2*i]
        
        arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
        
        arr_all = np.array(arr_theta_y + arr_ciclop + arr_theta_x)
        
    else:
        arr_ciclop = [arr_sol[0]]
        arr_theta_x = [arr_sol[i] for i in range(2, N+1, 2)]
        arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
        arr_all = np.array(arr_theta_y + arr_ciclop + arr_theta_x)
    
    # Diffs with Theta1
    diff = np.mod(arr_all - arr_all[K] + np.pi, 2*np.pi) - np.pi
    start_idx = (T-200) * 100
    
    # Create diagramm
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # View data
    shifted_hsv = shifted_hsv_cmap(0.685)
    cax = ax.imshow(diff[:, start_idx:], aspect='auto', cmap=shifted_hsv,
                    extent=[arr_t[start_idx], T, 0.5, N+0.5], origin='upper',
                    interpolation='none', vmin=-np.pi, vmax=np.pi)
    
    # Настраиваем оси
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('k', fontsize=12)
    ax.set_yticks(np.arange(1, 12))
    ax.set_yticklabels(np.arange(1, 12))

    # Color bar
    cbar = fig.colorbar(cax)
    cbar.set_label('Разность фаз с уединённым элементом', rotation=270, labelpad=15)

    plt.title('Диаграмма разностей фаз', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Common variables
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.6
    K = (N-1) // 2
    
    #  Stable examples
    # stable_ex = [2.272566, 0.08, [-0.044528, 2.541541, 0.154365, 44.659187], True]
    # n_ciclop = 3
    # stable_ex = [2.398229, 0.08, [-0.048831, 2.496219, 0.144098, 43.932610], True]
    # n_ciclop = 3
    
    # stable_ex = [1.9961058553661977, 0.08, [-0.03359744015159435, 2.6830724626023508, 0.17867310245657522, 47.05472020125708], True]
    # n_ciclop = 0
    
    # stable_ex = [2.5, 0.08, [-0.052096650048026875, 2.465919663536965, 0.13640774969849193, 43.45974058398454], True]
    # n_ciclop = 0
    
    stable_ex = [-2.7037167544041196, 0.08, [-0.07604374953866636, 2.3880081439424705, 0.10081147902837394, 41.215762422260774], True] 
    n_ciclop = 2
    
    # stable_eigv = create_example([N, mu, epsilon1, alpha1], stable_ex,
    #                               f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_ciclop)
    draw_phase_diff([N, mu, epsilon1, alpha1], stable_ex, 
                    f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_ciclop)
    
    
    # Unstable examples
    # unstable_ex = [2.523893, 0.08, [-0.052839, 2.459533, 0.134690, 43.361433], False]
    # n_ciclop = 4
    # unstable_ex = [2.649557, 0.08, [-0.056621, 2.430033, 0.126243, 42.912005], False]
    # n_ciclop = 1
    
    # unstable_ex = [1.6945129606215776, 0.08, [-0.0186408567985006, 2.9468677840319804, 0.1982185503528971, 52.112803327402816], False]
    # n_ciclop = 0
    
    unstable_ex = [-2.854513201776431, 0.08, [-0.074726393131724, 2.372992393366901, 0.09952988138941797, 41.430975121830905], False]
    n_ciclop = 2
    
    # unstable_eigv = create_example([N, mu, epsilon1, alpha1], unstable_ex,
    #                                 f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_ciclop)
    draw_phase_diff([N, mu, epsilon1, alpha1], unstable_ex, 
                    f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_ciclop)
    