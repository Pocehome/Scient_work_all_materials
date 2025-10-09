import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from syst_without_reduc import (np, full_syst_stability_determination, 
                                create_full_syst_func, num_integration,
                                draw_start_end)


def draw_examples_with_theta1(arr_sol, arr_t, N, T, n_cyc, d_lims):
    # Claster arrays
    arr_theta_1 = [arr_sol[2*n_cyc]]
    arr_d_theta_1 = [arr_sol[2*n_cyc+1]]
    
    arr_theta_x  = [arr_sol[0]] * K
    for i in range(K):
        if i == n_cyc-1: continue
        arr_theta_x[i] = arr_sol[2+2*i]
    
    arr_d_theta_x  = [arr_sol[1]] * K
    for i in range(K):
        if i == n_cyc-1: continue
        arr_d_theta_x[i] = arr_sol[3+2*i]
    
    arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    
    # Drawing
    # draw graph for theta_x by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_x, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_x, \theta_1$', T, ex_legend=[r'$\theta_1$'], draw_start=True)

    # draw graph for theta_y by T
    draw_start_end(np.mod(arr_theta_1+arr_theta_y, 2*np.pi) - np.pi,
                   arr_t, r'$\theta_y, \theta_1$', T, ex_legend=[r'$\theta_1$'], draw_start=True)
    
    # draw graph for d_theta_x by T
    draw_start_end(arr_d_theta_1+arr_d_theta_x, arr_t, r'$\dot{\theta}_x$', 
                  T, ylims=d_lims, ex_legend=[r'$\dot{\theta}_1$'], draw_start=False)
    
    # draw graph for d_theta_y by T
    draw_start_end(arr_d_theta_1+arr_d_theta_y, arr_t, r'$\dot{\theta}_y$', 
                  T, ylims=d_lims, ex_legend=[r'$\dot{\theta}_1$'], draw_start=False)
    

def draw_examples_relative_theta1(arr_sol, arr_t, N, T, n_cyc, d_lims):
    # arr_theta_1 = [arr_sol[0]]
    # arr_d_theta_1 = [arr_sol[1]]
    
    arr_theta_x  = [arr_sol[0]] * K
    for i in range(K):
        if i == n_cyc-1: continue
        arr_theta_x[i] = arr_sol[2+2*i]
    
    arr_d_theta_x  = [arr_sol[1]] * K
    for i in range(K):
        if i == n_cyc-1: continue
        arr_d_theta_x[i] = arr_sol[3+2*i]
    
    arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
    arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
    
    arr_cyclop = np.array(arr_sol[2*n_cyc])
    arr_d_cyclop = np.array(arr_sol[2*n_cyc+1])
    
    arr_claster_1 = np.mod(np.array(arr_theta_x) - arr_cyclop, 2*np.pi) - np.pi
    arr_d_claster_1 = np.array(arr_d_theta_x) - arr_d_cyclop
    
    arr_claster_2 = np.mod(np.array(arr_theta_y) - arr_cyclop, 2*np.pi) - np.pi
    arr_d_claster_2 = np.array(arr_d_theta_y) - arr_d_cyclop
    
    # draw graph for theta_x by T
    draw_start_end(arr_claster_1, arr_t, r'$\theta_x - \theta_1$', T, draw_start=True)
    
    # draw graph for theta_y by T
    draw_start_end(arr_claster_2, arr_t, r'$\theta_y - \theta_1$', T, draw_start=True)
    
    # draw graph for d_theta_x by T
    draw_start_end(arr_d_claster_1, arr_t, r'$\dot{\theta}_x - \dot{\theta}_1$', T,
                   ylims=d_lims, draw_start=False)
    
    # draw graph for d_theta_y by T
    draw_start_end(arr_d_claster_2, arr_t, r'$\dot{\theta}_y - \dot{\theta}_1$', T,
                   ylims=d_lims, draw_start=False)


def save_example(arr_sol, arr_t, time, eigv, file_name, dir_name='Examples/'):
    eigv_write = [str(num) for num in eigv]
    with open(dir_name+file_name, 'w') as fw:
        json.dump([arr_sol, arr_t, time, eigv_write], fw)


def create_example(params, area_el, file_name, n_cyclop=0, 
                   d_limits=(-1, 1), relative_theta1=True, new_calc=False):
    # print(file_name)
    # Variables
    N, mu, eps1, alp1 = params
    alp2 = area_el[0]
    eps2 = area_el[1]
    init_vec = np.array(area_el[2])
    
    try:
        if new_calc:
            1/0
        with open('Examples/'+file_name, 'r') as fr:
            arr_sol, arr_t, time, eigv = json.load(fr)
            eigv = np.array([complex(num) for num in eigv])
        
    except:
        print('New calculating')
        # a = 1/0
        
        # stable determination
        f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2)
        isStable, eigv = f_stab_det(init_vec)
        print(isStable)
        
        # numerical integration
        time = 1000
        init_xy_vec = np.array([0., init_vec[0], init_vec[1], init_vec[2]])
        init_full_syst_vec = np.array([0., 0.] + 
                               [init_xy_vec[0], init_xy_vec[1]]*K + 
                               [init_xy_vec[2], init_xy_vec[3]]*K)
        init_full_syst_vec += np.random.uniform(0., 1, 2*N)
        rhs = create_full_syst_func(N, mu, eps1, alp1, eps2, alp2)
        arr_sol, arr_t = num_integration(rhs, init_full_syst_vec, time)
        
        save_example(arr_sol.tolist(), arr_t.tolist(), time, eigv.tolist(), file_name)        
    
    # Drawing
    if relative_theta1:
        draw_examples_relative_theta1(arr_sol, arr_t, N, time, n_cyclop, d_limits)
    else:
        draw_examples_with_theta1(arr_sol, arr_t, N, time, n_cyclop, d_limits)
    
    return eigv


def shifted_hsv_cmap(shift=0.675):
    # Стандартная hsv палитра
    hsv = plt.get_cmap('hsv')
    
    # Создаем новые цвета со сдвигом
    x = np.linspace(0, 1, 256)
    new_colors = hsv((x + shift) % 1)
    
    return mcolors.ListedColormap(new_colors)


def draw_phase_diff(params, area_el, file_name, n_cyc=0):
    # N, mu, eps1, alp1 = params
    # alp2 = area_el[0]
    # eps2 = area_el[1]
    # init_vec = np.array(area_el[2])
    
    with open('Examples/'+file_name, 'r') as fr:
        arr_sol, arr_t, T, eigv = json.load(fr)
        eigv = np.array([complex(num) for num in eigv])
    
    # Data arrs
    if n_cyc:
        print(n_cyc)
        arr_cyclop = [arr_sol[2*n_cyc]]
        
        arr_theta_x  = [arr_sol[0]] * K
        for i in range(K):
            if i == n_cyc-1: continue
            arr_theta_x[i] = arr_sol[2+2*i]
        
        arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
        
        arr_all = np.array(arr_theta_y + arr_theta_x + arr_cyclop)
        
    else:
        arr_cyclop = [arr_sol[0]]
        arr_theta_x = [arr_sol[i] for i in range(2, N+1, 2)]
        arr_theta_y = [arr_sol[i] for i in range(N+1, 2*N, 2)]
        arr_all = np.array(arr_theta_x + arr_theta_y + arr_cyclop)
    
    # Diffs with Theta1
    diff = np.mod(arr_all - arr_all[-1] + np.pi, 2*np.pi) - np.pi
    start_idx = (T-150) * 100
    
    # Create diagramm
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # View data
    shifted_hsv = shifted_hsv_cmap()
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
    cbar.set_label(r'$\theta_k - \theta_1$', rotation=270, labelpad=15, fontsize=16)

    # plt.title('Диаграмма разностей фаз', fontsize=14)
    plt.tight_layout()
    plt.show()


def draw_speed_diff(params, area_el, file_name, n_cyc=0, lim=1):
    # N, mu, eps1, alp1 = params
    # alp2 = area_el[0]
    # eps2 = area_el[1]
    # init_vec = np.array(area_el[2])
    
    with open('Examples/'+file_name, 'r') as fr:
        arr_sol, arr_t, T, eigv = json.load(fr)
        eigv = np.array([complex(num) for num in eigv])
    
    # Data arrs
    if n_cyc:
        print(n_cyc)
        arr_d_cyclop = [arr_sol[2*n_cyc+1]]
        
        arr_d_theta_x  = [arr_sol[1]] * K
        for i in range(K):
            if i == n_cyc-1: continue
            arr_d_theta_x[i] = arr_sol[3+2*i]
        
        arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
        
        arr_all = np.array(arr_d_theta_y + arr_d_theta_x + arr_d_cyclop)
        
    else:
        arr_d_cyclop = [arr_sol[1]]
        arr_d_theta_x = [arr_sol[i] for i in range(3, N+2, 2)]
        arr_d_theta_y = [arr_sol[i] for i in range(N+2, 2*N, 2)]
        arr_all = np.array(arr_d_theta_x + arr_d_theta_y + arr_d_cyclop)
    
    # Diffs with dTheta1
    diff = arr_all - arr_all[-1]
    start_idx = (T-150) * 100
    
    # Create diagramm
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # View data
    shifted_hsv = shifted_hsv_cmap()
    cax = ax.imshow(diff[:, start_idx:], aspect='auto', cmap=shifted_hsv,
                    extent=[arr_t[start_idx], T, 0.5, N+0.5], origin='upper',
                    interpolation='none', vmin=-lim, vmax=lim)
    
    # Настраиваем оси
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('k', fontsize=12)
    ax.set_yticks(np.arange(1, 12))
    ax.set_yticklabels(np.arange(1, 12))

    # Color bar
    cbar = fig.colorbar(cax)
    cbar.set_label(r'$\dot{\theta}_k - \dot{\theta}_1$', rotation=270, labelpad=15, fontsize=16)

    # plt.title('Диаграмма разностей фаз', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_complex_numbers(complex_numbers, alp_eps = False):
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Рисуем единичную окружность (до всех точек)
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(circle)
    
    # Словарь для хранения количества точек в каждой позиции
    point_counts = {}
    
    # Обрабатываем каждое комплексное число
    for z in complex_numbers:
        # Округляем координаты до 6 знаков после запятой
        x = round(z.real, 5)
        y = round(z.imag, 5)
        pos = (x, y)
        
        # Увеличиваем счетчик для этой позиции
        if pos in point_counts:
            point_counts[pos] += 1
        else:
            point_counts[pos] = 1
    
    cntr = 0
    corner = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    # Рисуем точки и подписи
    for (x, y), count in point_counts.items():
        # Определяем цвет точки
        distance = np.sqrt(x**2 + y**2)
        if abs(distance - 1) < 1e-5:  # На окружности (с учетом погрешности)
            color = 'black'
        elif distance < 1:  # Внутри круга
            color = 'blue'
        else:  # Снаружи круга
            color = 'red'
        
        # Рисуем точку
        ax.scatter(x, y, color=color, s=50)
        
        # Добавляем текст с количеством точек
        ax.text(x + 0.05*corner[cntr][0], y + 0.05*corner[cntr][1], str(count), fontsize=8)
        cntr += 1
        if cntr > 3: cntr = 0
    
    # Настраиваем оси
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'$Re \lambda$')
    ax.set_ylabel(r'$Im \lambda$')
    if alp_eps:
        ax.set_title(fr'Eigenvalues for $\alpha_2$={alp_eps[0]:.3f}, $\varepsilon_2$={alp_eps[1]:.3f}')
    else:
        ax.set_title('Eigenvalues')
    
    # Автоматически подбираем границы, чтобы все точки были видны
    max_abs = max(abs(np.array(list(point_counts.keys()))).max(), 1.5)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    
    plt.show()


if __name__ == '__main__':
    # Font settings
    rc_fonts = {
        'font.size': 20,
        "text.usetex": True,
        'mathtext.default': 'regular',
        'text.latex.preamble': r"\usepackage{bm}",
        "font.family": "serif",
        "font.serif": "computer modern roman",
    }
    plt.rcParams.update(rc_fonts)
    

    # Common variables
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    K = (N-1) // 2
    
    
    # Eigvs drawing
    # with open('arr_for_C_eigv.txt', 'r') as fr:
    #     arr_for_eigv = json.load(fr)
    # for el in arr_for_eigv:
    #     # stable determination
    #     f_stab_det = full_syst_stability_determination(N, mu, epsilon1, alpha1, el[1], el[0])
    #     isStable, eigv = f_stab_det(el[2])
    #     plot_complex_numbers(eigv, alp_eps = [el[0], el[1]])
    
    
    #  Stable examples
    # stable_ex = [2.272566, 0.08, [-0.044528, 2.541541, 0.154365, 44.659187], True]
    # n_cyclop = 3
    
    # stable_ex = [1.9961058553661977, 0.08, [-0.03359744015159435, 2.6830724626023508, 0.17867310245657522, 47.05472020125708], True]
    # n_cyclop = 0
    
    # stable_ex = [2.5, 0.08, [-0.052096650048026875, 2.465919663536965, 0.13640774969849193, 43.45974058398454], True]
    # n_cyclop = 0
    
    # stable_ex = [-2.7037167544041196, 0.08, [-0.07604374953866636, 2.3880081439424705, 0.10081147902837394, 41.215762422260774], True] 
    # n_cyclop = 5
    
    # stable_eigv = create_example([N, mu, epsilon1, alpha1], stable_ex,
    #                               f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_cyclop)
    # draw_phase_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_cyclop)
    # draw_speed_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_cyclop, lim=0.82)
    
    
    # Unstable examples
    # unstable_ex = [2.649557, 0.08, [-0.056621, 2.430033, 0.126243, 42.912005], False]
    # n_cyclop = 1
    
    # unstable_ex = [1.6945129606215776, 0.08, [-0.0186408567985006, 2.9468677840319804, 0.1982185503528971, 52.112803327402816], False]
    # n_cyclop = 0
    
    # unstable_ex = [-2.854513201776431, 0.08, [-0.074726393131724, 2.372992393366901, 0.09952988138941797, 41.430975121830905], False]
    # n_cyclop = 2
    # d_lims = (-1, 0.4)
    
    # unstable_ex = [-1.5224779166543516, 0.08, [-0.01641520862168968, 3.1880745836709075, 0.18165200738585982, 48.47399967988414], False]
    # n_cyclop = 4
    # d_limits = (-0.6, 0.4)
    
    unstable_ex = [1.3929200658769576, 0.05040000000000007, [-0.0062030234757833175, 3.3212513049087726, 0.17482221063826936, 60.01523059447888], False]
    n_cyclop = 3
    d_limits = (-0.2, 0.12)
    rel_theta = True
    
    unstable_eigv = create_example([N, mu, epsilon1, alpha1], unstable_ex,
                                    f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_cyclop,
                                    d_limits, new_calc=False, relative_theta1=rel_theta)
    # draw_phase_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_cyclop)
    # draw_speed_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}.txt', n_cyclop,
    #                 lim=max(abs(x) for x in d_limits))
