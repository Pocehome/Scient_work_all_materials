import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from syst_without_reduc import (full_syst_stability_determination, 
                                create_full_syst_func, num_integration)
from system_dynamic import draw_start_end
# from reduc_syst_eq_state import reduc_syst_eq_state, fsolve


def claster_partion(arr_sol, n_cyc=0):
    # Data arrs  
    arr_without_cyc = []
    for i in range (N):
        if i == n_cyc: continue
        arr_without_cyc.append(arr_sol[2*i])
        arr_without_cyc.append(arr_sol[2*i+1])
    
    arr_cyclop = [arr_sol[2*n_cyc]]
    arr_d_cyclop = [arr_sol[2*n_cyc+1]]
    
    arr_theta_x = []
    arr_d_theta_x = []
    arr_theta_y = []
    arr_d_theta_y = []
    for i in range(2*K):
        if not arr_theta_x:
            arr_theta_x.append(arr_without_cyc[2*i])
            arr_d_theta_x.append(arr_without_cyc[2*i+1])
            continue
        
        if abs(np.mod(arr_theta_x[-1][95000] - arr_without_cyc[2*i][95000] + np.pi, 2*np.pi) - np.pi) < 0.1:
            arr_theta_x.append(arr_without_cyc[2*i])
            arr_d_theta_x.append(arr_without_cyc[2*i+1])
            
        else:
            arr_theta_y.append(arr_without_cyc[2*i])
            arr_d_theta_y.append(arr_without_cyc[2*i+1])
            
    return arr_theta_x, arr_d_theta_x, arr_cyclop, arr_d_cyclop, arr_theta_y, arr_d_theta_y


def draw_examples_with_theta1(arr_sol, arr_t, N, T, draw_start=False, tspan=(950, 1000)):
    # draw graph for all theta by T
    arr_thetas = np.array([arr_sol[2*i] for i in range(N)])
    arr_thetas = np.mod(arr_thetas + np.pi, 2*np.pi) - np.pi
    
    draw_start_end(arr_thetas, arr_t, r'$\theta_k$', T, tspan=tspan, draw_start=draw_start, draw_theta=True)


def draw_examples_relative_theta1(arr_sol, arr_t, N, T, n_cyc, draw_start=False):
    # Thetas arrays
    arr_theta_x, arr_d_theta_x, arr_cyclop, arr_d_cyclop, arr_theta_y, arr_d_theta_y = claster_partion(arr_sol, n_cyc)

    arr_thetas_xy = np.mod(np.array(arr_theta_x + arr_theta_y) - np.array(arr_cyclop) + np.pi, 2*np.pi) - np.pi
    # arr_d_thetas_xy = np.array(arr_d_theta_x + arr_d_cyclop + arr_d_theta_y) - np.array(arr_d_cyclop) 
    
    # Drawing
    # draw graph for d_thetas by T
    # draw_start_end(arr_d_thetas_xy, arr_t, r'$\dot{\theta}_k - \dot{\theta}_M$', T, draw_start=True, tspan=(200, 350))

    # draw graph for thetas by T
    draw_start_end(arr_thetas_xy, arr_t, r'$\theta_k - \theta_M$', T, draw_start=draw_start, tspan=(800, 1000), draw_theta=True, T_inter=200)
    

def save_example(arr_sol, arr_t, time, eigv, file_name, dir_name='Examples/'):
    eigv_write = [str(num) for num in eigv]
    with open(dir_name+file_name, 'w') as fw:
        json.dump([arr_sol, arr_t, time, eigv_write], fw)

def create_example(params, area_el, file_name, n_cyclop=0,  
                   new_calc=False, x0=0., start_difs=(0., 1.),
                   relative_theta1=1, draw_start=False, x_vs_y=True, t_snaps=[]):
    # print(file_name)
    # Variables
    N, mu, eps1, alp1 = params
    alp2 = area_el[0]
    eps2 = area_el[1]
    init_vec = np.array(area_el[2])
    T = init_vec[3]
    
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
        f_stab_det = full_syst_stability_determination(N, mu, eps1, alp1, eps2, alp2, x0)
        isStable, eigv = f_stab_det(init_vec)
        print(isStable)
        print(max(np.abs(eigv)))
        
        # numerical integration
        time = 1000
        # time = 500
        init_xy_vec = np.array([x0, init_vec[0], init_vec[1], init_vec[2]])
        init_full_syst_vec = np.array([0., 0.] + 
                               [init_xy_vec[0], init_xy_vec[1]]*K + 
                               [init_xy_vec[2], init_xy_vec[3]]*K)
        
        if start_difs:
            # init_full_syst_vec += np.random.uniform(start_difs[0], start_difs[1], 2*N)
            init_full_syst_vec[2::2] += np.random.uniform(start_difs[0], start_difs[1], N-1)
        
        rhs = create_full_syst_func(N, mu, eps1, alp1, eps2, alp2)
        arr_sol, arr_t = num_integration(rhs, init_full_syst_vec, time)
        
        save_example(arr_sol.tolist(), arr_t.tolist(), time, eigv.tolist(), file_name)        
    
    # Drawing
    if relative_theta1 == 1:
        draw_examples_with_theta1(arr_sol, arr_t, N, time, n_cyclop, draw_start)
    elif relative_theta1 == 2:
        draw_examples_relative_theta1(arr_sol, arr_t, N, time, n_cyclop, draw_start)
        
    # X vs Y
    if x_vs_y:
        draw_relative_phase_y_vs_x(arr_sol, arr_t, 300, 300+T+1, n_cyclop)
    
    # Snapshots
    if t_snaps:
        draw_phase_snapshots(arr_sol, arr_t, t_snaps, n_cyclop)
    
    return eigv


def shifted_hsv_cmap(shift=0.675):
    # Стандартная hsv палитра
    hsv = plt.get_cmap('hsv')
    
    # Создаем новые цвета со сдвигом
    x = np.linspace(0, 1, 256)
    new_colors = hsv((x + shift) % 1)
    
    return mcolors.ListedColormap(new_colors)

def draw_phase_diff(params, area_el, file_name, tspan, n_cyc=0, t_snaps=[]):
    with open('Examples/'+file_name, 'r') as fr:
        arr_sol, arr_t, T, eigv = json.load(fr)
    
    arr_theta_x, _, arr_cyclop, _, arr_theta_y, _ = claster_partion(arr_sol, n_cyc)
    arr_all = np.array(arr_theta_y + arr_cyclop + arr_theta_x)
    
    diff = np.mod(arr_all - arr_all[K] + np.pi, 2*np.pi) - np.pi
    
    T_start, T_finish = tspan
    start_idx, final_idx = int(T_start * 100), int(T_finish * 100)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    shifted_hsv = shifted_hsv_cmap()
    cax = ax.imshow(diff[:, start_idx:final_idx], aspect='auto', cmap=shifted_hsv,
                    extent=[arr_t[start_idx], T_finish, 0.5, N+0.5], origin='upper',
                    interpolation='none', vmin=-np.pi, vmax=np.pi)
    
    # 1. Отрисовка вертикальных линий и их символьных подписей t_i
    for i, t_val in enumerate(t_snaps):
        if T_start <= t_val <= T_finish:
            ax.axvline(x=t_val, color='black', linestyle='--', linewidth=3, alpha=0.7)
            # transform=ax.get_xaxis_transform() ставит y=0 на нижнюю границу графика
            # y=-0.01 смещает текст чуть-чуть под линию графика
            ax.text(t_val, -0.01, rf'$t_{i+1}$', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=font_size, color='black')
    
    ax.axhline(y=5.5, color='black', linewidth=2.5)
    ax.axhline(y=6.5, color='black', linewidth=2.5)
    
    # 2. Оставляем стандартную числовую шкалу
    x_ticks = np.arange(T_start, T_finish + 1, 50) 
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'${int(x)}$' for x in x_ticks])
    
    ax.set_xlabel(r'$t$', labelpad=20)
    ax.set_ylabel(r'$k$', rotation=0, labelpad=25, y=0.5, va='center')
    ax.set_yticks([1, 6, 11])

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\theta_k - \theta_6$', rotation=270, labelpad=35)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    plt.tight_layout()
    plt.show()

def draw_speed_diff(params, area_el, file_name, n_cyc=0, t_snaps=[]):    
    with open('Examples/'+file_name, 'r') as fr:
        arr_sol, arr_t, T, eigv = json.load(fr)
    
    _, arr_d_theta_x, _, arr_d_cyclop, _, arr_d_theta_y = claster_partion(arr_sol, n_cyc)
    arr_all = np.array(arr_d_theta_y + arr_d_cyclop + arr_d_theta_x)
    
    diff = arr_all - arr_all[K]
    
    T_start, T_finish = 300, 350
    start_idx, final_idx = int(T_start * 100), int(T_finish * 100)
    
    diff_slice = diff[:, start_idx:final_idx]
    lims = (diff_slice.min(), diff_slice.max())
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    cax = ax.imshow(diff_slice, aspect='auto', cmap='jet',
                    extent=[arr_t[start_idx], T_finish, 0.5, N+0.5], origin='upper',
                    interpolation='none', vmin=lims[0] - 0.05, vmax=lims[1] + 0.05)
    
    # 1. Отрисовка вертикальных линий и подписей t_i
    for i, t_val in enumerate(t_snaps):
        if T_start <= t_val <= T_finish:
            ax.axvline(x=t_val, color='black', linestyle='--', linewidth=3, alpha=0.7)
            ax.text(t_val, -0.01, rf'$t_{i+1}$', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=font_size, color='black')
    
    ax.axhline(y=5.5, color='black', linewidth=2.5)
    ax.axhline(y=6.5, color='black', linewidth=2.5)
    
    # 2. Стандартная числовая шкала
    x_ticks = np.arange(T_start, T_finish + 1, 50)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'${int(x)}$' for x in x_ticks])
    
    ax.set_xlabel(r'$t$', labelpad=60)
    ax.set_ylabel(r'$k$', rotation=0, labelpad=30, y=0.5, va='center')
    ax.set_yticks([1, 6, 11])

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\dot{\theta}_k - \dot{\theta}_{6}$', rotation=270, labelpad=40)
    cbar.set_ticks([lims[0], 0, lims[1]])
    cbar.set_ticklabels([rf'${round(lims[0], 1)}$', r'$0$', rf'${round(lims[1], 1)}$'])

    plt.tight_layout()
    plt.show()


def draw_eigenvalues(complex_numbers):
    # Создаем фигуру с запасом снизу под текст
    fig, ax = plt.subplots(figsize=(12, 14))
    
    complex_numbers = np.array(complex_numbers)
    abs_vals = np.abs(complex_numbers)
    
    # 1. Находим индекс критической точки (серая)
    idx_on_circle = np.argmin(np.abs(abs_vals - 1))
    
    # 2. Фиксируем границы, чтобы круг всегда был крупным
    limit = 1.5 
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # 3. Геометрия фона
    circle = plt.Circle((0, 0), 1, color='black', fill=False, 
                        linestyle='--', linewidth=3, zorder=1)
    ax.add_patch(circle)
    ax.axhline(0, color='black', linewidth=1.5, zorder=2)
    
    unstable_far = [] 

    # 4. Отрисовка точек
    for i, z in enumerate(complex_numbers):
        if i == idx_on_circle:
            continue
            
        is_inside = np.abs(z) < 1.0
        p_color = 'royalblue' if is_inside else 'tomato'
        
        # Проверка выхода за границы (с небольшим отступом)
        if np.abs(z.real) > limit - 0.15 or np.abs(z.imag) > limit - 0.15:
            # Вычисляем угол вылета
            angle = np.angle(z)
            
            # Координаты начала и конца стрелки у края графика
            # Конец стрелки (острие)
            x_end = (limit - 0.05) * np.cos(angle)
            y_end = (limit - 0.05) * np.sin(angle)
            # Начало стрелки (хвост)
            x_start = (limit - 0.25) * np.cos(angle)
            y_start = (limit - 0.25) * np.sin(angle)
            
            # Рисуем очевидную стрелку вместо треугольника
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                        arrowprops=dict(facecolor=p_color, edgecolor='black', 
                                        shrink=0.05, width=10, headwidth=25),
                        zorder=4)
            
            # Добавляем в список для текстовой подписи
            val_str = f"{z.real:.2f}{z.imag:+.2f}i" if abs(z.imag) > 1e-3 else f"{z.real:.2f}"
            unstable_far.append(val_str)
        else:
            # Обычные точки внутри масштаба
            ax.scatter(z.real, z.imag, color=p_color, s=400, 
                       edgecolors='black', linewidth=1.5, zorder=3)

    # 5. Серая точка (Критическая) - ВСЕГДА ПОВЕРХ ВСЕХ
    z_crit = complex_numbers[idx_on_circle]
    ax.scatter(z_crit.real, z_crit.imag, color='gray', s=400, 
               edgecolors='black', linewidth=1.5, zorder=3)

    # 6. Оформление осей
    ax.set_aspect('equal', 'box')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r'$\text{Re}\,\lambda$', labelpad=20)
    ax.set_ylabel(r'$\text{Im}\,\lambda$', labelpad=20)

    # 7. Текстовый блок под рисунком
    # if unstable_far:
    #     # Убираем дубликаты, если СЧ кратные, и сортируем для красоты
    #     unique_unstable = sorted(list(set(unstable_far)), key=lambda x: float(x.split('+')[0] if '+' in x else x.split('-')[0] if '-' in x and x[0]!='-' else x))
    #     text_val = r"$\lambda_{\text{out}} \in \{" + ", ".join(unique_unstable) + r"\}$"
        
    #     fig.text(0.5, 0.02, text_val, ha='center', fontsize=45, 
    #              bbox=dict(facecolor='white', alpha=0.8, edgecolor='tomato', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def draw_relative_phase_y_vs_x(arr_sol, arr_t, t_start, t_end, n_cyc=0):
    # 1. Индексы времени (шаг 0.01)
    idx_start = int(t_start * 100)
    idx_end = int(t_end * 100)
    idx_end = min(idx_end, len(arr_t))
    
    # 2. Получение данных через claster_partion
    # arr_theta_x/y - списки массивов для каждого элемента в кластере
    # arr_cyclop - список с одним массивом для уединенного элемента
    arr_theta_x, _, arr_cyclop, _, arr_theta_y, _ = claster_partion(arr_sol, n_cyc)
    
    # Берем данные за указанный период для первых элементов кластеров и уединенного
    phi_x_raw = np.array(arr_theta_x[0][idx_start:idx_end])
    phi_y_raw = np.array(arr_theta_y[0][idx_start:idx_end])
    phi_m_raw = np.array(arr_cyclop[0][idx_start:idx_end])
    
    # 3. Вычисление относительных фаз и нормировка к [-pi, pi]
    # Формула: x = (theta_X - theta_M) mod 2pi
    rel_x = np.mod(phi_x_raw - phi_m_raw + np.pi, 2 * np.pi) - np.pi
    rel_y = np.mod(phi_y_raw - phi_m_raw + np.pi, 2 * np.pi) - np.pi
    
    # 4. Обработка разрывов линии при перескоке через pi/-pi
    mask_x = np.abs(np.diff(rel_x)) > np.pi
    mask_y = np.abs(np.diff(rel_y)) > np.pi
    mask = mask_x | mask_y
    
    # Создаем массивы для отрисовки с разрывами
    x_plot = np.copy(rel_x[:-1])
    y_plot = np.copy(rel_y[:-1])
    x_plot[mask] = np.nan
    y_plot[mask] = np.nan

    # 5. Отрисовка
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Отрисовка траектории
    ax.plot(x_plot, y_plot, color='black', linewidth=3, zorder=3)
    
    # Настройка осей
    ax.set_aspect('equal', 'box')
    limit = np.pi + 0.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # Тики только -pi, 0, pi
    ticks = [-np.pi, 0, np.pi]
    tick_labels = [r'$-\pi$', r'$0$', r'$\pi$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    
    # Подписи осей в соответствии с физическим смыслом
    # ax.set_xlabel(r'$\theta_x - \theta_M$', labelpad=20)
    # ax.set_ylabel(r'$\theta_y - \theta_M$', rotation=0, labelpad=50, va='center')
    ax.set_xlabel(r'$x$', labelpad=10)
    ax.set_ylabel(r'$y$', rotation=0, labelpad=10, va='center')
    
    # Легкая сетка
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()


def draw_phase_snapshots(arr_sol, arr_t, t_points, n_cyc=0):
    cmap = shifted_hsv_cmap(0.675)
    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    
    arr_theta_x, arr_d_theta_x, arr_cyclop, arr_d_cyclop, arr_theta_y, arr_d_theta_y = claster_partion(arr_sol, n_cyc)

    for i, t in enumerate(t_points):
        ax = axes[i]
        idx = int(t * 100)
        
        # 1. Сетка и окружность
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=3, zorder=1)
        ax.add_patch(circle)
        ax.plot([0, 0], [-1, 1], color='gray', linestyle='--', linewidth=2, zorder=1)
        
        # 2. Уединенный элемент (M)
        phi_m = arr_cyclop[0][idx]
        v_m = arr_d_cyclop[0][idx]
        ax.scatter(0, 1, color=cmap(0.5), s=600, edgecolors='black', linewidth=2.5, zorder=1)
        ax.text(0.1, 0.1, r'$\varphi_k$', fontsize=45)
        
        def plot_cluster(thetas, velocities):
            # Фаза первого элемента нужна только для позиционирования стрелки
            rel_phi_ref = np.mod(thetas[0][idx] - phi_m + np.pi, 2 * np.pi) - np.pi
            
            num_el = len(thetas)
            offset = (num_el - 1) / 2
            
            # 3. Отрисовка элементов по ИНДИВИДУАЛЬНЫМ фазам и ЦВЕТАМ
            for k in range(num_el):
                # Относительная фаза именно этого (k-го) элемента
                rel_phi_k = np.mod(thetas[k][idx] - phi_m + np.pi, 2 * np.pi) - np.pi
                angle_k = np.pi/2 + rel_phi_k 
                
                # ИНДИВИДУАЛЬНЫЙ ЦВЕТ для каждого элемента k
                # Переводим [-pi, pi] в [0, 1] для cmap
                individual_color = cmap((rel_phi_k + np.pi) / (2 * np.pi))
                
                # Радиальное смещение для формирования "столбика"
                r = 1.0 + (k - offset) * 0.12
                x, y = r * np.cos(angle_k), r * np.sin(angle_k)
                
                ax.scatter(x, y, color=individual_color, s=450, edgecolors='black', linewidth=1.5, zorder=k+4)
            
            # 4. Отрисовка стрелки (по первому элементу)
            v_rel = velocities[0][idx] - v_m
            if abs(v_rel) > 0.001:
                direction = np.sign(v_rel)
                arc_half_width = 0.25 
                
                angle_ref = np.pi/2 + rel_phi_ref
                phi_start = angle_ref - direction * arc_half_width
                phi_end = angle_ref + direction * arc_half_width
                
                r_arrow = 1.0 + (num_el - offset) * 0.12 + 0.2
                x_s, y_s = r_arrow * np.cos(phi_start), r_arrow * np.sin(phi_start)
                x_e, y_e = r_arrow * np.cos(phi_end), r_arrow * np.sin(phi_end)
                
                curv = 0.3 * direction
                ax.annotate('', xy=(x_e, y_e), xytext=(x_s, y_s),
                            arrowprops=dict(arrowstyle="->", color='black', lw=4, 
                                            mutation_scale=30,
                                            connectionstyle=f"arc3,rad={curv}"),
                            zorder=2)

        plot_cluster(arr_theta_x, arr_d_theta_x)
        plot_cluster(arr_theta_y, arr_d_theta_y)
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.9, 1.9)
        ax.axis('off')
        ax.set_title(rf'$t = t_{i+1}$', y=-0.15, fontsize=55)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Font settings
    font_size = 50
    rc_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        'text.latex.preamble': r"\usepackage{bm} \usepackage{amsmath}",
        'mathtext.default': 'regular',
    }
    plt.rcParams.update(rc_fonts)
    

    # Common variables
    N = 11
    mu = 1.0
    epsilon1 = 1.0
    alpha1 = 1.7
    K = (N-1) // 2
    
    
    # Rotoreater examples
    #  Stable examples
    # stable_ex = [2.272566, 0.08, [-0.044528, 2.541541, 0.154365, 44.659187], True]
    # n_cyclop = 3
    
    # stable_ex = [1.9961058553661977, 0.08, [-0.03359744015159435, 2.6830724626023508, 0.17867310245657522, 47.05472020125708], True]
    # n_cyclop = 0
    
    # stable_ex = [2.5, 0.08, [-0.052096650048026875, 2.465919663536965, 0.13640774969849193, 43.45974058398454], True]
    # n_cyclop = 0
    
    # Point A
    stable_ex = [-2.7037167544041196, 0.08, [-0.07604374953866636, 2.3880081439424705, 0.10081147902837394, 41.215762422260774], True] 
    n_cyclop = 5
    tspan = [900, 1000]
    t_list = [908, 916, 926, 932]
    
    # stable_ex = [-1.597876, 0.1008, [-0.027601, 3.045886, 0.182489, 44.566605], True]
    # n_cyclop = 0
    
    stable_eigv = create_example([N, mu, epsilon1, alpha1], stable_ex,
                                  f'stable_example_alp2={stable_ex[0]:.5f}_eps2={stable_ex[1]:.5f}.txt',
                                  n_cyclop, new_calc=False, start_difs=(0.001, 0.01), 
                                  relative_theta1=1, draw_start=True, x_vs_y=False, t_snaps=t_list)
    # draw_phase_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}_eps2={stable_ex[1]:.5f}.txt', tspan, n_cyclop, t_snaps=t_list)
    # draw_speed_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}.txt', n_cyclop, t_list)
    # draw_eigenvalues(stable_eigv)
    
    
    # Unstable examples
    # Point B
    # unstable_ex = [2.649557, 0.08, [-0.056621, 2.430033, 0.126243, 42.912005], False]
    # n_cyclop = 1
    # tspan = [900, 1000]
    # t_list = [908, 920, 924, 928]
    
    # unstable_ex = [1.6945129606215776, 0.08, [-0.0186408567985006, 2.9468677840319804, 0.1982185503528971, 52.112803327402816], False]
    # n_cyclop = 9
    
    # unstable_ex = [-2.854513201776431, 0.08, [-0.074726393131724, 2.372992393366901, 0.09952988138941797, 41.430975121830905], False]
    # n_cyclop = 2
    
    # unstable_ex = [-1.5224779166543516, 0.08, [-0.01641520862168968, 3.1880745836709075, 0.18165200738585982, 48.47399967988414], False]
    # n_cyclop = 4
    
    # unstable_ex = [1.3929200658769576, 0.05040000000000007, [-0.0062030234757833175, 3.3212513049087726, 0.17482221063826936, 60.01523059447888], False]
    # n_cyclop = 3
    
    # unstable_ex = [-1.522478, 0.1008, [-0.015575, 3.216441, 0.179102, 48.253889], False]
    # n_cyclop = 4
    
    # unstable_ex = [-1.547611, 0.1496, [-0.019625, 3.186891, 0.178268, 45.556725], False]
    # n_cyclop = 0
    
    # Point C
    # unstable_ex = [-1.974867, 0.2, [-0.131715, 2.278865, 0.033062, 40.690468], False]
    # n_cyclop = 0
    # tspan = [900, 1000]
    # t_list = [906, 912, 918, 923]
    
    # unstable_eigv = create_example([N, mu, epsilon1, alpha1], unstable_ex,
    #                                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt',
    #                                 n_cyclop, new_calc=False, start_difs=(0.001, 0.01), 
    #                                 relative_theta1=0, draw_start=True, x_vs_y=False, t_snaps=t_list)
    # draw_phase_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt', tspan, n_cyclop, t_snaps=t_list)
    # draw_speed_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt', n_cyclop)  
    # draw_eigenvalues(unstable_eigv) 
    
    
    
    # Breater examples
    #  Stable examples
    
    # Point A
    # stable_ex = [-0.376992, 0.049, [0.271143, -1.006908, 0.158154, 27.506302], True]
    # sol0 = [2., stable_ex[2][1]]
    # n_cyclop = 8
    # tspan = [900, 1000]
    # t_list = [908, 916, 924, 932]
    
    # stable_ex = [-1.69646, 0.019, [0.31953, -0.714144, 0.17832, 30.17577], True]
    # sol0 = [2., stable_ex[2][1]]
    # n_cyclop = 8
    
    # stable_ex = [2.827433, 0.063, [0.268015, -0.408193, 0.121988, 37.334077], True]
    # sol0 = [2., stable_ex[2][1]]
    # n_cyclop = 0
    
    # stable_ex = [2.764601, 0.041, [0.279162, -0.561898, 0.137114, 32.039501], True]
    # sol0 = [2., stable_ex[2][1]]
    # n_cyclop = 0
    
    # sol = fsolve(reduc_syst_eq_state, sol0, args=(epsilon1, stable_ex[1], alpha1, stable_ex[0], N))
    # init_x = sol[0]
    # init_x = 2.
    
    # stable_eigv = create_example([N, mu, epsilon1, alpha1], stable_ex,
    #                               f'stable_example_alp2={stable_ex[0]:.5f}_eps2={stable_ex[1]:.5f}.txt',
    #                               n_cyclop, new_calc=False, x0=init_x, start_difs=False, 
    #                               relative_theta1=0, draw_start=True, x_vs_y=True, t_snaps=t_list)
    # draw_phase_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}_eps2={stable_ex[1]:.5f}.txt', tspan, n_cyclop, t_snaps=t_list)
    # draw_speed_diff([N, mu, epsilon1, alpha1], stable_ex, 
    #                 f'stable_example_alp2={stable_ex[0]:.5f}_eps2={stable_ex[1]:.5f}.txt', n_cyclop)
    # draw_eigenvalues(stable_eigv)
    
    
    # Unstable examples
    # unstable_ex = [-2.387611, 0.039, [0.320957, -0.470981, 0.166504, 38.797184], False]
    # sol0 = [2., unstable_ex[2][1]]
    # n_cyclop = 0
    
    # Point B
    # unstable_ex = [-1.31947, 0.1494, [0.440857, -0.74913, 0.314026, 27.02518], False]
    # sol0 = [2., unstable_ex[2][1]]
    # n_cyclop = 10
    # tspan = (900, 1000)
    # t_list = [910, 918, 926, 934]
    
    # Point C
    # unstable_ex = [-0.86708, 0.1294, [0.057213, -1.485202, 0.084569, 22.477423], False]
    # sol0 = [2., unstable_ex[2][1]]
    # n_cyclop = 9
    # # n_cyclop = 0
    # # n_cyclop = 3
    # tspan = (200, 350)
    # t_list = [240, 265, 280, 290]
    
    # unstable_ex = [2.848975, 0.094371, [0.287714, -0.58644, 0.174879, 44.101412], False]
    # sol0 = [2., unstable_ex[2][1]]
    # n_cyclop = 0
    
    # sol = fsolve(reduc_syst_eq_state, sol0, args=(epsilon1, unstable_ex[1], alpha1, unstable_ex[0], N))
    # init_x = sol[0]
    # init_x = 2.
    
    # unstable_eigv = create_example([N, mu, epsilon1, alpha1], unstable_ex,
    #                                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt',
    #                                 n_cyclop, new_calc=False, x0=init_x, start_difs=(0.001, 0.01), 
    #                                 relative_theta1=0, draw_start=True, x_vs_y=False, t_snaps=t_list)
    # draw_phase_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt', tspan, n_cyclop, t_snaps=t_list)
    # draw_speed_diff([N, mu, epsilon1, alpha1], unstable_ex, 
    #                 f'unstable_example_alp2={unstable_ex[0]:.5f}_eps2={unstable_ex[1]:.5f}.txt', n_cyclop)  
    # draw_eigenvalues(unstable_eigv)  
