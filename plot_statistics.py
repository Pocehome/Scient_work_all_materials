import numpy as np
import json
import os
import matplotlib.pyplot as plt
from generate_examples import draw_examples_with_theta1


def draw_regime_probabilities(eps2, n_points, n_samples):
    # 1. Формирование пути к файлу в папке StatisticResults
    results_dir = "StatisticResults"
    filename = f"stats_eps2_{eps2:.4f}_pts_{n_points}_smp_{n_samples}.json"
    filepath = os.path.join(results_dir, filename)

    if not os.path.exists(filepath):
        print(f"Ошибка: Файл {filepath} не найден!")
        return

    # 2. Загрузка и обработка данных
    with open(filepath, 'r') as f:
        data = json.load(f)

    raw_results = data["raw_results"]
    alp2_range = np.linspace(-np.pi, np.pi, n_points)
    
    # Списки для хранения вычисленных вероятностей
    p_stat, p_breather, p_roto, p_other = [], [], [], []
    
    regimes = ["Stationary", "Breather", "Rotobreather", "Other"]
    
    for res_list in raw_results:
        count = len(res_list)
        if count == 0:
            for p_list in [p_stat, p_breather, p_roto, p_other]: p_list.append(0)
            continue
            
        p_stat.append(res_list.count("Stationary") / count)
        p_breather.append(res_list.count("Breather") / count)
        p_roto.append(res_list.count("Rotobreather") / count)
        p_other.append(res_list.count("Other") / count)

    # 3. Настройка шрифтов
    font_size = 50
    rc_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": 35,
        'text.latex.preamble': r"\usepackage{bm} \usepackage{amsmath}",
    }
    plt.rcParams.update(rc_fonts)

    fig, ax = plt.subplots(figsize=(22, 12))

    # 4. Отрисовка полых маркеров
    # Stationary (Круги)
    ax.scatter(alp2_range, p_stat, marker='o', s=450, linewidths=3, 
               edgecolors='royalblue', facecolors='none', label='Stationary', zorder=3)
    
    # Breather (Крестики - только цвет линий)
    ax.scatter(alp2_range, p_breather, marker='x', s=450, linewidths=4, 
               color='forestgreen', label='Breather', zorder=3)
    
    # Rotobreather (Треугольники)
    ax.scatter(alp2_range, p_roto, marker='^', s=500, linewidths=3, 
               edgecolors='tomato', facecolors='none', label='Rotobreather', zorder=3)
    
    # Other (Квадраты)
    ax.scatter(alp2_range, p_other, marker='s', s=400, linewidths=3, 
               edgecolors='black', facecolors='none', label='Other', zorder=3)

    # 5. Оформление осей
    ax.set_xlabel(r'$\alpha_2$', labelpad=20)
    ax.set_ylabel(r'$P$', rotation=0, labelpad=40, va='center')
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$0$', r'$+\pi$'])
    
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, shadow=True, borderpad=1)

    plt.tight_layout()
    # Автоматическое сохранение графика в ту же папку (опционально)
    # plt.savefig(os.path.join(results_dir, filename.replace('.json', '.png')))
    plt.show()


def draw_Other(fname):
    results_dir = "StatisticResults\Debug_Other"
    filepath = os.path.join(results_dir, fname)

    if not os.path.exists(filepath):
        print(f"Ошибка: Файл {filepath} не найден!")
        return

    # 2. Загрузка и обработка данных
    with open(filepath, 'r') as f:
        # json.dump([sol_save, t_save, T_TOTAL, [str(e) for e in eigv]], f)
        arr_sol, arr_t, T_max, eigv = json.load(f)
        draw_examples_with_theta1(arr_sol, arr_t, 11, T_max, tspan=(2900, 3000))


if __name__ == "__main__":
    # draw_regime_probabilities(eps2=0.01, n_points=1, n_samples=5)
    # draw_regime_probabilities(eps2=0.01, n_points=20, n_samples=10)
    
    nameOther = 'Other_a2=-0.1653_smp=1.txt'
    draw_Other(nameOther)
