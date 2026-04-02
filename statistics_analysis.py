import numpy as np
import json
import os
import multiprocessing
import sys
from syst_without_reduc import create_full_syst_func, num_integration

# --- Настройки системы ---
N_OSC = 11
K = (N_OSC - 1) // 2
MU = 1.0
EPS1 = 1.0
ALP1 = 1.7
T_STEP = 1000 
T_MAX = 5000

def get_random_ic():
    ic = np.empty(2 * N_OSC)
    ic[0::2] = np.random.uniform(-np.pi, np.pi, N_OSC)
    ic[1::2] = np.random.uniform(-1, 1, N_OSC)
    return ic

def classify_regime(arr_sol, arr_t):
    win_idx = -10000 
    final_phases_mod = np.mod(np.array([np.mean(sol[-5000:]) for sol in arr_sol[0::2]]), 2*np.pi)
    
    def find_clusters(phases, threshold=0.3):
        clusters = []
        used = set()
        for i in range(len(phases)):
            if i in used: continue
            curr = [i]
            used.add(i)
            for j in range(i + 1, len(phases)):
                diff = np.abs(np.mod(phases[i] - phases[j] + np.pi, 2*np.pi) - np.pi)
                if diff < threshold:
                    curr.append(j)
                    used.add(j)
            clusters.append(curr)
        return sorted(clusters, key=len)

    clusters = find_clusters(final_phases_mod)
    if len(clusters) != 3 or len(clusters[0]) != 1 or len(clusters[1]) != 5:
        return "Other"

    m_idx = clusters[0][0]
    x_idx = clusters[1][0]
    phi_m, phi_x = np.array(arr_sol[2*m_idx][win_idx:]), np.array(arr_sol[2*x_idx][win_idx:])
    rel_x = phi_x - phi_m

    variation = np.max(rel_x) - np.min(rel_x)
    displacement = np.abs(rel_x[-1] - rel_x[0])

    if variation < 0.05: return "Stationary"
    if displacement > np.pi: return "Rotobreather"
    return "Breather"

def worker_task(task_info):
    idx_alp, alp2, sample_idx, eps2, results_dir = task_info
    rhs = create_full_syst_func(N_OSC, MU, EPS1, ALP1, eps2, alp2)
    
    initial_ic = get_random_ic()
    current_ic = initial_ic.copy()
    accumulated_time = 0
    
    while accumulated_time < T_MAX:
        try:
            sol, t = num_integration(rhs, current_ic, T_STEP)
            regime = classify_regime(sol, t)
            
            is_final_attempt = (accumulated_time + T_STEP) >= T_MAX
            
            if regime != "Other" or is_final_attempt:
                if regime == "Other":
                    debug_dir = os.path.join(results_dir, "Debug_Other")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    t_shifted = t + accumulated_time
                    min_l = min(len(s) for s in sol)
                    debug_name = f"Other_a2={alp2:.4f}_eps2={eps2:.4f}_seg={accumulated_time}-{accumulated_time+T_STEP}.txt"
                    
                    # Сохранение БЕЗ вычисления собственных чисел
                    with open(os.path.join(debug_dir, debug_name), 'w') as f:
                        json.dump([
                            [s[:min_l].tolist() for s in sol], 
                            t_shifted[:min_l].tolist(), 
                            float(accumulated_time + T_STEP), 
                            [], # Собственные числа больше не считаются, пишем пустой список
                            initial_ic.tolist() 
                        ], f)
                
                return idx_alp, alp2, regime, accumulated_time + T_STEP
            
            # Подготовка к следующему шагу (1000 ед. времени)
            current_ic = np.array([s[-1] for s in sol])
            accumulated_time += T_STEP
            
        except Exception:
            return idx_alp, alp2, "Other", accumulated_time + T_STEP

class StatisticsManager:
    def __init__(self, eps2, n_points, n_samples):
        self.eps2 = eps2
        self.n_points = n_points
        self.n_samples = n_samples
        self.alp2_range = np.linspace(-np.pi, np.pi, n_points).tolist()
        self.results_dir = "StatisticResults"
        os.makedirs(self.results_dir, exist_ok=True)
        self.filepath = os.path.join(self.results_dir, f"stats_eps2_{eps2:.4f}_pts_{n_points}_smp_{n_samples}.json")
        self.data = {"metadata": {"eps2": eps2, "n_points": n_points, "target_samples": n_samples, "T_step": T_STEP, "T_max": T_MAX},
                     "raw_results": [[] for _ in range(n_points)]}
        self.load_existing()

    def load_existing(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                loaded = json.load(f)
                if len(loaded["raw_results"]) == self.n_points:
                    self.data["raw_results"] = loaded["raw_results"]
                    print(f"Загружено: {sum(len(r) for r in self.data['raw_results'])} результатов.")

    def save(self):
        temp = self.filepath + ".tmp"
        with open(temp, 'w') as f:
            json.dump(self.data, f, indent=4)
        os.replace(temp, self.filepath)

    def run(self):
        tasks = []
        for i, alp2 in enumerate(self.alp2_range):
            for s_idx in range(len(self.data["raw_results"][i]), self.n_samples):
                tasks.append((i, alp2, s_idx, self.eps2, self.results_dir))

        if not tasks:
            print("Все задачи выполнены.")
            return

        total_tasks = len(tasks)
        print(f"Запуск: {multiprocessing.cpu_count()} ядер. Режим: Адаптивное T (без СЧ для Other).")

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        try:
            for i, result in enumerate(pool.imap_unordered(worker_task, tasks)):
                idx_alp, alp2_val, regime, t_reached = result
                self.data["raw_results"][idx_alp].append(regime)
                
                percent = (i + 1) / total_tasks * 100
                sys.stdout.write(f"\rПрогресс: {percent:.1f}% | {i+1}/{total_tasks} | a2: {alp2_val:7.4f} | T: {t_reached} | Режим: {regime:13}")
                sys.stdout.flush()
                
                if (i + 1) % 10 == 0: self.save()
            self.save()
            print(f"\nГотово. Результаты в {self.filepath}")
        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            pool.terminate()
        finally:
            pool.close()
            pool.join()

if __name__ == "__main__":
    manager = StatisticsManager(eps2=0.01, n_points=50, n_samples=20)
    manager.run()