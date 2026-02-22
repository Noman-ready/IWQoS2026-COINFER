import numpy as np
import networkx as nx
import heapq
import time
import os
import csv
import warnings
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
from utils import initialize_parameters


warnings.filterwarnings("ignore")
class EnergyAwareTaskScheduler:
    def __init__(self, S, M, f_max, k_m, T_active, D_s,
                 Mem_req, Mem_avail, BW_max, sigma, P_tx, G, L, E_conn,
                 alpha=1.0, beta=0.1, mem_step=5.0):
        self.S = S
        self.M = M
        self.f_max = f_max
        self.k_m = k_m
        self.T_active = T_active
        self.D_s = D_s
        self.Mem_req = Mem_req
        self.Mem_avail = Mem_avail
        self.BW_max = BW_max
        self.sigma = sigma
        self.P_tx = P_tx
        self.G = G
        self.L = L
        self.E_conn = E_conn
        self.alpha = alpha
        self.beta = beta
        self.mem_step = mem_step
        self.f_vec = np.array(self.f_max)
        self.R_base = np.zeros((M, M))
        for m in range(M):
            for n in range(M):
                if m != n and E_conn[m, n] > 0:
                    snr = P_tx[m, n] * G[m, n] / (sigma ** 2)
                    self.R_base[m, n] = BW_max * np.log2(1 + snr)

        self.update_matrices()
        self.compute_reference_values()

    def compute_reference_values(self):
        self.set_frequencies(self.f_max)
        self._ref_mode = True
        val_ref, _, res_ref = self.solve_astar()
        self.T_ref = res_ref["T_total"]
        self.E_ref = res_ref["E_total"]
        self._ref_mode = False
        # print(f"[Ref] T_ref={self.T_ref:.2f}, E_ref={self.E_ref:.2e}")

    def update_matrices(self):
        self.T_comp = (self.f_max.reshape(1, -1) / self.f_vec.reshape(1, -1)) * self.T_active
        self.P_comp = self.k_m * (self.f_vec ** 3)
        self.E_comp_mat = np.zeros((self.S, self.M))
        for s in range(self.S):
            for m in range(self.M):
                self.E_comp_mat[s, m] = self.P_comp[m] * self.T_comp[s, m]

        self.c_k_min = np.zeros(self.S)
        for k in range(self.S):
            comp_term = self.alpha * self.T_comp[k, :] + self.beta * (self.P_comp * self.T_comp[k, :])
            self.c_k_min[k] = np.min(comp_term)

        self.t_k_min = np.zeros(self.S - 1)
        for k in range(self.S - 1):
            best = np.inf
            for i in range(self.M):
                for j in range(self.M):
                    if i == j:
                        cand = 0.0
                    elif self.E_conn[i, j] > 0:
                        R = self.R_base[i, j]
                        t_trans = self.D_s[k] / R + self.L[i, j]
                        e_trans = self.P_tx[i, j] * t_trans
                        cand = self.alpha * t_trans + self.beta * e_trans
                        best = min(best, cand)
            self.t_k_min[k] = best if best < np.inf else 0.0

        self.c_k_min_suffix = np.zeros(self.S + 1)
        self.t_k_min_suffix = np.zeros(self.S + 1)
        for k in range(self.S - 1, -1, -1):
            self.c_k_min_suffix[k] = self.c_k_min[k] + self.c_k_min_suffix[k + 1]
        for k in range(self.S - 2, -1, -1):
            self.t_k_min_suffix[k] = self.t_k_min[k] + self.t_k_min_suffix[k + 1]

    def set_frequencies(self, f_vec):
        self.f_vec = np.array(f_vec)
        self.update_matrices()

    def _build_graph(self):
        G = nx.DiGraph()
        for m in range(self.M):
            if self.Mem_req[0, m] <= self.Mem_avail[m]:
                mem_used = self.Mem_req[0, m]
                mem_bucket = self.get_mem_bucket(mem_used)
                G.add_node((1, m, mem_bucket))

        for s in range(1, self.S):
            layer_nodes = [n for n in G.nodes if n[0] == s]
            for node in layer_nodes:
                s_curr, m_curr, mem_used_bucket = node
                mem_used_curr = mem_used_bucket * self.mem_step
                for m_next in range(self.M):
                    if m_next != m_curr:
                        new_mem_used = self.Mem_req[s, m_next]
                    else:
                        new_mem_used = mem_used_curr + self.Mem_req[s, m_next]
                    if new_mem_used > self.Mem_avail[m_next]:
                        continue
                    new_mem_bucket = self.get_mem_bucket(new_mem_used)
                    if m_curr == m_next:
                        trans_time = trans_energy = 0.0
                    elif self.E_conn[m_curr, m_next] > 0:
                        R = self.R_base[m_curr, m_next]
                        trans_time = self.D_s[s - 1] / R + self.L[m_curr, m_next]
                        trans_energy = self.P_tx[m_curr, m_next] * trans_time
                    else:
                        continue
                    comp_time_next = self.T_comp[s, m_next]
                    e_comp_next = self.E_comp_mat[s, m_next]
                    edge_weight = self.alpha * (trans_time + comp_time_next) + self.beta * (e_comp_next + trans_energy)
                    new_node = (s + 1, m_next, new_mem_bucket)
                    if not G.has_node(new_node):
                        G.add_node(new_node)
                    G.add_edge(node, new_node, weight=edge_weight)
        return G

    def solve_astar(self):
        G = self._build_graph()
        start_nodes = [n for n in G.nodes if n[0] == 1]
        end_nodes = [n for n in G.nodes if n[0] == self.S]
        best_value = np.inf
        best_path, best_res = None, None
        for start in start_nodes:
            for end in end_nodes:
                try:
                    path_nodes = custom_astar(G, start, end, self.preferenceDegree)
                    ordered = [(n[0], n[1]) for n in path_nodes]
                    res = self.calculate_objective_function(ordered)
                    if res["objective_value"] < best_value:
                        best_value = res["objective_value"]
                        best_path, best_res = ordered, res
                except RuntimeError:
                    continue
        return best_value, best_path, best_res

    def calculate_objective_function(self, path):
        T_comp_total, T_trans_total, E_comp_total, E_trans_total = 0.0, 0.0, 0.0, 0.0
        for (s, m) in path:
            T_comp_total += self.T_comp[s - 1, m]
            E_comp_total += self.E_comp_mat[s - 1, m]
        for i in range(len(path) - 1):
            s_cur, m_cur = path[i]
            s_next, m_next = path[i + 1]
            if m_cur != m_next and self.E_conn[m_cur, m_next] > 0:
                R = self.R_base[m_cur, m_next]
                t_trans = self.D_s[s_cur - 1] / R + self.L[m_cur, m_next]
                e_trans = self.P_tx[m_cur, m_next] * t_trans
                T_trans_total += t_trans
                E_trans_total += e_trans

        total_time = T_comp_total + T_trans_total
        E_total = E_comp_total + E_trans_total

        if hasattr(self, "_ref_mode") and self._ref_mode:
            norm_time, norm_energy = total_time, E_total
            J = self.alpha * total_time + self.beta * E_total
        else:
            norm_time = total_time / self.T_ref
            norm_energy = E_total / self.E_ref
            J = self.alpha * norm_time + self.beta * norm_energy

        return {
            "objective_value": J,
            "T_comp": T_comp_total,
            "T_trans": T_trans_total,
            "E_comp": E_comp_total,
            "E_trans": E_trans_total,
            "T_total": total_time,
            "E_total": E_total,
            "norm_time": norm_time,
            "norm_energy": norm_energy
        }



def custom_astar(graph, start, end, heuristic_func):
    open_set = []
    heapq.heappush(open_set, (heuristic_func(start), start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for neighbor in graph.neighbors(current):
            tentative = g_score[current] + graph[current][neighbor]["weight"]
            if tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic_func(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))
    raise RuntimeError("No path found")


def make_optuna_objective(scheduler, f_max,step=0.1):
    def objective(trial):
        f_vec = np.zeros(scheduler.M)
        for m in range(scheduler.M):
            f_vec[m] = trial.suggest_float(f"f_{m}", 0.5 * f_max[m], f_max[m],step=step)
        scheduler.set_frequencies(f_vec)
        try:
            val, _, _ = scheduler.solve_astar()
            return val
        except Exception:
            return 1e12
    return objective

batch_values = []


def logging_callback(study, trial):
    global batch_values
    if trial.value is not None:
        batch_values.append(trial.value)

    if len(study.trials) % 30 == 0 and len(batch_values) > 0:
        avg_val = np.mean(batch_values)
        batch_values = []

def early_stop_callback(study, trial):
    N_window, eps = 0,0
    trials = [t.value for t in study.trials if t.value is not None]
    if len(trials) > N_window:
        recent = trials[-N_window:]
        std_recent = np.std(recent)
        if len(trials) % 20 == 0:
            print(f"[Monitor] iter={len(trials)} | std_recent={std_recent:.3e} | best={min(recent):.4f}")
        if std_recent < eps:
            tpe_status["stopped_iter"] = len(trials)
            tpe_status["stop_reason"] = f"Converged(std={std_recent:.2e})"
            print(f"Early stopping")
            study.stop()


if __name__ == "__main__":
    S, M, f_max, k_m, T_active, D_s, Mem_req, Mem_avail, BW_max, sigma, P_tx, G, L, E_conn = initialize_parameters()

    N_TRIALS = 0
    SEED = 42

    alpha_list = np.arange(0.1, 2.01, 0.1)
    beta_list = np.arange(0.1, 2.01, 0.1)

    fieldnames = [
        "alpha", "beta", "n_trials", "tpe_time",
        "stopped_iter", "stop_reason",
        "best_obj", "best_path",
        "T_comp", "T_trans", "E_comp", "E_trans",
        "T_total", "E_total",
        "norm_time", "norm_energy",
        "best_f_vec"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        T_matrix = np.zeros((len(beta_list), len(alpha_list)))
        E_matrix = np.zeros((len(beta_list), len(alpha_list)))

        for i, alpha in enumerate(alpha_list):
            for j, beta in enumerate(beta_list):
                tpe_status = {"stopped_iter": None, "stop_reason": None}
                scheduler = EnergyAwareTaskScheduler(S, M, f_max, k_m, T_active, D_s,
                                                     Mem_req, Mem_avail, BW_max, sigma,
                                                     P_tx, G, L, E_conn,
                                                     alpha=alpha, beta=beta)

                sampler = TPESampler(seed=SEED)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                objective = make_optuna_objective(scheduler, f_max)

                t0 = time.time()
                study.optimize(
                    objective,
                    n_trials=N_TRIALS,
                    show_progress_bar=False,
                    callbacks=[early_stop_callback, logging_callback]
                )
                t1 = time.time()

                best_trial = study.best_trial
                best_f_vec = [best_trial.params.get(f"f_{m}", f_max[m]) for m in range(M)]
                scheduler.set_frequencies(best_f_vec)
                best_value, best_path, best_res = scheduler.solve_astar()

                row = {
                    "alpha": alpha,
                    "beta": beta,
                    "n_trials": N_TRIALS,
                    "tpe_time": t1 - t0,
                    "stopped_iter": tpe_status.get("stopped_iter", len(study.trials)),
                    "stop_reason": tpe_status.get("stop_reason", "MaxTrialsReached"),
                    "best_obj": best_value,
                    "best_path": str(best_path),
                    "T_comp": best_res["T_comp"],
                    "T_trans": best_res["T_trans"],
                    "E_comp": best_res["E_comp"],
                    "E_trans": best_res["E_trans"],
                    "T_total": best_res["T_total"],
                    "E_total": best_res["E_total"],
                    "norm_time": best_res["norm_time"],
                    "norm_energy": best_res["norm_energy"],
                    "best_f_vec": str(best_f_vec)
                }

                writer.writerow(row)

                # 填充矩阵
                T_matrix[j, i] = best_res["T_total"]
                E_matrix[j, i] = best_res["E_total"]


