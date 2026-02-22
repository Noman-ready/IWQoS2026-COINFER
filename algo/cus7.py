import numpy as np
import networkx as nx
import heapq
import time
from utils import initialize_parameters

class EnergyAwareTaskScheduler:
    def __init__(self, S, M, f_max, k_m, T_active, D_s,
                 Mem_req, Mem_avail, BW_max, sigma, P_tx, G, L, E_conn,
                 alpha=0.5, beta=0.5, mem_step=0):
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

        self._heuristic_cache = {}
        self._edge_weight_cache = {}

        self.f_vec = np.array(self.f_max)
        self.R_base = np.zeros((M, M))
        for m in range(M):
            for n in range(M):
                if m != n and E_conn[m, n] > 0:
                    snr = P_tx[m, n] * G[m, n] / (sigma ** 2)
                    self.R_base[m, n] = BW_max * np.log2(1 + snr)
        self.update_matrices()

    def update_matrices(self):
        self.T_comp = (self.f_vec.reshape(1, -1) / self.f_max.reshape(1, -1)) * self.T_active
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
                        best = min(best, cand)
                    elif self.E_conn[i, j] > 0:
                        R = self.R_base[i, j]
                        if R <= 0:
                            continue
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

        self._heuristic_cache.clear()
        self._edge_weight_cache.clear()

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
                        if R <= 0:
                            continue
                        trans_time = self.D_s[s - 1] / R + self.L[m_curr, m_next]
                        trans_energy = self.P_tx[m_curr, m_next] * trans_time
                    else:
                        continue
                    comp_time_next = self.T_comp[s, m_next]
                    e_comp_next = self.E_comp_mat[s, m_next]

                    edge_key = (node, (s + 1, m_next, new_mem_bucket), s - 1, m_curr, m_next)
                    if edge_key in self._edge_weight_cache:
                        edge_weight = self._edge_weight_cache[edge_key]
                    else:
                        edge_weight = self.alpha * (trans_time + comp_time_next) + self.beta * (
                                    e_comp_next + trans_energy)
                        self._edge_weight_cache[edge_key] = edge_weight

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
        T_comp_total = T_trans_total = E_comp_total = E_trans_total = 0.0
        for (s, m) in path:
            T_comp_total += self.T_comp[s - 1, m]
            E_comp_total += self.E_comp_mat[s - 1, m]
        for i in range(len(path) - 1):
            s_cur, m_cur = path[i]
            s_next, m_next = path[i + 1]
            if m_cur != m_next and self.E_conn[m_cur, m_next] > 0:
                R = self.R_base[m_cur, m_next]
                # if R <= 0:
                #     return {"objective_value": np.inf,
                #             "T_comp": np.inf, "T_trans": np.inf, "E_comp": np.inf, "E_trans": np.inf,
                #             "T_total": np.inf, "E_total": np.inf}
                t_trans = self.D_s[s_cur - 1] / R + self.L[m_cur, m_next]
                e_trans = self.P_tx[m_cur, m_next] * t_trans
                T_trans_total += t_trans
                E_trans_total += e_trans
        total_time = T_comp_total + T_trans_total
        E_total = E_comp_total + E_trans_total
        J = self.alpha * total_time + self.beta * E_total
        return {
            "objective_value": J,
            "T_comp": T_comp_total,
            "T_trans": T_trans_total,
            "E_comp": E_comp_total,
            "E_trans": E_trans_total,
            "T_total": total_time,
            "E_total": E_total
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
    raise RuntimeError("No path")


if __name__ == '__main__':
    S, M, f_max, k_m, T_active, D_s, Mem_req, Mem_avail, BW_max, sigma, P_tx, G, L, E_conn = initialize_parameters()

    alpha = 0.5
    beta = 0.5
    start_time = time.time()
    scheduler = EnergyAwareTaskScheduler(S, M, f_max, k_m, T_active, D_s,
                                         Mem_req, Mem_avail, BW_max, sigma,
                                         P_tx, G, L, E_conn,
                                         alpha=alpha, beta=beta)
    astar_start = time.time()
    best_value_astar, best_path_astar, best_res_astar = scheduler.solve_astar()
    astar_time = time.time() - astar_start
