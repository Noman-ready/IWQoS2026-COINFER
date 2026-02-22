import time
import numpy as np
from itertools import product
from utils import initialize_parameters


def exhaustive_search():
    best_obj_ex = float('inf')
    best_assign_ex = None

    total_combinations = M ** B

    for i, assign in enumerate(product(range(M), repeat=B)):
        mem_ok = True
        mem_usages = [0] * M
        for j in range(M):
            for b in range(B):
                if assign[b] == j:
                    mem_usages[j] += Mem_req[b][j]
            if mem_usages[j] > Mem_avail[j]:
                mem_ok = False
                break
        if not mem_ok:
            continue

        total_time = 0
        for b in range(B):
            total_time += T_comp[b][assign[b]]

        for k in range(K):
            m = assign[k]
            n = assign[k + 1]
            if m != n:
                total_time += D_s[k] / BW[m][n] + L[m][n]

        if total_time < best_obj_ex:
            best_obj_ex = total_time
            best_assign_ex = assign

    if best_assign_ex is not None:
        mem_usage_final = [0] * M
        for j in range(M):
            for b in range(B):
                if best_assign_ex[b] == j:
                    mem_usage_final[j] += Mem_req[b][j]

    print(f"Best objective: {best_obj_ex:.3f}")
    for idx in range(B):
        print(f"Fragment {idx} -> Device {best_assign_ex[idx]}")
    print(f"Memory usage: {mem_usage_final} vs capacity {Mem_avail}")

    return best_obj_ex, best_assign_ex


if __name__ == '__main__':
    start = time.time()
    best_ex, best_assign_ex = exhaustive_search()
    end = time.time()
    print(f'Exhaustive Search Time: {end - start:.4f}s')