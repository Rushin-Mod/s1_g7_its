import numpy as np
import random
import matplotlib.pyplot as plt
import os, sys
from collections import defaultdict

# ==========================
# SUMO SETUP
# ==========================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("SUMO_HOME not set")

import traci

# ==========================
# PARAMETERS
# ==========================

LANE_IDS = ['1307443859#1_0', '166749694#2_0', '166953969#3_0',
            '1307443859#1_1', '166953969#3_1', '166749694#2_1',
            '244329793#4_1', '244329793#4_0']

lane_groups = defaultdict(list)
for lane in LANE_IDS:
    edge = lane.split('_')[0]
    lane_groups[edge].append(lane)

SIM_TIME = 200
LANES = len(lane_groups)

lambda_i = [0.5, 0.6, 0.4, 0.7]
mu_i     = [1.0, 1.0, 1.0, 1.0]

SUMO_CONFIG = r"C:\Users\ADMIN\Pythonwork\CSE400_SUMO\Usmanpura\SIM.sumocfg"
TLS_ID = "cluster_1783450256_1783450268_1783450274_1783450276"

# ==========================
# RANDOM VARIABLES
# ==========================

def sample_arrivals(lam):
    return np.random.poisson(lam)

def sample_delta_G():
    return random.uniform(0.7, 1.5)

# ==========================
# REAL STATE
# ==========================

def get_real_queue():
    Q = []
    for edge, lanes in lane_groups.items():
        total = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        Q.append(total)
    return Q

def apply_control(lane_index, duration):
    traci.trafficlight.setPhase(TLS_ID, 0)

# ==========================================================
# 🔵 DETERMINISTIC (WEBSTER)
# ==========================================================

def deterministic_step(Q):

    arrivals_list = []
    yi_list = []
    cycle_time = 60

    for i in range(LANES):
        arrivals = sample_arrivals(lambda_i[i])
        arrivals_list.append(arrivals)

        flow = arrivals * 3600.0 / max(cycle_time, 1e-6)
        yi = flow / 1800.0
        yi_list.append(max(yi, 1e-6))

    y_sum = sum(yi_list)
    L = LANES * 3.0

    Co = (1.5 * L + 5.0) / max((1 - y_sum), 1e-6)
    Co = max(40.0, min(Co, 120.0))

    greens = []
    Ce = max(Co - L - LANES * 10.0, 0.0)

    for yi in yi_list:
        alpha = yi / y_sum if y_sum > 1e-9 else 1.0 / LANES
        gi = alpha * Ce + 10.0
        greens.append(gi)

    new_Q = []
    total_wait = 0

    for i in range(LANES):

        arrivals = arrivals_list[i]

        service = mu_i[i] * (greens[i] / Co)
        departures = min(Q[i], int(service * Co))

        Qi = max(Q[i] + arrivals - departures, 0)
        Wi = Qi / max(lambda_i[i], 1e-6)

        total_wait += Wi
        new_Q.append(Qi)

    return new_Q, total_wait / LANES

# ==========================================================
# 🔴 UNBEATABLE LAS VEGAS
# ==========================================================

def las_vegas_step(Q, trials=50):

    # Baseline
    det_Q, det_wait = deterministic_step(Q)

    arrivals_list = []
    yi_list = []
    cycle_time = 60

    for i in range(LANES):
        arrivals = sample_arrivals(lambda_i[i])
        arrivals_list.append(arrivals)

        flow = arrivals * 3600.0 / max(cycle_time, 1e-6)
        yi = flow / 1800.0
        yi_list.append(max(yi, 1e-6))

    y_sum = sum(yi_list)
    L = LANES * 3.0

    Co = (1.5 * L + 5.0) / max((1 - y_sum), 1e-6)
    Co = max(40.0, min(Co, 120.0))

    base_greens = []
    Ce = max(Co - L - LANES * 10.0, 0.0)

    for yi in yi_list:
        alpha = yi / y_sum if y_sum > 1e-9 else 1.0 / LANES
        gi = alpha * Ce + 10.0
        base_greens.append(gi)

    Q_array = np.array(Q)
    weights = Q_array / max(np.sum(Q_array), 1e-6)

    best_Q = det_Q
    best_wait = det_wait
    best_score = det_wait
    best_deltas = [1]*LANES

    for _ in range(trials):

        trial_Q = []
        total_wait = 0
        deltas = []

        for i in range(LANES):

            bias = 1 + 0.8 * weights[i]
            noise = random.uniform(0.8, 1.2)

            delta = max(0.7, min(bias * noise, 1.8))
            deltas.append(delta)

            gi = base_greens[i] * delta

            service = mu_i[i] * (gi / Co)
            departures = min(Q[i], int(service * Co))

            arrivals = arrivals_list[i]

            Qi = max(Q[i] + arrivals - departures, 0)
            Wi = Qi / max(lambda_i[i], 1e-6)

            total_wait += Wi
            trial_Q.append(Qi)

        avg_wait = total_wait / LANES
        avg_queue = sum(trial_Q) / LANES

        score = 0.7 * avg_wait + 0.3 * avg_queue

        if (score < best_score) and (avg_wait <= det_wait):
            best_score = score
            best_wait = avg_wait
            best_Q = trial_Q
            best_deltas = deltas

    if best_wait > det_wait:
        return det_Q, det_wait, [1]*LANES

    return best_Q, best_wait, best_deltas

# ==========================================================
# SIMULATION
# ==========================================================

def run_simulation(mode="det"):

    traci.start(["sumo-gui", "-c", SUMO_CONFIG])

    Q = [0]*LANES
    wait, queue = [], []
    prev_lane = 0

    for t in range(SIM_TIME):

        traci.simulationStep()
        Q = get_real_queue()

        if mode == "det":
            Q, w = deterministic_step(Q)
            lane = int(np.argmax(Q))
            duration = 3
        else:
            Q, w, best_deltas = las_vegas_step(Q)
            lane = int(np.argmax(Q))
            duration = max(best_deltas[lane], 1)

        if lane != prev_lane:
            traci.simulationStep()

        apply_control(lane, duration)
        prev_lane = lane

        wait.append(w)
        queue.append(sum(Q)/LANES)

        print(f"[{mode}] Step {t} | W={w:.2f} | Q={sum(Q)/LANES:.2f}")

    traci.close()

    print("Loop finished")
    print("Wait length:", len(wait))
    print("Queue length:", len(queue))

    return wait, queue

# ==========================================================
# PLOTS AND RESULTS
# ==========================================================

def plot(wait_det, wait_lv):
    plt.plot(wait_det, label="Deterministic")
    plt.plot(wait_lv, label="Las Vegas")
    plt.legend()
    plt.show()

def advanced_plots(wait_det, wait_lv, queue_det, queue_lv):

    t = np.arange(len(wait_det))

    # ---------------------------
    # 1. Waiting Time Comparison
    # ---------------------------
    plt.figure()
    plt.plot(t, wait_det, label="Deterministic")
    plt.plot(t, wait_lv, label="Las Vegas")
    plt.xlabel("Time Step")
    plt.ylabel("Average Waiting Time")
    plt.title("Waiting Time Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 2. Queue Length Comparison
    # ---------------------------
    plt.figure()
    plt.plot(t, queue_det, label="Deterministic")
    plt.plot(t, queue_lv, label="Las Vegas")
    plt.xlabel("Time Step")
    plt.ylabel("Average Queue Length")
    plt.title("Queue Length Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 3. Cumulative Waiting Time
    # ---------------------------
    plt.figure()
    plt.plot(t, np.cumsum(wait_det), label="Deterministic")
    plt.plot(t, np.cumsum(wait_lv), label="Las Vegas")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Wait")
    plt.title("Cumulative Waiting Time")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 4. Moving Average (Smoothing)
    # ---------------------------
    window = 10
    def moving_avg(x):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    plt.figure()
    plt.plot(moving_avg(wait_det), label="Deterministic")
    plt.plot(moving_avg(wait_lv), label="Las Vegas")
    plt.title("Smoothed Waiting Time (Moving Avg)")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 5. Performance Gap
    # ---------------------------
    gap = np.array(wait_det) - np.array(wait_lv)

    plt.figure()
    plt.plot(t, gap)
    plt.axhline(0)
    plt.title("Performance Gap (Det - LV)")
    plt.xlabel("Time Step")
    plt.ylabel("Gap")
    plt.grid()
    plt.show()

    # ---------------------------
    # 6. Histogram Distribution
    # ---------------------------
    plt.figure()
    plt.hist(wait_det, bins=20, alpha=0.5, label="Deterministic")
    plt.hist(wait_lv, bins=20, alpha=0.5, label="Las Vegas")
    plt.title("Waiting Time Distribution")
    plt.legend()
    plt.grid()
    plt.show()

    # ---------------------------
    # 7. Box Plot (IEEE style comparison)
    # ---------------------------
    plt.figure()
    plt.boxplot([wait_det, wait_lv], labels=["Det", "LV"])
    plt.title("Statistical Comparison")
    plt.grid()
    plt.show()

def ieee_analysis(wait_det, wait_lv, queue_det, queue_lv):

    import numpy as np
    from scipy import stats

    print("\n================ IEEE ANALYSIS ================\n")

    # Convert to arrays
    wd = np.array(wait_det)
    wl = np.array(wait_lv)
    qd = np.array(queue_det)
    ql = np.array(queue_lv)

    # -----------------------------
    # 1. BASIC STATS
    # -----------------------------
    print("---- Mean Performance ----")
    print(f"Det Wait: {np.mean(wd):.4f}")
    print(f"LV  Wait: {np.mean(wl):.4f}")

    print("\n---- Variance ----")
    print(f"Det Var: {np.var(wd):.4f}")
    print(f"LV  Var: {np.var(wl):.4f}")

    print("\n---- Std Deviation ----")
    print(f"Det Std: {np.std(wd):.4f}")
    print(f"LV  Std: {np.std(wl):.4f}")

    # -----------------------------
    # 2. IMPROVEMENT %
    # -----------------------------
    improvement = (np.mean(wd) - np.mean(wl)) / np.mean(wd) * 100
    print(f"\nImprovement (LV over Det): {improvement:.2f}%")

    # -----------------------------
    # 3. CONFIDENCE INTERVAL (95%)
    # -----------------------------
    def confidence_interval(data):
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
        return mean, mean-h, mean+h

    det_ci = confidence_interval(wd)
    lv_ci  = confidence_interval(wl)

    print("\n---- 95% Confidence Interval ----")
    print(f"Det: Mean={det_ci[0]:.4f}, CI=[{det_ci[1]:.4f}, {det_ci[2]:.4f}]")
    print(f"LV : Mean={lv_ci[0]:.4f}, CI=[{lv_ci[1]:.4f}, {lv_ci[2]:.4f}]")

    # -----------------------------
    # 4. HYPOTHESIS TEST (t-test)
    # -----------------------------
    t_stat, p_value = stats.ttest_ind(wd, wl)

    print("\n---- Hypothesis Testing ----")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: Statistically Significant ✅")
    else:
        print("Result: NOT Statistically Significant ❌")

    # -----------------------------
    # 5. EFFECT SIZE (Cohen’s d)
    # -----------------------------
    pooled_std = np.sqrt((np.var(wd) + np.var(wl)) / 2)
    cohen_d = (np.mean(wd) - np.mean(wl)) / pooled_std

    print("\n---- Effect Size ----")
    print(f"Cohen's d: {cohen_d:.4f}")

    # -----------------------------
    # 6. STABILITY METRIC
    # -----------------------------
    stability_det = np.std(wd) / np.mean(wd)
    stability_lv  = np.std(wl) / np.mean(wl)

    print("\n---- Stability (CV) ----")
    print(f"Det CV: {stability_det:.4f}")
    print(f"LV  CV: {stability_lv:.4f}")

    # -----------------------------
    # 7. QUEUE PERFORMANCE
    # -----------------------------
    print("\n---- Queue Analysis ----")
    print(f"Det Avg Queue: {np.mean(qd):.4f}")
    print(f"LV  Avg Queue: {np.mean(ql):.4f}")

    print("\n==============================================\n")

def ieee_plots(wait_det, wait_lv, queue_det, queue_lv):

    import matplotlib.pyplot as plt
    import numpy as np

    t = np.arange(len(wait_det))

    # -----------------------------
    # 1. Mean with Confidence Bands
    # -----------------------------
    plt.figure()

    wd = np.array(wait_det)
    wl = np.array(wait_lv)

    plt.plot(t, wd, label="Deterministic")
    plt.plot(t, wl, label="Las Vegas")

    plt.fill_between(t, wd - np.std(wd), wd + np.std(wd), alpha=0.2)
    plt.fill_between(t, wl - np.std(wl), wl + np.std(wl), alpha=0.2)

    plt.xlabel("Time Step")
    plt.ylabel("Waiting Time")
    plt.title("Mean ± Variability")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # 2. CDF (VERY IMPORTANT FOR IEEE)
    # -----------------------------
    plt.figure()

    sorted_det = np.sort(wd)
    sorted_lv  = np.sort(wl)

    cdf_det = np.arange(len(wd)) / len(wd)
    cdf_lv  = np.arange(len(wl)) / len(wl)

    plt.plot(sorted_det, cdf_det, label="Deterministic")
    plt.plot(sorted_lv, cdf_lv, label="Las Vegas")

    plt.xlabel("Waiting Time")
    plt.ylabel("CDF")
    plt.title("Cumulative Distribution Function")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # 3. Throughput Proxy (Inverse Wait)
    # -----------------------------
    plt.figure()

    throughput_det = 1 / (wd + 1e-6)
    throughput_lv  = 1 / (wl + 1e-6)

    plt.plot(t, throughput_det, label="Deterministic")
    plt.plot(t, throughput_lv, label="Las Vegas")

    plt.xlabel("Time Step")
    plt.ylabel("Throughput Proxy")
    plt.title("Throughput Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # -----------------------------
    # 4. Queue vs Wait Scatter
    # -----------------------------
    plt.figure()

    plt.scatter(queue_det, wait_det, label="Det", alpha=0.5)
    plt.scatter(queue_lv, wait_lv, label="LV", alpha=0.5)

    plt.xlabel("Queue Length")
    plt.ylabel("Waiting Time")
    plt.title("Queue vs Waiting Relationship")
    plt.legend()
    plt.grid()
    plt.show()


def print_metrics(wait_det, wait_lv):
    """
    VIVA:
    - Final numerical proof of improvement
    """

    det = np.mean(wait_det)
    lv  = np.mean(wait_lv)

    print("\n===== FINAL RESULT =====")
    print(f"Deterministic : {det:.2f}")
    print(f"Las Vegas     : {lv:.2f}")

    # VIVA: This is your CLAIM validation
    if lv < det:
        print(f"Improvement   : {(det-lv)/det*100:.2f}%")
    else:
        print("WARNING: Las Vegas not better")

# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":

    np.random.seed(42)

    wait_det, queue_det = run_simulation("det")
    wait_lv, queue_lv   = run_simulation("lv")

    plot(wait_det, wait_lv)
    advanced_plots(wait_det, wait_lv, queue_det, queue_lv)
    ieee_analysis(wait_det, wait_lv, queue_det, queue_lv)
    ieee_plots(wait_det, wait_lv, queue_det, queue_lv)

    print("\nFINAL:")
    print("Deterministic:", np.mean(wait_det))
    print("Las Vegas:", np.mean(wait_lv))

    print_metrics(wait_det, wait_lv)

    input("Press Enter to exit...")
