"""
Adaptive Traffic Signal Control — Real-Time SUMO
================================================================================
Based on: Ali et al., IEEE Access, 2021. DOI: 10.1109/ACCESS.2021.3094270

FIX: Single SUMO session — phases are read inside the runner after SUMO starts,
     removing the broken double-start that caused immediate TraCI termination.

Usage
-----
  # Run directly in VS Code (uses hardcoded Args below):
  python adaptive_traffic_realtime.py

  # Or via CLI:
  python adaptive_traffic_realtime.py --sumo-cfg intersection.sumocfg --tl-id J0
  python adaptive_traffic_realtime.py --sumo-cfg intersection.sumocfg --tl-id J0 --gui
  python adaptive_traffic_realtime.py --sumo-cfg intersection.sumocfg --tl-id J0 --formula modified
"""

import sys
import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    import traci
    import traci.constants as tc
except ImportError:
    print("[ERROR] traci not found. Install SUMO and run:  pip install traci")
    sys.exit(1)

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    print("[WARNING] scikit-fuzzy not found.  pip install scikit-fuzzy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("AdaptiveTC")


# =========================================================================== #
# 1.  HARDCODED ARGS — used when running directly in VS Code
# =========================================================================== #

class Args:
    sumo_cfg    = "SIM.sumocfg"
    tl_id       = "clusterJ23_J26_clusterJ18_J19_J20_cluster_11693513535_1783433672_1783433674_1783433694_#1more"
    formula     = "webster"
    gui         = True
    max_steps   = 7200
    min_green   = 10.0
    step_length = 1.0
    save_plots  = None


# =========================================================================== #
# 2.  DATA STRUCTURES
# =========================================================================== #

class FormulaType(Enum):
    WEBSTER          = "Webster"
    MODIFIED_WEBSTER = "Modified Webster"


@dataclass
class Phase:
    phase_id:        int
    sumo_phase_idx:  int
    lanes:           List[str]
    detector_ids:    List[str]
    min_green:       float = 10.0
    green_time:      float = 10.0
    extension:       float = 0.0


@dataclass
class IntersectionConfig:
    n_phases:                int   = 4
    saturation_flow:         float = 1800.0
    lost_time_per_phase:     float = 3.0
    min_cycle:               float = 40.0
    max_cycle:               float = 120.0
    min_green:               float = 10.0
    max_green_factor:        float = 1.30
    fuzzy_trigger_remaining: float = 15.0

    @property
    def total_lost_time(self) -> float:
        return self.n_phases * self.lost_time_per_phase


@dataclass
class CycleMetrics:
    cycle_num:         int
    optimal_cycle:     float
    formula:           str
    phase_green_times: List[float]
    avg_waiting_time:  float = 0.0
    avg_travel_time:   float = 0.0
    avg_speed:         float = 0.0


@dataclass
class StepPerformance:
    wait_samples:  List[float] = field(default_factory=list)
    tt_samples:    List[float] = field(default_factory=list)
    speed_samples: List[float] = field(default_factory=list)

    def record(self, wait: float, tt: float, speed_ms: float):
        self.wait_samples.append(wait)
        self.tt_samples.append(tt)
        self.speed_samples.append(speed_ms * 3.6)

    def averages(self) -> Tuple[float, float, float]:
        def _avg(lst): return float(np.mean(lst)) if lst else 0.0
        return _avg(self.wait_samples), _avg(self.tt_samples), _avg(self.speed_samples)


# =========================================================================== #
# 3.  WEBSTER FORMULAS
# =========================================================================== #

def critical_lane_flow_ratio(eps_s, eps_l, eps_sr, sat):
    return max(eps_s, eps_l, eps_sr) / sat if sat > 0 else 0.0


def webster_cycle(y_sum: float, L: float) -> float:
    """Eq. 3: Co = (1.5L + 5) / (1 - sum_yi)"""
    denom = 1.0 - y_sum
    return (1.5 * L + 5.0) / denom if denom > 1e-6 else float("inf")


def modified_webster_cycle(y_sum: float, L: float) -> float:
    """Eq. 4: Co = (1.978L + 5.109) / (1 - 0.9013 * sum_yi)"""
    denom = 1.0 - 0.9013 * y_sum
    return (1.978 * L + 5.109) / denom if denom > 1e-6 else float("inf")


def compute_optimal_cycle(y_sum, L, formula, min_c, max_c):
    Co_raw = (webster_cycle(y_sum, L) if formula == FormulaType.WEBSTER
              else modified_webster_cycle(y_sum, L))
    return float(np.clip(Co_raw, min_c, max_c))


def distribute_green_times(yi_list, Co, L, min_green):
    m     = len(yi_list)
    Ce    = max(Co - L - m * min_green, 0.0)
    y_sum = sum(yi_list)
    greens = []
    for yi in yi_list:
        alpha = yi / y_sum if y_sum > 1e-9 else 1.0 / m
        greens.append(round(alpha * Ce + min_green, 2))
    return greens


def flow_to_hourly(n_veh, cycle_time):
    return n_veh * 3600.0 / cycle_time if cycle_time > 0 else 0.0


# =========================================================================== #
# 4.  FUZZY LOGIC SYSTEM
# =========================================================================== #

class FuzzyGreenAdjuster:
    def __init__(self):
        if not SKFUZZY_AVAILABLE:
            log.warning("scikit-fuzzy unavailable - fuzzy adjuster will return 0.")
            self._ready = False
            return
        self._ready = True
        self._build()

    def _build(self):
        rql  = ctrl.Antecedent(np.arange(0,  30.5, 0.5),  "RQL")
        pr   = ctrl.Antecedent(np.arange(0,   4.1, 0.05), "PR")
        rt   = ctrl.Antecedent(np.arange(0,  15.5, 0.5),  "RT_CGP")
        eors = ctrl.Consequent(np.arange(-3,  3.1,  0.1), "EorS")

        rql["zero"]   = fuzz.trimf(rql.universe,  [ 0,  0,  2])
        rql["short"]  = fuzz.trimf(rql.universe,  [ 0,  4,  8])
        rql["medium"] = fuzz.trimf(rql.universe,  [ 4, 11, 15])
        rql["long"]   = fuzz.trimf(rql.universe,  [11, 23, 30])

        pr["zero"]    = fuzz.trimf(pr.universe,   [0,  0,  1])
        pr["low"]     = fuzz.trimf(pr.universe,   [0,  1,  2])
        pr["medium"]  = fuzz.trimf(pr.universe,   [1,  2,  3])
        pr["high"]    = fuzz.trimf(pr.universe,   [2,  4,  4])

        rt["short"]   = fuzz.trimf(rt.universe,   [0,  0,  5])
        rt["medium"]  = fuzz.trimf(rt.universe,   [0,  5, 10])
        rt["long"]    = fuzz.trimf(rt.universe,   [5, 15, 15])

        eors["NM"]    = fuzz.trimf(eors.universe, [-3, -3, -2])
        eors["NS"]    = fuzz.trimf(eors.universe, [-3, -2, -1])
        eors["zero"]  = fuzz.trimf(eors.universe, [-1,  0,  1])
        eors["PS"]    = fuzz.trimf(eors.universe, [ 0,  1,  2])
        eors["PM"]    = fuzz.trimf(eors.universe, [ 1,  2,  3])

        rules = []
        for rt_mf in ["short", "medium", "long"]:
            for pr_mf in ["zero", "low", "medium", "high"]:
                rules.append(ctrl.Rule(rql["zero"] & rt[rt_mf] & pr[pr_mf], eors["NM"]))

        for pr_mf in ["zero", "low", "medium", "high"]:
            rules.append(ctrl.Rule(rql["short"] & rt["short"]  & pr[pr_mf], eors["NM"]))
        rules.append(ctrl.Rule(rql["short"] & rt["medium"] & pr["zero"],    eors["NS"]))
        rules.append(ctrl.Rule(rql["short"] & rt["medium"] & pr["low"],     eors["NS"]))
        rules.append(ctrl.Rule(rql["short"] & rt["medium"] & pr["medium"],  eors["zero"]))
        rules.append(ctrl.Rule(rql["short"] & rt["medium"] & pr["high"],    eors["zero"]))
        rules.append(ctrl.Rule(rql["short"] & rt["long"]   & pr["zero"],    eors["zero"]))
        rules.append(ctrl.Rule(rql["short"] & rt["long"]   & pr["low"],     eors["zero"]))
        rules.append(ctrl.Rule(rql["short"] & rt["long"]   & pr["medium"],  eors["PS"]))
        rules.append(ctrl.Rule(rql["short"] & rt["long"]   & pr["high"],    eors["PS"]))

        rules.append(ctrl.Rule(rql["medium"] & rt["short"]  & pr["zero"],   eors["NM"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["short"]  & pr["low"],    eors["NS"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["short"]  & pr["medium"], eors["zero"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["short"]  & pr["high"],   eors["zero"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["medium"] & pr["zero"],   eors["zero"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["medium"] & pr["low"],    eors["zero"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["medium"] & pr["medium"], eors["PS"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["medium"] & pr["high"],   eors["PS"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["long"]   & pr["zero"],   eors["PS"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["long"]   & pr["low"],    eors["PS"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["long"]   & pr["medium"], eors["PM"]))
        rules.append(ctrl.Rule(rql["medium"] & rt["long"]   & pr["high"],   eors["PM"]))

        rules.append(ctrl.Rule(rql["long"] & rt["short"]  & pr["zero"],     eors["zero"]))
        rules.append(ctrl.Rule(rql["long"] & rt["short"]  & pr["low"],      eors["PS"]))
        rules.append(ctrl.Rule(rql["long"] & rt["short"]  & pr["medium"],   eors["PS"]))
        rules.append(ctrl.Rule(rql["long"] & rt["short"]  & pr["high"],     eors["PM"]))
        rules.append(ctrl.Rule(rql["long"] & rt["medium"] & pr["zero"],     eors["PS"]))
        rules.append(ctrl.Rule(rql["long"] & rt["medium"] & pr["low"],      eors["PS"]))
        rules.append(ctrl.Rule(rql["long"] & rt["medium"] & pr["medium"],   eors["PM"]))
        rules.append(ctrl.Rule(rql["long"] & rt["medium"] & pr["high"],     eors["PM"]))
        rules.append(ctrl.Rule(rql["long"] & rt["long"]   & pr["zero"],     eors["zero"]))
        rules.append(ctrl.Rule(rql["long"] & rt["long"]   & pr["low"],      eors["PS"]))
        rules.append(ctrl.Rule(rql["long"] & rt["long"]   & pr["medium"],   eors["PM"]))
        rules.append(ctrl.Rule(rql["long"] & rt["long"]   & pr["high"],     eors["PM"]))

        self._sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        log.info(f"[FUZZY] Built {len(rules)}-rule Mamdani system.")

    def compute(self, rql_val, pr_val, rt_val, trace=False):
        if not self._ready:
            return 0.0
        rql_val = float(np.clip(rql_val, 0.0, 30.0))
        pr_val  = float(np.clip(pr_val,  0.0,  4.0))
        rt_val  = float(np.clip(rt_val,  0.0, 15.0))
        try:
            self._sim.input["RQL"]    = rql_val
            self._sim.input["PR"]     = pr_val
            self._sim.input["RT_CGP"] = rt_val
            self._sim.compute()
            adj = float(self._sim.output["EorS"])
        except Exception as e:
            log.debug(f"Fuzzy inference error: {e}")
            adj = 0.0
        if trace:
            log.info(f"  [FUZZY] RQL={rql_val:.1f} PR={pr_val:.2f} "
                     f"RT={rt_val:.1f} -> EorS={adj:+.3f}s")
        return adj


# =========================================================================== #
# 5.  ADAPTIVE CONTROLLER
# =========================================================================== #

class AdaptiveTrafficController:
    def __init__(self, config: IntersectionConfig, phases: List[Phase],
                 formula: FormulaType = FormulaType.WEBSTER,
                 fuzzy_trace: bool = False):
        self.config     = config
        self.phases     = phases
        self.formula    = formula
        self.fuzzy      = FuzzyGreenAdjuster()
        self._trace     = fuzzy_trace
        self._phase_idx = 0
        self._phase_timer = float(phases[0].green_time)
        self._cycle_num   = 0
        self._metrics: List[CycleMetrics] = []
        self._perf        = StepPerformance()
        self._reset_counts()
        log.info(f"[CONTROLLER] {formula.value} controller ready ({len(phases)} phases).")

    def step(self, sensor_data: Dict) -> Dict:
        pid    = self.current_phase.phase_id
        result = {"phase": pid, "remaining": self._phase_timer,
                  "fuzzy_adj": 0.0, "cycle_complete": False}

        self._counts[pid]["s"]  += sensor_data.get(f"count_s_{pid}",  0)
        self._counts[pid]["l"]  += sensor_data.get(f"count_l_{pid}",  0)
        self._counts[pid]["sr"] += sensor_data.get(f"count_sr_{pid}", 0)

        for key, val in sensor_data.items():
            if key.startswith("_wait_"):
                self._perf.wait_samples.append(float(val))
            elif key.startswith("_tt_"):
                self._perf.tt_samples.append(float(val))
            elif key.startswith("_speed_"):
                self._perf.speed_samples.append(float(val) * 3.6)

        if self._phase_timer < self.config.fuzzy_trigger_remaining:
            rql = float(sensor_data.get(f"queue_{pid}",   0))
            pr  = float(sensor_data.get(f"passing_{pid}", 0))
            adj = self.fuzzy.compute(rql, pr, self._phase_timer, trace=self._trace)
            self._phase_timer            += adj
            self.current_phase.extension += adj
            max_ext = self.current_phase.green_time * self.config.max_green_factor
            if self._phase_timer > max_ext:
                self._phase_timer = max_ext
            result["fuzzy_adj"] = adj

        self._phase_timer -= 1.0

        if self._phase_timer <= 0:
            cycle_done = self._advance_phase()
            result["cycle_complete"] = cycle_done
            if cycle_done:
                self._end_of_cycle()

        result["remaining"] = self._phase_timer
        return result

    @property
    def current_phase(self) -> Phase:
        return self.phases[self._phase_idx]

    @property
    def metrics(self) -> List[CycleMetrics]:
        return self._metrics

    def _reset_counts(self):
        self._counts = {p.phase_id: {"s": 0, "l": 0, "sr": 0} for p in self.phases}

    def _advance_phase(self) -> bool:
        self.current_phase.extension = 0.0
        next_idx        = (self._phase_idx + 1) % len(self.phases)
        cycle_done      = next_idx == 0
        self._phase_idx = next_idx
        self._phase_timer = float(self.current_phase.green_time)
        return cycle_done

    def _end_of_cycle(self):
        self._cycle_num += 1
        Co_prev = (sum(p.green_time for p in self.phases) + self.config.total_lost_time)

        yi_list = []
        for ph in self.phases:
            pid    = ph.phase_id
            eps_s  = flow_to_hourly(self._counts[pid]["s"],  Co_prev)
            eps_l  = flow_to_hourly(self._counts[pid]["l"],  Co_prev)
            eps_sr = flow_to_hourly(self._counts[pid]["sr"], Co_prev)
            yi     = critical_lane_flow_ratio(eps_s, eps_l, eps_sr, self.config.saturation_flow)
            yi_list.append(max(yi, 1e-6))

        y_sum  = sum(yi_list)
        L      = self.config.total_lost_time
        Co     = compute_optimal_cycle(y_sum, L, self.formula,
                                       self.config.min_cycle, self.config.max_cycle)
        greens = distribute_green_times(yi_list, Co, L, self.config.min_green)

        log.info(
            f"[CYCLE {self._cycle_num:03d}] {self.formula.value} "
            f"Co={Co:.1f}s  y_sum={y_sum:.4f}  "
            f"Greens={[f'{g:.1f}' for g in greens]}"
        )

        for i, ph in enumerate(self.phases):
            ph.green_time = greens[i]

        avg_w, avg_tt, avg_sp = self._perf.averages()
        self._metrics.append(CycleMetrics(
            self._cycle_num, Co, self.formula.value, greens,
            avg_w, avg_tt, avg_sp))

        self._reset_counts()
        self._perf = StepPerformance()


# =========================================================================== #
# 6.  SUMO RUNNER  — single session, phases built after SUMO starts
# =========================================================================== #

class SUMORunner:
    """
    KEY FIX: Phases are built INSIDE run() after SUMO starts — no double-start.
    The controller is initialised here too, so everything happens in one session.
    """

    HALT_SPEED = 0.1

    def __init__(self,
                 sumo_cfg:     str,
                 tl_id:        str,
                 formula:      FormulaType,
                 detector_map: Optional[Dict[int, List[str]]] = None,
                 use_gui:      bool  = False,
                 max_steps:    int   = 7200,
                 step_length:  float = 1.0,
                 min_green:    float = 10.0,
                 save_plots:   Optional[str] = None):
        self.sumo_cfg    = sumo_cfg
        self.tl_id       = tl_id
        self.formula     = formula
        self.det_map     = detector_map or {}
        self.use_gui     = use_gui
        self.max_steps   = max_steps
        self.step_length = step_length
        self.min_green   = min_green
        self.save_plots  = save_plots
        self._phase_lane_map: Dict[int, List[str]] = {}
        self.ctl: Optional[AdaptiveTrafficController] = None

    def run(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary, "-c", os.path.abspath(self.sumo_cfg),
            "--no-step-log",
            "--waiting-time-memory", "10000",
            "--step-length", str(self.step_length),
        ]
        log.info(f"Starting SUMO: {' '.join(cmd)}")
        traci.start(cmd)

        try:
            # Build phases from the live TL programme — no second start needed
            log.info(f"[INIT] Reading TL programme for '{self.tl_id}'...")
            phases = self._build_phases_from_sumo()

            config = IntersectionConfig(
                n_phases  = len(phases),
                min_green = self.min_green,
            )
            self.ctl = AdaptiveTrafficController(config, phases, self.formula)
            self._phase_lane_map = self._build_phase_lane_map(phases)

            log.info(f"[INIT] Ready. {len(phases)} phases, formula={self.formula.value}")

            # Warmup: advance steps until vehicles start appearing (max 300 steps)
            log.info("[INIT] Warming up - waiting for vehicles to load...")
            for _w in range(300):
                traci.simulationStep()
                if len(traci.vehicle.getIDList()) > 0:
                    log.info(f"[INIT] First vehicles appeared after {_w+1} warmup steps.")
                    break
            else:
                log.warning("[INIT] No vehicles after 300 warmup steps - check your trips file.")

            step = 0
            while (step < self.max_steps and
                   traci.simulation.getMinExpectedNumber() > 0):

                traci.simulationStep()

                # Set TL phase in SUMO to match controller
                try:
                    traci.trafficlight.setPhase(
                        self.tl_id,
                        self.ctl.current_phase.sumo_phase_idx)
                except traci.TraCIException as e:
                    log.warning(f"setPhase error: {e}")

                sensor = self._read_sensors()
                result = self.ctl.step(sensor)

                if result["cycle_complete"] and self.ctl.metrics:
                    m = self.ctl.metrics[-1]
                    log.info(
                        f"  CYCLE {m.cycle_num:03d}  Co={m.optimal_cycle:.1f}s  "
                        f"wait={m.avg_waiting_time:.1f}s  "
                        f"speed={m.avg_speed:.1f}km/h  "
                        f"greens={[f'{g:.1f}' for g in m.phase_green_times]}")
                step += 1

            log.info(f"[DONE] Simulation finished after {step} steps.")

        finally:
            traci.close()
            log.info("SUMO closed.")

    # ------------------------------------------------------------------ #

    def _build_phases_from_sumo(self) -> List[Phase]:
        programme  = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
        ctrl_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        phases     = []
        ph_id      = 1
        for idx, sumo_phase in enumerate(programme.phases):
            state = sumo_phase.state
            if "G" not in state and "g" not in state:
                continue
            lanes = list(dict.fromkeys(
                ctrl_lanes[i] for i, s in enumerate(state)
                if s in ("G", "g") and i < len(ctrl_lanes)
            ))
            phases.append(Phase(
                phase_id       = ph_id,
                sumo_phase_idx = idx,
                lanes          = lanes,
                detector_ids   = [],
                min_green      = self.min_green,
                green_time     = max(sumo_phase.duration, self.min_green),
            ))
            ph_id += 1
        log.info(f"[PHASES] {len(phases)} green phases from TL '{self.tl_id}'.")
        return phases

    def _build_phase_lane_map(self, phases: List[Phase]) -> Dict[int, List[str]]:
        try:
            ctrl_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
            phase_lanes: Dict[int, List[str]] = {}
            for ph in phases:
                programme  = traci.trafficlight.getAllProgramLogics(self.tl_id)[0]
                sumo_phase = programme.phases[ph.sumo_phase_idx]
                state      = sumo_phase.state
                lanes      = []
                for link_idx, sig in enumerate(state):
                    if sig in ("G", "g") and link_idx < len(ctrl_lanes):
                        lane_id = ctrl_lanes[link_idx]
                        if lane_id not in lanes:
                            lanes.append(lane_id)
                phase_lanes[ph.phase_id] = lanes
            log.info(f"[TL] Phase-lane map built.")
            return phase_lanes
        except Exception as e:
            log.error(f"Failed to build phase-lane map: {e}")
            raise

    def _read_sensors(self) -> Dict:
        data: Dict = {}
        pid   = self.ctl.current_phase.phase_id
        lanes = self._phase_lane_map.get(pid, [])

        queue = 0
        for lane_id in lanes:
            try:
                queue += traci.lane.getLastStepHaltingNumber(lane_id)
            except traci.TraCIException:
                pass
        data[f"queue_{pid}"] = float(queue)

        detectors = self.det_map.get(pid, [])
        if detectors:
            crossed = 0
            for det_id in detectors:
                try:
                    crossed += traci.inductionloop.getLastStepVehicleNumber(det_id)
                except traci.TraCIException:
                    pass
            passing_rate = crossed / self.step_length
        else:
            moving = 0
            for lane_id in lanes:
                try:
                    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        if traci.vehicle.getSpeed(veh_id) > self.HALT_SPEED:
                            moving += 1
                except traci.TraCIException:
                    pass
            passing_rate = float(moving) / self.step_length
        data[f"passing_{pid}"] = float(passing_rate)

        s = l = r = 0
        for lane_id in lanes:
            try:
                veh_ids    = traci.lane.getLastStepVehicleIDs(lane_id)
                if not veh_ids:
                    continue
                links      = traci.lane.getLinks(lane_id)
                directions = set()
                for link in links:
                    if len(link) > 6:
                        directions.add(str(link[6]).lower())
                n = len(veh_ids)
                if "s" in directions and "l" not in directions and "r" not in directions:
                    s += n
                elif "l" in directions and "s" not in directions:
                    l += n
                elif "r" in directions and "s" not in directions:
                    r += n
                else:
                    s += int(n * 0.60)
                    l += int(n * 0.20)
                    r += n - int(n * 0.60) - int(n * 0.20)
            except traci.TraCIException:
                pass
        data[f"count_s_{pid}"]  = s
        data[f"count_l_{pid}"]  = l
        data[f"count_sr_{pid}"] = r

        for veh_id in traci.vehicle.getIDList():
            try:
                wait  = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                # Travel time approximated as time since vehicle departed
                depart = traci.vehicle.getDeparture(veh_id)
                sim_t  = traci.simulation.getTime()
                tt     = (sim_t - depart) if depart >= 0 else 0.0
                speed  = traci.vehicle.getSpeed(veh_id)
                data[f"_wait_{veh_id}"]  = wait
                data[f"_tt_{veh_id}"]    = tt
                data[f"_speed_{veh_id}"] = speed
            except traci.TraCIException:
                pass

        return data


# =========================================================================== #
# 7.  PLOTS
# =========================================================================== #

def plot_performance(metrics: List[CycleMetrics], formula_name: str,
                     save_path: Optional[str] = None):
    if not metrics:
        log.warning("No metrics to plot.")
        return
    cycles = [m.cycle_num        for m in metrics]
    co     = [m.optimal_cycle    for m in metrics]
    wait   = [m.avg_waiting_time for m in metrics]
    tt     = [m.avg_travel_time  for m in metrics]
    speed  = [m.avg_speed        for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Adaptive Traffic Signal Control - Real-Time SUMO\n"
        f"{formula_name}  (Ali et al., IEEE Access 2021)",
        fontsize=12, fontweight="bold")

    for ax, y, ylabel, title in [
        (axes[0, 0], co,    "Seconds", "Optimal Cycle Length Co(n)"),
        (axes[0, 1], wait,  "Seconds", "Avg Waiting Time per Vehicle"),
        (axes[1, 0], tt,    "Seconds", "Avg Travel Time per Vehicle"),
        (axes[1, 1], speed, "km/h",    "Avg Speed"),
    ]:
        ax.plot(cycles, y, "-o", color="#1f77b4", markersize=4,
                linewidth=2.0, alpha=0.85)
        ax.set_xlabel("Cycle #", fontsize=9)
        ax.set_ylabel(ylabel,    fontsize=9)
        ax.set_title(title,      fontsize=10)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save_path}")
    else:
        plt.show()


def plot_green_distribution(metrics: List[CycleMetrics],
                             save_path: Optional[str] = None):
    if not metrics:
        return
    data = np.array([m.phase_green_times for m in metrics])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=10, vmax=60)
    ax.set_xlabel("Cycle #")
    ax.set_ylabel("Phase")
    ax.set_yticks(range(data.shape[1]))
    ax.set_yticklabels([f"Ph {i+1}" for i in range(data.shape[1])])
    ax.set_title("Green-Time Allocation per Phase per Cycle", fontsize=11)
    fig.colorbar(im, ax=ax, label="Green Time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save_path}")
    else:
        plt.show()


# =========================================================================== #
# 8.  CLI
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive TSC - real-time SUMO data (Ali et al. 2021)")
    p.add_argument("--sumo-cfg",    required=True)
    p.add_argument("--tl-id",       required=True)
    p.add_argument("--formula",     choices=["webster", "modified"], default="webster")
    p.add_argument("--gui",         action="store_true")
    p.add_argument("--max-steps",   type=int,   default=7200)
    p.add_argument("--min-green",   type=float, default=10.0)
    p.add_argument("--step-length", type=float, default=1.0)
    p.add_argument("--save-plots",  default=None)
    return p.parse_args()


# =========================================================================== #
# 9.  MAIN
# =========================================================================== #

def main():
    # Use hardcoded Args when running directly in VS Code (no CLI args)
    if len(sys.argv) > 1:
        a = parse_args()
    else:
        a = Args()

    if not os.path.exists(a.sumo_cfg):
        log.error(f"SUMO config not found: {a.sumo_cfg}")
        sys.exit(1)

    formula = (FormulaType.MODIFIED_WEBSTER if a.formula == "modified"
               else FormulaType.WEBSTER)

    save_dir = a.save_plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    def _sp(name): return os.path.join(save_dir, name) if save_dir else None

    # Single runner — SUMO starts once, phases built inside, Webster runs
    runner = SUMORunner(
        sumo_cfg    = a.sumo_cfg,
        tl_id       = a.tl_id,
        formula     = formula,
        use_gui     = a.gui,
        max_steps   = a.max_steps,
        step_length = a.step_length,
        min_green   = a.min_green,
        save_plots  = save_dir,
    )
    runner.run()

    # Summary and plots
    ctl = runner.ctl
    if ctl and ctl.metrics:
        avg = lambda a: np.mean([getattr(m, a) for m in ctl.metrics])
        log.info("\n" + "=" * 60)
        log.info(f"  Formula           : {formula.value}")
        log.info(f"  Cycles completed  : {len(ctl.metrics)}")
        log.info(f"  Avg waiting time  : {avg('avg_waiting_time'):.2f} s")
        log.info(f"  Avg travel time   : {avg('avg_travel_time'):.2f} s")
        log.info(f"  Avg speed         : {avg('avg_speed'):.2f} km/h")
        log.info("=" * 60)

        plot_performance(ctl.metrics, formula.value, _sp("performance.png"))
        plot_green_distribution(ctl.metrics,          _sp("green_dist.png"))
    else:
        log.warning("No cycles completed — check your .sumocfg and tl_id in Args.")


if __name__ == "__main__":
    main()