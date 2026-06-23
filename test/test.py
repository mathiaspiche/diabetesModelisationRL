import os, numpy as np
from collections import deque
from datetime import datetime
from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from envs.myenv import CustomT1DSimEnv, PatientAction
from agents.TD3DDPGagent import TD3Agent, build_state, get_iob
from utils.config_locale import BASE

START_TIME  = datetime(2018, 1, 1, 6, 0, 0)
MAX_STEPS, SEQ_LEN, STATE_DIM = 480, 12, 6
SEEDS       = range(10)
EVAL_SCENARIOS = [
    [(10, 10)],
    [(1.5, 50)],
    [(8, 20)],
    [(1.5, 55), (7.5, 75), (13.5, 65)],
    [(3, 15), (7, 28), (13, 32)],
    [(2, 45), (5.5, 25), (11, 85), (16, 40)],
    [(1, 95)],
    [(2, 70), (3, 60)],
    [(8, 50), (14, 110)],
    [(6, 120)],
    [(1.5, 60), (4, 40), (7, 80), (11, 55), (15, 70), (19, 50)],
]

def print_episode(agent, scenario_meals, seed=0):
    patient  = T1DPatient.withName("adult#001")
    sensor   = CGMSensor.withName("Dexcom", seed=seed)
    pump     = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=scenario_meals)
    env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
    obs = env.reset()

    cgm_history  = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque((np.zeros(STATE_DIM, dtype=np.float32) for _ in range(SEQ_LEN)),
                         maxlen=SEQ_LEN)
    prev_basal, step, done = 0.0, 0, False

    agent.actor.eval()
    print(f"\n{'step':>4} {'CGM':>7} {'delta':>6} {'basal':>7} {'bolus':>7} {'IOB':>6} {'meal':>6}")
    print("-" * 52)
    while step < MAX_STEPS and not done:
        current_cgm = float(obs.observation.CGM)
        meal = float(obs.info['meal'])
        cgm_history.append(current_cgm)
        meal_history.append(meal)

        if len(cgm_history) < 3:
            obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
            step += 1
            continue

        cgm_list = list(cgm_history)
        iob = get_iob(env.patient)                      # IOB the agent acted on
        state = build_state(current_cgm, cgm_list[-2], cgm_list[-3], iob, prev_basal, meal_history)
        state_buffer.append(state)
        basal, bolus = agent.select_action(np.array(state_buffer, dtype=np.float32))
        prev_basal = basal

        obs = env.step(PatientAction(basal, bolus), cho=0.0)
        next_cgm = float(obs.observation.CGM)
        done = bool(obs.done) or next_cgm <= 39.0
        delta = next_cgm - current_cgm

        print(f"{step:>4} {next_cgm:>7.1f} {delta:>+6.1f} {basal:>7.3f} {bolus:>7.3f} {iob:>6.2f} {meal:>6.1f}")
        step += 1
def run_eval(agent, scenario_meals, seed):
    patient  = T1DPatient.withName("adult#001")
    sensor   = CGMSensor.withName("Dexcom", seed=seed)
    pump     = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=scenario_meals)
    env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)
    obs = env.reset()

    cgm_history  = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque((np.zeros(STATE_DIM, dtype=np.float32) for _ in range(SEQ_LEN)),
                         maxlen=SEQ_LEN)
    prev_basal, step, done = 0.0, 0, False
    cgm_log = []

    agent.actor.eval()
    while step < MAX_STEPS and not done:
        current_cgm = float(obs.observation.CGM)
        meal = float(obs.info['meal'])
        cgm_history.append(current_cgm)
        meal_history.append(meal)

        if len(cgm_history) < 3:
            obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
            step += 1
            continue

        cgm_list = list(cgm_history)
        iob = get_iob(env.patient)
        state = build_state(current_cgm, cgm_list[-2], cgm_list[-3], iob, prev_basal, meal_history)
        state_buffer.append(state)
        basal, bolus = agent.select_action(np.array(state_buffer, dtype=np.float32))
        prev_basal = basal

        obs = env.step(PatientAction(basal, bolus), cho=0.0)
        next_cgm = float(obs.observation.CGM)
        done = bool(obs.done) or next_cgm <= 39.0
        cgm_log.append(next_cgm)
        step += 1

    cgm = np.array(cgm_log) if cgm_log else np.array([400.0])
    return {
        "tir": float(np.mean((cgm >= 72) & (cgm <= 200)) * 100),
        "hypo": float(np.mean(cgm < 72) * 100),
        "severe": float(np.mean(cgm < 54) * 100),
        "hyper": float(np.mean(cgm > 200) * 100),
        "sev_hyper": float(np.mean(cgm > 275) * 100),
        "mean": float(cgm.mean()),
    }

best = None
for p in [4000]:
    path = os.path.join(BASE, f"checkpoint_{p}")

    if not os.path.exists(path):
        print(f"skip {p}: not found"); continue

    agent = TD3Agent(state_dim=STATE_DIM)
    agent.load(path)
    per_scen = []
    for sc in EVAL_SCENARIOS:
        runs = [run_eval(agent, sc, s) for s in SEEDS]
        per_scen.append({k: float(np.mean([r[k] for r in runs])) for k in runs[0]})

    mean_tir = float(np.mean([d["tir"] for d in per_scen]))
    worst_tir = float(np.min([d["tir"] for d in per_scen]))
    mean_hypo = float(np.mean([d["hypo"] for d in per_scen]))
    mean_sev = float(np.mean([d["severe"] for d in per_scen]))
    mean_hyper = float(np.mean([d["hyper"] for d in per_scen]))
    mean_shyp = float(np.mean([d["sev_hyper"] for d in per_scen]))

    print(f"checkpoint {p:>5}: mean TIR={mean_tir:5.1f}%  worst-scenario_time_in_range={worst_tir:5.1f}%  "
          f"mean_hypo={mean_hypo:4.1f}%  mean_severe_hypo={mean_sev:4.1f}%  "
          f"mean_hyper={mean_hyper:4.1f}%  mean_sev-hyper={mean_shyp:4.1f}%")

    summary = {"p": p, "tir": mean_tir, "worst": worst_tir,
               "hypo": mean_hypo, "severe": mean_sev,
               "hyper": mean_hyper, "sev_hyper": mean_shyp}
    if best is None or summary["tir"] > best["tir"]:
        best = summary
if best:
    pp, mt, wt, hy, sv = best
    print(f"\nBest by mean time in range: checkpoint_{p}  mean={mt:.1f}%  worst-scenario={wt:.1f}%  "
          f"mean_hypo={hy:.1f}%  mean_severe={sv:.1f}%")
