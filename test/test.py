import torch
import numpy as np
from collections import deque
from datetime import datetime

from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from envs.myenv import CustomT1DSimEnv, PatientAction
from agents.TD3DDPGagent import TD3Agent, build_state, glucose_reward, get_iob

# ── Config ────────────────────────────────────────────────────────────────────

LOAD_PATH    = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation\agents\checkpoints\checkpoint_500"
RESULTS_PATH = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation\test_results.txt"
START_TIME   = datetime(2018, 1, 1, 6, 0, 0)
MAX_STEPS    = 480
SEQ_LEN      = 12
STATE_DIM    = 6
MAX_BASAL    = 0.75
MAX_BOLUS    = 3.0

meal_scenario = [(1, 45), (6, 70), (10, 20), (12, 80)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load agent ────────────────────────────────────────────────────────────────

agent = TD3Agent(state_dim=STATE_DIM)
agent.load(LOAD_PATH)
agent.actor.eval()

# ── Environment ───────────────────────────────────────────────────────────────

patient  = T1DPatient.withName("adult#001")
sensor   = CGMSensor.withName("Dexcom")
pump     = InsulinPump.withName("Insulet")
scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
env      = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

obs = env.reset()

# ── Episode ───────────────────────────────────────────────────────────────────

cgm_history  = deque(maxlen=13)
meal_history = deque(maxlen=72)
state_buffer = deque(maxlen=SEQ_LEN)

for _ in range(SEQ_LEN):
    state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

prev_basal   = 0.0
total_reward = 0.0
step         = 0
done         = False

cgm_log    = []
basal_log  = []
bolus_log  = []
reward_log = []
meal_log   = []

print(f"\n{'Step':>5}  {'CGM':>7}  {'Delta':>6}  {'Basal':>6}  {'Bolus':>6}  {'Meal':>6}  {'Reward':>8}")
print("-" * 60)

while step < MAX_STEPS and not done:

    current_cgm = float(obs.observation.CGM)
    meal        = float(obs.info['meal'])
    cgm_history.append(current_cgm)
    meal_history.append(meal)

    if len(cgm_history) < 3:
        obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
        step += 1
        continue

    cgm_list     = list(cgm_history)
    iob          = get_iob(env.patient)

    state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                        iob, prev_basal, meal_history)
    state_buffer.append(state)
    state_seq = np.array(state_buffer, dtype=np.float32)

    basal, bolus = agent.select_action(state_seq)

    prev_basal = basal

    obs      = env.step(PatientAction(basal, bolus), cho=0.0)
    done     = obs.done
    next_cgm = float(obs.observation.CGM)
    next_iob = get_iob(env.patient)
    delta    = next_cgm - current_cgm

    reward = -100.0 if (done and next_cgm < 40.0) else \
        glucose_reward(next_cgm, prev_cgm=current_cgm, basal=basal,
                       bolus=bolus, meal=meal, step=step)
    total_reward += reward

    cgm_log.append(next_cgm)
    basal_log.append(basal)
    bolus_log.append(bolus)
    reward_log.append(reward)
    meal_log.append(meal)

    print(f"{step:>5}  {current_cgm:>7.1f}  {delta:>+6.1f}  {basal:>6.3f}  "
          f"{bolus:>6.3f}  {meal:>6.1f}  {reward:>8.2f}")

    step += 1

# ── Summary ───────────────────────────────────────────────────────────────────

cgm_arr        = np.array(cgm_log)
time_in_range  = np.mean((cgm_arr >= 70) & (cgm_arr <= 180)) * 100
time_hypo      = np.mean(cgm_arr < 70) * 100
time_severe    = np.mean(cgm_arr < 54) * 100
time_hyper     = np.mean(cgm_arr > 180) * 100
time_severe_h  = np.mean(cgm_arr > 250) * 100

summary = f"""
{'=' * 50}
TEST SUMMARY
{'=' * 50}
  Steps completed    : {step}
  Total reward       : {total_reward:.2f}
  Mean CGM           : {cgm_arr.mean():.1f} mg/dL
  CGM std            : {cgm_arr.std():.1f} mg/dL
  Min CGM            : {cgm_arr.min():.1f} mg/dL
  Max CGM            : {cgm_arr.max():.1f} mg/dL
  Time in range      : {time_in_range:.1f}%  (70-180 mg/dL)
  Time hypo          : {time_hypo:.1f}%  (< 70 mg/dL)
  Time severe hypo   : {time_severe:.1f}%  (< 54 mg/dL)
  Time hyper         : {time_hyper:.1f}%  (> 180 mg/dL)
  Time severe hyper  : {time_severe_h:.1f}%  (> 250 mg/dL)
  Mean basal         : {np.mean(basal_log):.3f} U/hr
  Total basal        : {sum(basal_log):.3f} U
  Mean bolus         : {np.mean(bolus_log):.3f} U
  Total bolus        : {sum(bolus_log):.3f} U
  Total CHO          : {sum(meal_log):.1f} g
{'=' * 50}
"""

print(summary)

with open(RESULTS_PATH, "w") as f:
    f.write(summary)
print(f"Results saved to {RESULTS_PATH}")