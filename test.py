import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from datetime import datetime

from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from myenv import CustomT1DSimEnv, PatientAction
from reinforcementDIabetes import RecurrentTD3Agent
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOAD_PATH = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation/best"
RESULTS_PATH = "test_results.txt"

START_TIME = datetime(2018, 1, 1, 6, 0, 0)
MAX_STEPS = 480
SEQ_LEN = 12
STATE_DIM = 7
ACTION_DIM = 2

meal_scenario = [(1, 45), (6, 70), (10, 20), (12, 80)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Networks  (must match training definition exactly)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x) * self.max_action


# ---------------------------------------------------------------------------
# Helpers  (identical to training script)
# ---------------------------------------------------------------------------

def get_iob(patient) -> float:
    s1 = float(patient.state[10])
    s2 = float(patient.state[11])
    return (s1 + s2) / 2000.0


def build_state(cgm, prev_cgm, prev_prev_cgm, iob, prev_basal, prev_bolus, meal_history):
    cgm_norm = cgm / 400.0
    delta = cgm - prev_cgm
    delta2 = cgm - 2 * prev_cgm + prev_prev_cgm
    delta_norm = delta / 50.0
    delta2_norm = delta2 / 50.0
    iob_norm = min(iob, 3.0) / 3.0
    meal_sum = sum(list(meal_history)[-36:]) / 100.0
    basal_norm = prev_basal / 1.5
    bolus_norm = prev_bolus / 5.0
    return np.array(
        [cgm_norm, delta_norm, delta2_norm, iob_norm, meal_sum, basal_norm, bolus_norm],
        dtype=np.float32
    )


def glucose_reward(cgm, delta):
    if 70 <= cgm <= 180:
        reward = 1.0
        if cgm < 90 and delta < -3:
            reward -= 0.3
        if cgm > 160 and delta > 3:
            reward -= 0.3
    elif 55 <= cgm < 70:
        reward = -np.exp((70 - cgm) / 5.0)
    elif 180 < cgm <= 250:
        reward = -(cgm - 180) / 70
    elif cgm < 55:
        reward = -50.0
    else:
        reward = -5.0
    return reward


# ---------------------------------------------------------------------------
# Load actor
# ---------------------------------------------------------------------------

max_action = torch.tensor([1.5, 5.0], dtype=torch.float32).to(device)
agent = RecurrentTD3Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, seq_len=SEQ_LEN)
agent.load(LOAD_PATH)
agent.actor.eval()
print(f"Actor loaded from {LOAD_PATH}")

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

patient = T1DPatient.withName("adult#001")
sensor = CGMSensor.withName("Dexcom")
pump = InsulinPump.withName("Insulet")
scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

obs = env.reset()

step = 0
done = False
total_reward = 0.0

cgm_history = deque(maxlen=3)
meal_history = deque(maxlen=72)
state_buffer = deque(maxlen=SEQ_LEN)

prev_basal = 0.0
prev_bolus = 0.0

# Pre-fill state buffer with zeros
init_state = np.zeros(STATE_DIM, dtype=np.float32)
for _ in range(SEQ_LEN):
    state_buffer.append(init_state)

# Stats tracking
cgm_log = []
basal_log = []
bolus_log = []
reward_log = []

print(f"\n{'Step':>5}  {'CGM':>7}  {'Delta':>6}  {'Basal':>6}  {'Bolus':>6}  {'Reward':>8}")
print("-" * 50)

while step < MAX_STEPS and not done:

    current_cgm = float(obs.observation.CGM)
    cgm_history.append(current_cgm)
    meal_history.append(float(obs.info['meal']))

    # Need at least 3 CGM readings for trend features
    if len(cgm_history) < 3:
        obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
        step += 1
        continue

    cgm_list = list(cgm_history)
    delta = current_cgm - cgm_list[-2]

    state = build_state(
        cgm=current_cgm,
        prev_cgm=cgm_list[-2],
        prev_prev_cgm=cgm_list[-3],
        iob=get_iob(env.patient),
        prev_basal=prev_basal,
        prev_bolus=prev_bolus,
        meal_history=meal_history
    )
    state_buffer.append(state)
    state_seq = np.array(state_buffer, dtype=np.float32)

    # ---- Deterministic action (no noise) ----
    with torch.no_grad():
        state_t = torch.tensor(state_seq).unsqueeze(0).to(device)
        action = agent.actor(state_t).detach().cpu().numpy()[0]

    # Clip to valid physiological range
    action = np.clip(action, [0.0, 0.0], [1.5, 5.0])
    basal, bolus = float(action[0]), float(action[1])

    prev_basal, prev_bolus = basal, bolus

    # ---- Environment step ----
    obs = env.step(PatientAction(basal, bolus), cho=0.0)
    done = obs.done
    next_cgm = float(obs.observation.CGM)
    next_delta = next_cgm - current_cgm

    # ---- Reward ----
    if done and next_cgm < 40.0:
        reward = -1000.0
    else:
        reward = glucose_reward(next_cgm, next_delta)
        reward -= 0.01 * abs(next_delta)

    total_reward += reward
    step += 1

    cgm_log.append(next_cgm)
    basal_log.append(basal)
    bolus_log.append(bolus)
    reward_log.append(reward)

    print(f"{step:>5}  {current_cgm:>7.1f}  {delta:>+6.1f}  {basal:>6.3f}  {bolus:>6.3f}  {reward:>8.2f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

cgm_arr = np.array(cgm_log)
time_in_range = np.mean((cgm_arr >= 70) & (cgm_arr <= 180)) * 100
time_hypo = np.mean(cgm_arr < 70) * 100
time_hyper = np.mean(cgm_arr > 180) * 100
severe_hypo = np.mean(cgm_arr < 54) * 100

print("\n" + "=" * 50)
print("TEST SUMMARY")
print("=" * 50)
print(f"  Steps completed  : {step}")
print(f"  Total reward     : {total_reward:.2f}")
print(f"  Mean CGM         : {cgm_arr.mean():.1f} mg/dL")
print(f"  CGM std          : {cgm_arr.std():.1f} mg/dL")
print(f"  Time in range    : {time_in_range:.1f}%  (70–180 mg/dL)")
print(f"  Time hypo        : {time_hypo:.1f}%   (< 70 mg/dL)")
print(f"  Time severe hypo : {severe_hypo:.1f}%   (< 54 mg/dL)")
print(f"  Time hyper       : {time_hyper:.1f}%  (> 180 mg/dL)")
print(f"  Mean basal       : {np.mean(basal_log):.3f} U/h")
print(f"  Mean bolus       : {np.mean(bolus_log):.3f} U")
print("=" * 50)