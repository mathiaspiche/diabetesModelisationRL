import os
import math
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
from datetime import datetime
from simglucose.controller.basal_bolus_ctrller import BBController

from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from myenv import CustomT1DSimEnv, PatientAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

START_TIME = datetime(2018, 1, 1, 6, 0, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, max_basal=0.75):
        super().__init__()
        self.max_basal = max_basal
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        # start outputting ~0.2 U/hr basal
        nn.init.constant_(self.net[-1].bias, math.atanh(0.2 / max_basal))

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        return torch.tanh(self.net(x)) * self.max_basal


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, action):
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(torch.cat([x, action], dim=1))


class DoubleCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.q1 = Critic(state_dim)
        self.q2 = Critic(state_dim)

    def forward(self, x, action):
        return self.q1(x, action), self.q2(x, action)

    def q1_only(self, x, action):
        return self.q1(x, action)


class TD3BasalAgent:
    def __init__(self, state_dim, seq_len=12):
        self.state_dim = state_dim
        self.seq_len   = seq_len
        self.max_basal = 0.75

        self.actor         = Actor(state_dim, self.max_basal).to(device)
        self.actor_target  = Actor(state_dim, self.max_basal).to(device)
        self.critic        = DoubleCritic(state_dim).to(device)
        self.critic_target = DoubleCritic(state_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=2e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-4)

        self.replay_buffer = deque(maxlen=500000)

        self.gamma        = 0.97
        self.tau          = 0.005
        self.policy_noise = 0.05   # smaller since action range is [0, 0.75]
        self.noise_clip   = 0.15
        self.policy_delay = 2
        self.total_it     = 0

    def select_action(self, state_seq):
        state_t = torch.tensor(state_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.actor(state_t).cpu().numpy()[0, 0]

    def store_transition(self, state_seq, basal, reward, next_state_seq, done):
        self.replay_buffer.append((state_seq, basal, reward, next_state_seq, done))

    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1
        batch = random.sample(self.replay_buffer, batch_size)
        state_seq, basals, rewards, next_state_seq, dones = zip(*batch)

        state_seq      = torch.tensor(np.array(state_seq),      dtype=torch.float32).to(device)
        next_state_seq = torch.tensor(np.array(next_state_seq), dtype=torch.float32).to(device)
        basals         = torch.tensor(basals,  dtype=torch.float32).unsqueeze(1).to(device)
        rewards        = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones          = torch.tensor(dones,   dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            noise        = (torch.randn_like(basals) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_state_seq) + noise).clamp(0.0, self.max_basal)
            q1_t, q2_t  = self.critic_target(next_state_seq, next_actions)
            q_target     = rewards + self.gamma * (1 - dones) * torch.min(q1_t, q2_t)

        q1, q2      = self.critic(state_seq, basals)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(state_seq, self.actor(state_seq)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            actor_loss_val = actor_loss.item()

        return critic_loss.item(), actor_loss_val

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        print(f"Saved → {path}")

    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth",  map_location=device))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth", map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"Loaded ← {path}")


def get_iob(patient):
    return (float(patient.state[10]) + float(patient.state[11])) / 2000.0


def build_state(cgm, prev_cgm, prev_prev_cgm, iob, prev_basal, meal_history):
    delta  = cgm - prev_cgm
    delta2 = cgm - 2 * prev_cgm + prev_prev_cgm
    return np.array([
        cgm / 400.0,
        delta / 50.0,
        delta2 / 50.0,
        min(iob, 3.0) / 3.0,
        sum(list(meal_history)[-36:]) / 100.0,
        prev_basal / 0.75,
    ], dtype=np.float32)


def glucose_reward(cgm, delta, basal, iob):
    # dense distance-based reward
    target   = 110.0
    distance = abs(cgm - target)
    reward   = -distance / 150.0       # always in [-2.0, 0]

    # hard zone penalties
    if cgm < 70:
        reward -= 3.0
    if cgm < 55:
        reward -= 8.0
    if cgm > 250:
        reward -= 2.0
    elif cgm > 180:
        reward -= (cgm - 180) / 140.0

    # penalize giving insulin when already low or falling fast
    if cgm < 100 and basal > 0:
        reward -= basal * 2.0
    if cgm < 80 and basal > 0:
        reward -= basal * 4.0

    # penalize high IOB stacking when not needed
    if iob > 1.5 and cgm < 130:
        reward -= 0.3 * (iob - 1.5)

    return reward


def fixed_bolus(cgm, delta, iob):
    """Simple rule-based bolus — not learned."""
    if cgm > 160 and delta >= 0 and iob < 0.8:
        return float(np.clip((cgm - 150) / 80.0, 0.0, 2.0))
    return 0.0


def run_prefill_episode_bb():
    """Run one episode with BBController and store transitions."""
    obs = env.reset()

    # BBController needs patient name to look up basal rates
    controller = BBController(target=140)  # target glucose mg/dL

    cgm_history = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque(maxlen=SEQ_LEN)
    prev_basal = 0.0
    total_reward = 0.0
    step = 0
    done = False

    for _ in range(SEQ_LEN):
        state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

    while step < MAX_STEPS and not done:

        current_cgm = float(obs.observation.CGM)
        cgm_history.append(current_cgm)
        meal_history.append(float(obs.info['meal']))

        if len(cgm_history) < 3:
            obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
            step += 1
            continue

        cgm_list = list(cgm_history)
        iob = get_iob(env.patient)

        state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                            iob, prev_basal, meal_history)
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)

        # ask BBController what to do
        bb_action = controller.policy(
            obs.observation,
            obs.reward,
            obs.done,
            patient_name="adult#001",
            meal=obs.info['meal']
        )
        basal = float(np.clip(bb_action.basal, 0.0, 0.75))
        # ignore BBController bolus — keep fixed rule bolus
        bolus = fixed_bolus(current_cgm, current_cgm - cgm_list[-2], iob)

        prev_basal = basal

        obs = env.step(PatientAction(basal, bolus), cho=0.0)
        done = obs.done
        next_cgm = float(obs.observation.CGM)
        next_delta = next_cgm - current_cgm
        cgm_history.append(next_cgm)

        if done and next_cgm < 40.0:
            reward = -100.0
        else:
            reward = glucose_reward(next_cgm, next_delta, basal, get_iob(env.patient))

        total_reward += reward

        next_state = build_state(next_cgm, current_cgm, cgm_list[-2],
                                 get_iob(env.patient), basal, meal_history)
        next_state_seq = np.array(list(state_buffer)[1:] + [next_state], dtype=np.float32)
        state_buffer.append(next_state)

        agent.store_transition(state_seq, basal, reward, next_state_seq, done)
        step += 1

    return total_reward, step


SAVE_PATH = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation"
STATE_DIM  = 6    # removed bolus_norm since bolus is not learned
SEQ_LEN    = 12


if __name__ == "__main__":

    MAX_STEPS = 480

    meal_scenario = [(1, 45), (6, 70), (10, 20), (12, 80)]
    patient  = T1DPatient.withName("adult#001")
    sensor   = CGMSensor.withName("Dexcom")
    pump     = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
    env      = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    agent = TD3BasalAgent(state_dim=STATE_DIM, seq_len=SEQ_LEN)

    explore_noise       = 0.15
    explore_noise_decay = 0.997
    explore_noise_min   = 0.01
    best_reward         = -np.inf

    log_path = "training_log.txt"
    with open(log_path, "w") as f:
        f.write("TD3 Basal-Only Training Log\n")

    def run_episode(use_rule_based=False):
        obs = env.reset()

        cgm_history  = deque(maxlen=13)
        meal_history = deque(maxlen=72)
        state_buffer = deque(maxlen=SEQ_LEN)
        prev_basal   = 0.0
        total_reward = 0.0
        step         = 0
        done         = False

        for _ in range(SEQ_LEN):
            state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

        while step < MAX_STEPS and not done:

            current_cgm = float(obs.observation.CGM)
            cgm_history.append(current_cgm)
            meal_history.append(float(obs.info['meal']))

            if len(cgm_history) < 3:
                obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
                step += 1
                continue

            cgm_list = list(cgm_history)
            delta    = current_cgm - cgm_list[-2]
            iob      = get_iob(env.patient)

            state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                                iob, prev_basal, meal_history)
            state_buffer.append(state)
            state_seq = np.array(state_buffer, dtype=np.float32)

            # ── basal selection ───────────────────────────────────────
            if use_rule_based:
                basal = float(np.clip((current_cgm - 80) / 200.0, 0.0, 0.75)) \
                    if current_cgm > 80 else 0.0
            else:
                basal = agent.select_action(state_seq)
                basal += np.random.normal(0, explore_noise)
                basal  = float(np.clip(basal, 0.0, 0.75))


            # ── fixed bolus (rule-based always) ───────────────────────
            bolus = fixed_bolus(current_cgm, delta, iob)

            prev_basal = basal

            # ── step ──────────────────────────────────────────────────
            obs        = env.step(PatientAction(basal, bolus), cho=0.0)
            done       = obs.done
            next_cgm   = float(obs.observation.CGM)
            next_delta = next_cgm - current_cgm
            cgm_history.append(next_cgm)

            # ── reward ────────────────────────────────────────────────
            if done and next_cgm < 40.0:
                reward = -100.0
            else:
                reward = glucose_reward(next_cgm, next_delta, basal, get_iob(env.patient))

            total_reward += reward

            # ── next state ────────────────────────────────────────────
            next_state = build_state(next_cgm, current_cgm, cgm_list[-2],
                                     get_iob(env.patient), basal, meal_history)
            next_state_seq = np.array(list(state_buffer)[1:] + [next_state], dtype=np.float32)
            state_buffer.append(next_state)

            agent.store_transition(state_seq, basal, reward, next_state_seq, done)

            if not use_rule_based:
                agent.train()

            step += 1

        return total_reward, step

    # ── phase 1: prefill ──────────────────────────────────────────────
    print("Pre-filling buffer with BBController...")
    for ep in range(50):
        r, s = run_prefill_episode_bb()
        print(f"  Prefill {ep + 1}/50 | steps={s} | reward={r:.1f} | buffer={len(agent.replay_buffer)}")

    # ── phase 2: pretrain critic ──────────────────────────────────────
    print("Pretraining critic...")
    for _ in range(2000):
        agent.train()
    print(f"Done. Buffer: {len(agent.replay_buffer)}")

    # ── phase 3: RL ───────────────────────────────────────────────────
    print("Starting RL training (basal only)...")
    for episode in range(2500):

        r, s = run_episode(use_rule_based=False)

        explore_noise = max(explore_noise_min, explore_noise * explore_noise_decay)
        print(f"Episode {episode+1}: reward={r:.2f}, steps={s}, noise={explore_noise:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{episode+1},{r:.2f}\n")

        if r > best_reward:
            best_reward = r
            agent.save(SAVE_PATH + "/best")
            print(f"  New best: {best_reward:.2f}")

        if (episode + 1) % 500 == 0:
            agent.save(SAVE_PATH + f"/checkpoint_{episode+1}")

        if episode % 10 == 0:
            with torch.no_grad():
                test_s = torch.zeros(1, SEQ_LEN, STATE_DIM).to(device)
                print(f"  Actor sanity: {agent.actor(test_s).item():.4f} U/hr basal")

    agent.save(SAVE_PATH + "/final")