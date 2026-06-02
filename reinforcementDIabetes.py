import os
import torch
import torch.nn as nn
import random
from collections import deque
import numpy as np
from datetime import datetime

from simglucose.simulation.scenario import CustomScenario
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from myenv import CustomT1DSimEnv, PatientAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

START_TIME = datetime(2018, 1, 1, 6, 0, 0)

# ---------------------------------------------------------------------------
# Networks
# ---------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]  # take latest timestep only

        action = self.net(x)
        return action * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x, action):
        # x: [batch, state_dim] OR [batch, seq_len, state_dim]
        if x.dim() == 3:
            x = x[:, -1, :]  # keep only latest state

        return self.net(torch.cat([x, action], dim=1))

class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim)
        self.q2 = Critic(state_dim, action_dim)

    def forward(self, x, action):
        return self.q1(x, action), self.q2(x, action)

    def q1_only(self, x, action):
        return self.q1(x, action)

class RecurrentTD3Agent:
    def __init__(self, state_dim, action_dim=2, seq_len=12):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        self.max_action = torch.tensor([1.5, 5.0], dtype=torch.float32).to(device)
        self.min_action = torch.tensor([0.0, 0.0], dtype=torch.float32).to(device)

        # Networks
        self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)

        self.critic = DoubleCritic(state_dim, action_dim).to(device)
        self.critic_target = DoubleCritic(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Replay buffer (stores sequences now)
        self.replay_buffer = deque(maxlen=200_000)

        # TD3 params
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.15
        self.noise_clip = 0.4
        self.policy_delay = 2
        self.total_it = 0

    # -----------------------------
    def store_transition(self, state_seq, action, reward, next_state_seq, done):
        self.replay_buffer.append((state_seq, action, reward, next_state_seq, done))

    # -----------------------------
    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1
        batch = random.sample(self.replay_buffer, batch_size)

        state_seq, actions, rewards, next_state_seq, dones = zip(*batch)

        state_seq = torch.tensor(np.array(state_seq), dtype=torch.float32).to(device)
        next_state_seq = torch.tensor(np.array(next_state_seq), dtype=torch.float32).to(device)

        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # ---------------- TARGET ----------------
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_actions = (self.actor_target(next_state_seq) + noise).clamp(
                self.min_action, self.max_action
            )

            q1_t, q2_t = self.critic_target(next_state_seq, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_t, q2_t)

        # ---------------- CRITIC ----------------
        q1, q2 = self.critic(state_seq, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # ← add here
        self.critic_optimizer.step()

        actor_loss_val = None

        # ---------------- ACTOR ----------------
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(
                state_seq,
                self.actor(state_seq)
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # ← add here
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_val = actor_loss.item()

        return critic_loss.item(), actor_loss_val

    # -----------------------------
    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        print(f"Saved → {path}")

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth",  map_location=device))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth", map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"Loaded ← {path}")


def get_iob(patient) -> float:
    s1 = float(patient.state[10])
    s2 = float(patient.state[11])
    return (s1 + s2) / 2000.0


def build_state(
    cgm,
    prev_cgm,
    prev_prev_cgm,
    iob,
    prev_basal,
    prev_bolus,
    meal_history
):

    # ---------------- CGM ----------------
    cgm_norm = cgm / 400.0

    # ---------------- trend ----------------
    delta = cgm - prev_cgm
    delta2 = cgm - 2 * prev_cgm + prev_prev_cgm

    delta_norm = delta / 50.0
    delta2_norm = delta2 / 50.0

    # ---------------- IOB (USE ONLY THIS) ----------------
    iob_norm = min(iob, 3.0) / 3.0

    # ---------------- meals ----------------
    meal_sum = sum(list(meal_history)[-36:]) / 100.0

    # ---------------- previous actions ----------------
    basal_norm = prev_basal / 1.5
    bolus_norm = prev_bolus / 5.0

    return np.array([
        cgm_norm,
        delta_norm,
        delta2_norm,
        iob_norm,
        meal_sum,
        basal_norm,
        bolus_norm
    ], dtype=np.float32)

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

SAVE_PATH = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation"

if __name__ == "__main__":

    MAX_STEPS = 480
    SEQ_LEN = 12

    meal_scenario = [(1, 45), (6, 70), (10, 20), (12, 80)]

    patient  = T1DPatient.withName("adult#001")
    sensor   = CGMSensor.withName("Dexcom")
    pump     = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)

    env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    agent = RecurrentTD3Agent(state_dim=7, action_dim=2, seq_len=SEQ_LEN)

    explore_noise = np.array([0.08, 0.05])
    explore_noise_decay = 0.995
    explore_noise_min = np.array([0.01, 0.01])

    best_reward = -np.inf

    log_path = "training_log.txt"
    with open(log_path, "w") as f:
        f.write("Recurrent TD3 Training Log\n")

    for episode in range(2500):

        obs = env.reset()

        step = 0
        done = False
        total_reward = 0.0

        cgm_history = deque(maxlen=3)
        meal_history = deque(maxlen=72)
        state_buffer = deque(maxlen=SEQ_LEN)

        last_cgm = float(obs.observation.CGM)
        prev_basal = 0.0
        prev_bolus = 0.0

        init_state = np.zeros(7, dtype=np.float32)
        for _ in range(SEQ_LEN):
            state_buffer.append(init_state)

        while step < MAX_STEPS and not done:

            current_cgm = float(obs.observation.CGM)
            cgm_history.append(current_cgm)
            meal_history.append(float(obs.info['meal']))

            if len(cgm_history) < 3:
                obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
                step += 1
                continue

            cgm_list = list(cgm_history)
            delta = current_cgm - cgm_list[-2]  # ← compute delta here

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

            # ---------------- ACTION ----------------
            state_t = torch.tensor(state_seq).unsqueeze(0).to(device)
            action = agent.actor(state_t).detach().cpu().numpy()[0]
            noise = np.random.normal(0, explore_noise)
            action = np.clip(action, [0.0, 0.0], [1.5, 5.0])

            basal, bolus = float(action[0]), float(action[1])
            old_basal, old_bolus = prev_basal, prev_bolus  # ← save before update
            prev_basal, prev_bolus = basal, bolus

            # ---------------- ENV STEP ----------------
            obs = env.step(PatientAction(basal, bolus), cho=0.0)
            done = obs.done
            next_cgm = float(obs.observation.CGM)
            next_delta = next_cgm - current_cgm  # ← delta for next state

            # ---------------- REWARD ----------------
            if done and next_cgm < 40.0:
                reward = -1000.0
                done = True
            else:
                reward = glucose_reward(next_cgm, next_delta)  # cgm + trend
                reward -= 0.01 * abs(next_delta)  # penalize large swings
                reward -= 0.01 * (abs(basal - old_basal) + abs(bolus - old_bolus))  # smoothness

            total_reward += reward

            next_state = build_state(
                cgm=next_cgm,
                prev_cgm=current_cgm,
                prev_prev_cgm=cgm_list[-1],
                iob=get_iob(env.patient),
                prev_basal=basal,
                prev_bolus=bolus,
                meal_history=meal_history
            )

            state_buffer.append(next_state)
            next_state_seq = np.array(state_buffer, dtype=np.float32)

            agent.store_transition(state_seq, action.tolist(), reward, next_state_seq, done)
            agent.train()

            step += 1

            if episode % 50 == 0:
                print(
                    f"[{step}] CGM={current_cgm:.1f} delta={delta:+.1f} "
                    f"basal={basal:.3f} bolus={bolus:.3f} "
                    f"reward={reward:.2f}"
                )

        explore_noise = np.maximum(explore_noise_min, explore_noise * explore_noise_decay)
        if (episode + 1) % 200 == 0:
            agent.replay_buffer.clear()
        print(f"Episode {episode+1}: reward={total_reward:.2f}, nb_steps : {step}")

        with open(log_path, "a") as f:
            f.write(f"{episode+1},{total_reward:.2f}\n")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(SAVE_PATH + "/best")

        if (episode + 1) % 500 == 0:
            agent.save(SAVE_PATH + f"/checkpoint_{episode+1}")

    agent.save(SAVE_PATH + "/final")