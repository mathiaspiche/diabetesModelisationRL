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
from envs.myenv import CustomT1DSimEnv, PatientAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

START_TIME = datetime(2018, 1, 1, 6, 0, 0)
SAVE_PATH  = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation"
STATE_DIM  = 6
SEQ_LEN    = 12
MAX_STEPS  = 480
MAX_BASAL  = 0.75
MAX_BOLUS  = 3.0

MEAL_SCENARIOS = [
    [(1, 45), (6, 70), (10, 20), (12, 80)],
    [(2, 60), (7, 80), (12, 50)],
    [(1, 30), (5, 40), (9, 30), (13, 60), (17, 45)],
    [(3, 90), (11, 70)],
    [(1, 20), (6, 50), (12, 100)],
    [(8, 80), (14, 60)],
    [],
    [(1, 45), (6, 70)],
]

class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 2),
        )
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        nn.init.constant_(self.net[-1].bias, -6.0)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        out   = self.net(x)
        basal = torch.sigmoid(out[:, 0:1]) * MAX_BASAL
        bolus = torch.sigmoid(out[:, 1:2]) * MAX_BOLUS
        return torch.cat([basal, bolus], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, 256), nn.ReLU(),
            nn.Linear(256, 256),           nn.ReLU(),
            nn.Linear(256, 128),           nn.ReLU(),
            nn.Linear(128, 1),
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


class TD3Agent:
    def __init__(self, state_dim):
        self.actor          = Actor(state_dim).to(device)
        self.actor_target   = Actor(state_dim).to(device)
        self.critic         = DoubleCritic(state_dim).to(device)
        self.critic_target  = DoubleCritic(state_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4,
                                                  weight_decay=1e-4)

        self.replay_buffer = deque(maxlen=500_000)

        self.gamma        = 0.97
        self.tau          = 0.005
        self.policy_noise = 0.05
        self.noise_clip   = 0.15
        self.policy_delay = 2
        self.total_it     = 0


    def select_action(self, state_seq):
        state_t = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        return float(action[0]), float(action[1])


    def store_transition(self, state_seq, basal, bolus, reward, next_state_seq, done):
        self.replay_buffer.append((state_seq, basal, bolus, reward, next_state_seq, done))


    def train(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1
        batch = random.sample(self.replay_buffer, batch_size)
        state_seqs, basals, boluses, rewards, next_state_seqs, dones = zip(*batch)

        states      = torch.tensor(np.array(state_seqs),      dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_state_seqs), dtype=torch.float32).to(device)
        basals_t    = torch.tensor(np.array(basals),  dtype=torch.float32).unsqueeze(1).to(device)
        boluses_t   = torch.tensor(np.array(boluses), dtype=torch.float32).unsqueeze(1).to(device)
        actions     = torch.cat([basals_t, boluses_t], dim=1)
        rewards_t   = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        dones_t     = torch.tensor(np.array(dones),   dtype=torch.float32).unsqueeze(1).to(device)


        with torch.no_grad():
            noise        = (torch.randn(batch_size, 2) * self.policy_noise) \
                               .clamp(-self.noise_clip, self.noise_clip).to(device)
            next_actions = (self.actor_target(next_states) + noise)
            next_actions[:, 0] = next_actions[:, 0].clamp(0.0, MAX_BASAL)
            next_actions[:, 1] = next_actions[:, 1].clamp(0.0, MAX_BOLUS)

            q1_t, q2_t = self.critic_target(next_states, next_actions)
            q_target    = rewards_t + self.gamma * (1 - dones_t) * torch.min(q1_t, q2_t)

        q1, q2      = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor,  self.actor_target)
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
        prev_basal / MAX_BASAL,
    ], dtype=np.float32)


def glucose_reward(cgm, prev_cgm=None, basal=0.0, bolus=0.0, meal = 0.0, step = 0):
    if prev_cgm is None:
        return 0.0
    reward = 0
    if meal == 0.0 :
        reward -= bolus * 10
    else :
        reward += bolus
    if bolus <= 0.05 :
        reward -= bolus * 50
    if 70 <= cgm <= 180:
        reward += 1.0
    elif 180 < cgm <= 240:
        reward -= 0.5
    elif cgm > 240:
        reward -= 2.0
    elif cgm < 70:
        reward -= 3 - basal
    if step >= 400 and 70 <= cgm <= 180:
        reward += 3
    if cgm == 39.0 :
        reward -= 10 * bolus + 20 * basal
    return reward
def run_episode(explore_noise, episode_num, total_steps):
    meal_scenario = random.choice(MEAL_SCENARIOS)
    scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
    env.scenario = scenario
    obs = env.reset()

    cgm_history  = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque(maxlen=SEQ_LEN)
    prev_basal   = 0.0
    total_reward = 0.0
    step         = 0
    done         = False
    episode_log  = []

    for _ in range(SEQ_LEN):
        state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

    while step < MAX_STEPS and not done:
        current_cgm = float(obs.observation.CGM)
        cgm_history.append(current_cgm)
        meal = float(obs.info['meal'])
        meal_history.append(meal)

        if len(cgm_history) < 3:
            obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
            step += 1
            total_steps += 1
            continue

        cgm_list = list(cgm_history)
        iob      = get_iob(env.patient)

        state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                            iob, prev_basal, meal_history)
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)

        basal, bolus = agent.select_action(state_seq)
        basal = float(np.clip(basal + np.random.normal(0, explore_noise), 0.0, MAX_BASAL))
        bolus = float(np.clip(bolus + np.random.normal(0, explore_noise * 0.3), 0.0, MAX_BOLUS))

        prev_basal = basal

        obs      = env.step(PatientAction(basal, bolus), cho=0.0)
        done     = obs.done
        next_cgm = float(obs.observation.CGM)
        next_iob = get_iob(env.patient)

        reward = -100.0 if (done and next_cgm < 40.0) else \
            glucose_reward(next_cgm, prev_cgm=current_cgm, basal=basal,
                           bolus=bolus, meal=float(obs.info['meal']), step=step)
        total_reward += reward

        next_state = build_state(next_cgm, current_cgm, cgm_list[-2],
                                 next_iob, basal, meal_history)
        next_state_seq = np.array(list(state_buffer)[1:] + [next_state], dtype=np.float32)
        state_buffer.append(next_state)

        agent.store_transition(state_seq, basal, bolus, reward, next_state_seq, done)

        if total_steps >= RANDOM_STEPS:
            agent.train()

        episode_log.append((step, next_cgm, basal, bolus, reward, meal))
        step += 1
        total_steps += 1

    if episode_num % 10 == 0:
        print(f"{'step':>4} {'CGM':>7} {'basal':>7} {'bolus':>7} {'reward':>8} {'meal':>8}")
        for s, cgm, b, bo, r, m in episode_log:
            print(f"{s:>4} {cgm:>7.1f} {b:>7.3f} {bo:>7.3f} {r:>8.2f} {m : > 8.2f}")

    return total_reward, step, total_steps



if __name__ == "__main__":
    RANDOM_STEPS = 10000
    total_steps = 0
    patient = T1DPatient.withName("adult#001")
    sensor = CGMSensor.withName("Dexcom")
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=MEAL_SCENARIOS[0])
    env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    agent = TD3Agent(state_dim=STATE_DIM)

    explore_noise = 0.05
    explore_noise_decay = 0.999
    explore_noise_min = 0.005
    best_reward         = -np.inf

    log_path = "training_log.txt"
    with open(log_path, "w") as f:
        f.write("TD3 Basal+Bolus Training Log\n")

    print("Starting RL training (basal + bolus)...")
    for episode in range(2500):
        r, s, total_steps = run_episode(explore_noise, episode + 1, total_steps)

        if total_steps >= RANDOM_STEPS:
            explore_noise = max(explore_noise_min, explore_noise * explore_noise_decay)

        print(f"Episode {episode + 1}: reward={r:.2f}, steps={s}, "
              f"noise={explore_noise:.4f}, total_steps={total_steps}")


        with open(log_path, "a") as f:
            f.write(f"{episode+1},{r:.2f}\n")

        if (episode + 1) % 500 == 0:
            agent.save(SAVE_PATH + f"/checkpoint_{episode+1}")

        if episode % 10 == 0:
            with torch.no_grad():
                test_s = torch.zeros(1, SEQ_LEN, STATE_DIM).to(device)
                a = agent.actor(test_s).cpu().numpy()[0]
                print(f"  Actor sanity: basal={a[0]:.4f} U/hr  bolus={a[1]:.4f} U")

    agent.save(SAVE_PATH + "/final")
    print(f"Training complete. Best reward: {best_reward:.2f}")