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
from envs.myenv import CustomT1DSimEnv, PatientAction
from utils.config_locale import BASE
from utils.random_meals import random_meal_scenario

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
RISK_LOOKBACK = 5
START_TIME = datetime(2018, 1, 1, 6, 0, 0)
STATE_DIM  = 6
SEQ_LEN    = 12
MAX_STEPS  = 480
MAX_BASAL  = 0.75
MAX_BOLUS  = 3.0
GAMMA = 0.99
class Actor(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
        nn.init.uniform_(self.head[-1].weight, -0.003, 0.003)
        with torch.no_grad():
            self.head[-1].bias[0] = -0.4
            self.head[-1].bias[1] = -4.0

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        feat   = out[:, -1, :]
        a      = self.head(feat)
        basal  = torch.sigmoid(a[:, 0:1]) * MAX_BASAL
        bolus  = torch.sigmoid(a[:, 1:2]) * MAX_BOLUS
        return torch.cat([basal, bolus], dim=1)


class Critic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, action):
        if x.dim() == 3:  # (B, seq, dim) -> last frame
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

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = deque(maxlen=500_000)

        self.gamma        = GAMMA
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

    def save(self, path, extra=None):
        os.makedirs(path, exist_ok=True)
        payload = {
            'state_dim': STATE_DIM,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }
        if extra:  # [FIX-8] resume state lives here
            payload.update(extra)
        torch.save(payload, f"{path}/agent.pth")
        print(f"Saved -> {path}")

    def load(self, path):
        """Returns a dict of resume state, or {} if nothing/incompatible loaded."""
        ckpt_path = f"{path}/agent.pth"
        if not os.path.exists(ckpt_path):
            print(f"[load] no checkpoint at {path}; starting fresh.")
            return {}

        ckpt = torch.load(ckpt_path, map_location=device)

        # architecture-compatibility guard
        if ckpt.get('state_dim') != STATE_DIM:
            print(f"[load] WARNING: checkpoint state_dim={ckpt.get('state_dim')} "
                  f"!= current {STATE_DIM}. Architecture changed; starting fresh.")
            return {}
        try:
            self.actor.load_state_dict(ckpt['actor'])
            self.critic.load_state_dict(ckpt['critic'])
            self.actor_target.load_state_dict(ckpt['actor_target'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
            self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            self.total_it = ckpt['total_it']
        except (RuntimeError, KeyError) as e:
            print(f"[load] WARNING: incompatible checkpoint ({e}); starting fresh.")
            return {}

        print(f"[load] full state restored from {path}")
        return {k: ckpt[k] for k in ('total_steps', 'explore_noise', 'episode')
                if k in ckpt}


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


def glucose_reward(cgm, meal=0.0, bolus=0.0, basal=0.0, iob=0.0):
    rew = max(0.0, 1.0 - abs(cgm - 120) / 70.0)
    if cgm < 80:
        rew -= (80 - cgm) * 0.05
    if meal == 0.0 and bolus > 0.0:
        rew -= 0.8 * bolus
    return rew

def run_episode(explore_noise, episode_num, total_steps):
    meal_scenario = random_meal_scenario()
    scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
    env.scenario = scenario
    obs = env.reset()

    cgm_history  = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque(maxlen=SEQ_LEN)
    risk_window  = deque(maxlen=RISK_LOOKBACK + 1)   # holds r_{t-5} .. r_t
    prev_basal   = 0.0
    total_reward = 0.0
    step         = 0
    done         = False
    episode_log  = []

    # prime the LSTM window with zeros
    for _ in range(SEQ_LEN):
        state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

    while step < MAX_STEPS and not done:
        current_cgm = float(obs.observation.CGM)
        cgm_history.append(current_cgm)
        meal = float(obs.info['meal'])
        meal_history.append(meal)

        # warm-up: need a few CGM readings before deltas are valid
        if len(cgm_history) < 3:
            obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
            step += 1
            total_steps += 1
            continue

        cgm_list = list(cgm_history)
        iob      = get_iob(env.patient)

        # current state -> push into the rolling window
        state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                            iob, prev_basal, meal_history)
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)

        # 1) agent selects, then add exploration noise -> REQUESTED action
        basal, bolus = agent.select_action(state_seq)
        basal = float(np.clip(basal + np.random.normal(0, explore_noise), 0.0, MAX_BASAL))
        if meal != 0.0 and np.random.rand() < explore_noise * 2:
            bolus = float(np.random.uniform(0.0, MAX_BOLUS))
        else:
            bolus = float(np.clip(bolus + np.random.normal(0, explore_noise), 0.0, MAX_BOLUS))

        # 2) step the env (env applies the low-glucose suspend internally)
        obs      = env.step(PatientAction(basal, bolus), cho=0.0)
        done     = obs.done
        next_cgm = float(obs.observation.CGM)
        next_iob = get_iob(env.patient)

        # 3) overwrite with what the pump ACTUALLY delivered (post-suspend)
        basal = obs.info.get('delivered_basal', basal)
        bolus = obs.info.get('delivered_bolus', bolus)
        prev_basal = basal

        current_risk = float(obs.info['risk'])
        risk_window.append(current_risk)

        if next_cgm <= 39.0:
            reward = (-current_risk / 10.0) / (1.0 - GAMMA)
            done   = True
        elif len(risk_window) == risk_window.maxlen:
            reward = (risk_window[0] - current_risk) / 10.0
        else:
            reward = 0.0
        total_reward += reward

        next_state = build_state(next_cgm, current_cgm, cgm_list[-2],
                                 next_iob, basal, meal_history)
        next_state_seq = np.array(list(state_buffer)[1:] + [next_state], dtype=np.float32)
        state_buffer.append(next_state)

        agent.store_transition(state_seq, basal, bolus, reward, next_state_seq, done)

        if total_steps >= RANDOM_STEPS:
            c_loss, a_loss = agent.train()
            if total_steps % 2000 == 0 and c_loss is not None:
                print(f"[train] step={total_steps}  critic={c_loss:.4f}  actor={a_loss}  "
                      f"out_bias={[round(b, 3) for b in agent.actor.head[-1].bias.data.tolist()]}")

        episode_log.append((step, next_cgm, basal, bolus, reward, meal))
        step += 1
        total_steps += 1

        if done:
            break

    if episode_num % 10 == 0:
        print(f"{'step':>4} {'CGM':>7} {'basal':>7} {'bolus':>7} {'reward':>8} {'meal':>8}")
        for s, cgm, b, bo, r, m in episode_log:
            print(f"{s:>4} {cgm:>7.1f} {b:>7.3f} {bo:>7.3f} {r:>8.2f} {m:>8.2f}")

    return total_reward, step, total_steps

if __name__ == "__main__":
    RANDOM_STEPS = 12500
    total_steps = 0
    patient = T1DPatient.withName("adult#001")
    sensor = CGMSensor.withName("Dexcom")
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario = [])
    env = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    agent = TD3Agent(state_dim=STATE_DIM)
    explore_noise = 0.11
    explore_noise_decay = 0.999
    explore_noise_min = 0.005
    best_reward         = -np.inf

    log_path = "training_log.txt"
    with open(log_path, "w") as f:
        f.write("TD3 Basal+Bolus Training Log\n")

    print("Starting RL training (basal + bolus)...")
    start_ep = 0
    for episode in range(start_ep, start_ep + 3000):
        r, s, total_steps = run_episode(explore_noise, episode + 1, total_steps)

        if total_steps >= RANDOM_STEPS:
            explore_noise = max(explore_noise_min, explore_noise * explore_noise_decay)

        print(f"Episode {episode + 1}: reward={r:.2f}, steps={s}, "
              f"noise={explore_noise:.4f}, total_steps={total_steps}")


        with open(log_path, "a") as f:
            f.write(f"{episode+1},{r:.2f}\n")

        if (episode + 1) % 500 == 0:
            agent.save(BASE + f"/checkpoint_{episode+1}")

        if episode % 1 == 0:
            with torch.no_grad():
                test_s = torch.zeros(1, SEQ_LEN, STATE_DIM).to(device)
                a = agent.actor(test_s).cpu().numpy()[0]
                print(f"  Actor sanity: basal={a[0]:.4f} U/hr  bolus={a[1]:.4f} U")

    agent.save(BASE + "/final")
    print(f"Training complete. Best reward: {best_reward:.2f}")