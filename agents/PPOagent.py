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
from envs.myenv import CustomT1DSimEnv, PatientAction
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

START_TIME = datetime(2018, 1, 1, 6, 0, 0)
SAVE_PATH  = r"/"
STATE_DIM  = 16
SEQ_LEN    = 12
MAX_STEPS  = 480


class ActorCritic(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.basal_mean = nn.Linear(128, 1)
        self.bolus_mean = nn.Linear(128, 1)
        self.log_std_basal = nn.Parameter(torch.tensor([-2.0]))  # std ≈ 0.37
        self.log_std_bolus = nn.Parameter(torch.tensor([-1.5])) # ← add this
        self.value_head = nn.Linear(128, 1)
        self.ent_coef = 0.02
        for layer in [self.basal_mean, self.bolus_mean]:
            nn.init.uniform_(layer.weight, -0.003, 0.003)
        nn.init.constant_(self.basal_mean.bias, -3.0)
        nn.init.uniform_(self.bolus_mean.weight, -0.01, 0.01)
        nn.init.constant_(self.bolus_mean.bias, -3.0)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.trunk(x)
        basal_mean = torch.tanh(self.basal_mean(h)) * 0.375 + 0.375
        bolus_mean = F.softplus(self.bolus_mean(h))
        bolus_mean = bolus_mean.clamp(0.0, 3.0)
        mean = torch.cat([basal_mean, bolus_mean], dim=-1)
        std = torch.cat([
            torch.exp(self.log_std_basal.clamp(-3, 0)),
            torch.exp(self.log_std_bolus.clamp(-2, 1.0))  # allow larger bolus std
        ])
        value      = self.value_head(h)
        return mean, std, value

    def get_action(self, x):
        mean, std, value = self.forward(x)
        dist   = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_p  = dist.log_prob(action).sum(-1, keepdim=True)
        action_clipped = torch.stack([
            action[:, 0].clamp(0.0, 0.75),
            action[:, 1].clamp(0.0, 3.0),
        ], dim=1)
        return action_clipped, log_p, value

    def evaluate(self, x, action):
        mean, std, value = self.forward(x)
        dist  = torch.distributions.Normal(mean, std)
        log_p = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_p, value, entropy



class PPOAgent:
    def __init__(self, state_dim):
        self.net       = ActorCritic(state_dim).to(device)

        self.clip_eps = 0.1  # was 0.2, tighter clipping
        self.n_epochs = 4  # was 10, fewer passes per batch
        self.policy_optimizer = torch.optim.Adam(
            list(self.net.trunk.parameters()) +
            list(self.net.basal_mean.parameters()) +
            list(self.net.bolus_mean.parameters()) +
            [self.net.log_std_basal, self.net.log_std_bolus],
            lr=1e-4, eps=1e-5
        )
        self.value_optimizer = torch.optim.Adam(
            self.net.value_head.parameters(),
            lr=3e-4, eps=1e-5
        )
        self.gamma      = 0.97
        self.gae_lambda = 0.95
        self.ent_coef   = 0.01
        self.batch_size = 64

        self.reset_buffer()

    def reset_buffer(self):
        self.buf_states   = []
        self.buf_actions  = []
        self.buf_log_ps   = []
        self.buf_rewards  = []
        self.buf_values   = []
        self.buf_dones    = []

    def store(self, state_seq, action, log_p, reward, value, done):
        self.buf_states.append(state_seq)
        self.buf_actions.append(action)
        self.buf_log_ps.append(log_p)
        self.buf_rewards.append(reward)
        self.buf_values.append(value)
        self.buf_dones.append(done)

    @torch.no_grad()
    def select_action(self, state_seq):
        state_t = torch.tensor(state_seq).unsqueeze(0).to(device)
        action, log_p, value = self.net.get_action(state_t)
        return (action.cpu().numpy()[0],
                log_p.cpu().numpy()[0, 0],
                value.cpu().numpy()[0, 0])

    def compute_gae(self, last_value):
        """Generalized Advantage Estimation."""
        n       = len(self.buf_rewards)
        advs    = np.zeros(n, dtype=np.float32)
        gae     = 0.0
        values  = self.buf_values + [last_value]

        for t in reversed(range(n)):
            delta = (self.buf_rewards[t]
                     + self.gamma * values[t+1] * (1 - self.buf_dones[t])
                     - values[t])
            gae   = delta + self.gamma * self.gae_lambda * (1 - self.buf_dones[t]) * gae
            advs[t] = gae

        returns = advs + np.array(self.buf_values, dtype=np.float32)
        return advs, returns

    def update(self, last_value):
        advs, returns = self.compute_gae(last_value)

        # Normalize both advantages and returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states = torch.tensor(np.array(self.buf_states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.buf_actions), dtype=torch.float32).to(device)
        old_lps = torch.tensor(np.array(self.buf_log_ps), dtype=torch.float32).unsqueeze(1).to(device)
        advs_t = torch.tensor(advs, dtype=torch.float32).unsqueeze(1).to(device)
        rets_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)

        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        n = len(self.buf_states)
        idx = np.arange(n)

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_lps = old_lps[mb]
                mb_advs = advs_t[mb]
                mb_rets = rets_t[mb]

                new_lps, values, entropy = self.net.evaluate(mb_states, mb_actions)

                ratio = torch.exp(new_lps - mb_old_lps)
                surr1 = ratio * mb_advs
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                p_loss = policy_loss + self.ent_coef * entropy_loss
                self.policy_optimizer.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.net.trunk.parameters()) +
                    list(self.net.basal_mean.parameters()) +
                    list(self.net.bolus_mean.parameters()) +
                    [self.net.log_std_basal, self.net.log_std_bolus],
                    0.5
                )
                self.policy_optimizer.step()

                # Value step — second fresh forward pass, own graph
                _, values_fresh, _ = self.net.evaluate(mb_states, mb_actions)
                value_loss = nn.MSELoss()(values_fresh, mb_rets)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.net.value_head.parameters(), 0.5)
                self.value_optimizer.step()

        self.reset_buffer()
        return policy_loss.item(), value_loss.item()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), f"{path}/ppo_net.pth")
        print(f"Saved → {path}")

    def load(self, path):
        self.net.load_state_dict(torch.load(f"{path}/ppo_net.pth", map_location=device))
        print(f"Loaded ← {path}")


def build_state(patient_states, prev_basal, prev_bolus, current_meal):
    normalized = [s / norm for s, norm in zip(patient_states, [
        1000, 50000, 1000,
        500, 500, 500,
        1000, 1000, 1000,
        100, 100, 100,
        500,
    ])]
    return np.array(normalized + [
        prev_basal / 0.75,
        prev_bolus / 3.0,
        current_meal / 80.0,
    ], dtype=np.float32)


def glucose_reward(cgm, prev_cgm=None, basal=0, bolus=0, meal = 0, step = 0):
    if prev_cgm is None:
        return 0.0
    trend = cgm - prev_cgm
    reward = 0
    if trend < -0.5:  # meaningfully falling
        if cgm <= 50:
            reward -= bolus * 5 + basal * 3
        elif cgm <= 70:
            reward -= bolus * 3 + basal
        elif cgm <= 180 and meal == 0.0:
            reward += bolus * trend * (cgm - 125) / 55
        elif cgm <= 240:
            reward += bolus * 1.5
        else:
            reward += bolus * 2.6

    elif trend > 3:
        if cgm <= 50:
            reward -= bolus * 4 + basal * 1.5
        elif cgm <= 70:
            reward -= bolus * 2 + basal * 0.5
        elif cgm <= 180 and meal == 0.0:
            reward += bolus * (trend / 10) * (cgm - 125) / 55
        elif cgm <= 240:
            reward += bolus * 2
        else:
            reward += bolus * 4
    if 70 <= cgm <= 180:
        reward += 1.0
    elif 180 < cgm <= 240:
        reward -= 0.5
    elif cgm > 240:
        reward -= 2.0
    elif cgm < 70:
        reward -= 3.
    if step == 480 :
        reward += 100
    return reward

def run_ppo_episode(env, agent):
    obs = env.reset()

    state_buffer = deque(maxlen=SEQ_LEN)
    prev_basal, prev_bolus = 0.0, 0.0
    prev_cgm     = None
    total_reward = 0.0
    step         = 0
    done         = False
    episode_log  = []

    for _ in range(SEQ_LEN):
        state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

    while step < MAX_STEPS:

        current_cgm = float(obs.observation.CGM)

        state = build_state(
            list(env.patient.state),
            prev_basal,
            prev_bolus,
            float(obs.info['meal'])
        )
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)

        action, log_p, value = agent.select_action(state_seq)
        basal = float(action[0])
        bolus = float(action[1])

        prev_basal, prev_bolus = basal, bolus

        obs  = env.step(PatientAction(basal, bolus), cho=0.0)
        done = obs.done
        next_cgm = float(obs.observation.CGM)
        meal = float(obs.info['meal'])
        if next_cgm < 40.0:
            reward = -100.0
        else:
            reward = glucose_reward(next_cgm, prev_cgm=current_cgm,
                                    basal=basal, bolus=bolus, meal= meal, step = step)

        trend = current_cgm - next_cgm
        episode_log.append((step, next_cgm, basal, bolus, reward, done, trend))

        total_reward += reward
        agent.store(state_seq, action, log_p, reward, value, float(done))
        prev_cgm = current_cgm
        step    += 1

    # Print full episode
    print(f"{'step':>4} {'CGM':>7} {'basal':>7} {'bolus':>7} {'reward':>8} {'done':>5} {'trend':>8}")
    for s, cgm, b, bo, r, d, tr in episode_log:
        print(f"{s:>4} {cgm:>7.1f} {b:>7.3f} {bo:>7.3f} {r:>8.2f} {str(d):>5} {tr:>8.2f}")

    if not done:
        state = build_state(
            list(env.patient.state),
            prev_basal,
            prev_bolus,
            float(obs.info['meal'])
        )
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)
        state_t   = torch.tensor(state_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, last_value = agent.net.get_action(state_t)
            last_value = last_value.cpu().numpy()[0, 0]
    else:
        last_value = 0.0

    return total_reward, step, last_value


if __name__ == "__main__":

    meal_scenario = [(1, 45), (6, 70), (10, 20), (12, 80)]
    patient  = T1DPatient.withName("adult#001")
    sensor   = CGMSensor.withName("Dexcom")
    pump     = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START_TIME, scenario=meal_scenario)
    env      = CustomT1DSimEnv(patient=patient, sensor=sensor, pump=pump, scenario=scenario)

    agent = PPOAgent(state_dim=STATE_DIM)

    best_reward  = -np.inf
    log_path     = "../training_log_ppo.txt"
    with open(log_path, "w") as f:
        f.write("PPO Training Log\n")

    UPDATE_EVERY  = 480
    MAX_EPISODES  = 5000
    episode       = 0
    total_steps   = 0
    update_count  = 0
    reward_window = deque(maxlen=50)

    print("Starting training...")

    while episode < MAX_EPISODES:

        steps_collected = 0
        last_val        = 0.0

        while steps_collected < UPDATE_EVERY:
            r, s, last_val = run_ppo_episode(env, agent)
            steps_collected += s
            total_steps     += s
            episode         += 1
            reward_window.append(r)
            rolling_avg = np.mean(reward_window)

            print(f"Episode {episode}: reward={r:.2f}, steps={s}, "
                  f"total_steps={total_steps}, avg50={rolling_avg:.2f}")

            with open(log_path, "a") as f:
                f.write(f"{episode},{r:.2f}\n")

            if r > best_reward:
                best_reward = r
                agent.save(SAVE_PATH + "/best")
                print(f"  New best: {best_reward:.2f}")

            if episode % 500 == 0:
                agent.save(SAVE_PATH + f"/checkpoint_{episode}")

            if episode >= MAX_EPISODES:
                break

        pl, vl       = agent.update(last_val)
        update_count += 1
        print(f"  [Update {update_count}] policy_loss={pl:.4f}  value_loss={vl:.4f}")

    agent.save(SAVE_PATH + "/final")
    print(f"Training complete. Best reward: {best_reward:.2f}")