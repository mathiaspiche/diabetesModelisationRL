import os
import math
import torch
import torch.nn as nn
import numpy as np
from collections import deque
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
SAVE_PATH  = r"C:\Users\mathi\OneDrive\Documents\diabetesModelisation"
STATE_DIM  = 7
SEQ_LEN    = 12
MAX_STEPS  = 480


class ActorCritic(nn.Module):
    """Shared trunk, separate heads for policy (mean+std) and value."""
    def __init__(self, state_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
        )
        self.basal_mean  = nn.Linear(128, 1)
        self.bolus_mean  = nn.Linear(128, 1)
        self.log_std     = nn.Parameter(torch.zeros(2))
        self.value_head  = nn.Linear(128, 1)

        for layer in [self.basal_mean, self.bolus_mean]:
            nn.init.uniform_(layer.weight, -0.003, 0.003)
        nn.init.constant_(self.basal_mean.bias, 0.0)
        nn.init.constant_(self.bolus_mean.bias, -3.0)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        h = self.trunk(x)
        basal_mean = torch.tanh(self.basal_mean(h)) * 0.375 + 0.375
        bolus_mean = torch.sigmoid(self.bolus_mean(h)) * 3.0
        mean = torch.cat([basal_mean, bolus_mean], dim=-1)
        std        = torch.exp(self.log_std.clamp(-3, 0))
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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4, eps=1e-5)

        self.clip_eps   = 0.2
        self.gamma      = 0.97
        self.gae_lambda = 0.95
        self.vf_coef    = 0.5
        self.ent_coef   = 0.01
        self.n_epochs   = 10
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

        states  = torch.tensor(np.array(self.buf_states),  dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.buf_actions), dtype=torch.float32).to(device)
        old_lps = torch.tensor(np.array(self.buf_log_ps),  dtype=torch.float32).unsqueeze(1).to(device)
        advs_t  = torch.tensor(advs,    dtype=torch.float32).unsqueeze(1).to(device)
        rets_t  = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)

        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        n = len(self.buf_states)
        idx = np.arange(n)

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                mb_states  = states[mb]
                mb_actions = actions[mb]
                mb_old_lps = old_lps[mb]
                mb_advs    = advs_t[mb]
                mb_rets    = rets_t[mb]

                new_lps, values, entropy = self.net.evaluate(mb_states, mb_actions)

                ratio       = torch.exp(new_lps - mb_old_lps)
                surr1       = ratio * mb_advs
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.MSELoss()(values, mb_rets)
                entropy_loss= -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

        self.reset_buffer()
        return policy_loss.item(), value_loss.item()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), f"{path}/ppo_net.pth")
        print(f"Saved → {path}")

    def load(self, path):
        self.net.load_state_dict(torch.load(f"{path}/ppo_net.pth", map_location=device))
        print(f"Loaded ← {path}")




def get_iob(patient):
    return (float(patient.state[10]) + float(patient.state[11])) / 2000.0


def build_state(cgm, prev_cgm, prev_prev_cgm, iob, prev_basal, prev_bolus, meal_history):
    delta  = cgm - prev_cgm
    delta2 = cgm - 2 * prev_cgm + prev_prev_cgm
    return np.array([
        cgm / 400.0,
        delta / 50.0,
        delta2 / 50.0,
        min(iob, 3.0) / 3.0,
        sum(list(meal_history)[-36:]) / 100.0,
        prev_basal / 0.75,
        prev_bolus / 3.0,
    ], dtype=np.float32)


def glucose_reward(cgm, delta, basal, bolus, iob):
    target   = 110.0
    distance = abs(cgm - target)
    reward   = -distance / 150.0

    if cgm < 70:
        reward -= 3.0
    if cgm < 55:
        reward -= 8.0
    if cgm > 250:
        reward -= 2.0
    elif cgm > 180:
        reward -= (cgm - 180) / 140.0

    if cgm < 100 and basal > 0:
        reward -= basal * 2.0
    if cgm < 80 and basal > 0:
        reward -= basal * 4.0
    if cgm < 100 and bolus > 0:
        reward -= bolus * 3.0

    if iob > 1.5 and cgm < 130:
        reward -= 0.3 * (iob - 1.5)

    return reward



def collect_bc_data(env, n_episodes=100):
    print(f"Collecting {n_episodes} BBController demonstrations...")
    demo_states  = []
    demo_actions = []

    for ep in range(n_episodes):
        obs        = env.reset()
        controller = BBController(target=140)
        cgm_history  = deque(maxlen=13)
        meal_history = deque(maxlen=72)
        state_buffer = deque(maxlen=SEQ_LEN)
        prev_basal, prev_bolus = 0.0, 0.0

        for _ in range(SEQ_LEN):
            state_buffer.append(np.zeros(STATE_DIM, dtype=np.float32))

        done = False
        step = 0
        while step < MAX_STEPS and not done:
            current_cgm = float(obs.observation.CGM)
            cgm_history.append(current_cgm)
            meal_history.append(float(obs.info['meal']))

            if len(cgm_history) < 3:
                obs = env.step(PatientAction(0.0, 0.0), cho=0.0)
                step += 1
                continue

            cgm_list = list(cgm_history)
            iob      = get_iob(env.patient)

            state = build_state(current_cgm, cgm_list[-2], cgm_list[-3],
                                iob, prev_basal, prev_bolus, meal_history)
            state_buffer.append(state)
            state_seq = np.array(state_buffer, dtype=np.float32)

            bb_action = controller.policy(
                obs.observation, obs.reward, obs.done,
                patient_name="adult#001", meal=obs.info['meal']
            )
            basal = float(np.clip(bb_action.basal, 0.0, 0.75))
            bolus = float(np.clip(bb_action.bolus, 0.0, 3.0))

            demo_states.append(state_seq.copy())
            demo_actions.append([basal, bolus])

            prev_basal, prev_bolus = basal, bolus
            obs  = env.step(PatientAction(basal, bolus), cho=0.0)
            done = obs.done
            cgm_history.append(float(obs.observation.CGM))
            step += 1

        print(f"  Demo {ep+1}/{n_episodes} | steps={step} | collected={len(demo_states)}")

    return np.array(demo_states, dtype=np.float32), np.array(demo_actions, dtype=np.float32)


def behavioral_cloning(agent, demo_states, demo_actions, epochs=100):
    print("Behavioral cloning...")
    bc_opt  = torch.optim.Adam(agent.net.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(demo_states),
        torch.tensor(demo_actions)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for states_b, actions_b in loader:
            states_b  = states_b.to(device)
            actions_b = actions_b.to(device)

            mean, std, _ = agent.net(states_b)
            loss = nn.MSELoss()(mean, actions_b)

            bc_opt.zero_grad()
            loss.backward()
            bc_opt.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            with torch.no_grad():
                test_s = torch.zeros(1, SEQ_LEN, STATE_DIM).to(device)
                m, _, _ = agent.net(test_s)
                print(f"  Epoch {epoch+1}/{epochs} | loss={total_loss/len(loader):.4f} "
                      f"| basal={m[0,0].item():.3f} bolus={m[0,1].item():.3f}")

    print("Behavioral cloning done.")

def run_ppo_episode(env, agent):
    obs = env.reset()

    cgm_history  = deque(maxlen=13)
    meal_history = deque(maxlen=72)
    state_buffer = deque(maxlen=SEQ_LEN)
    prev_basal, prev_bolus = 0.0, 0.0
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
                            iob, prev_basal, prev_bolus, meal_history)
        state_buffer.append(state)
        state_seq = np.array(state_buffer, dtype=np.float32)

        action, log_p, value = agent.select_action(state_seq)
        basal = float(action[0])
        bolus = float(action[1])

        prev_basal, prev_bolus = basal, bolus

        obs      = env.step(PatientAction(basal, bolus), cho=0.0)
        done     = obs.done
        next_cgm = float(obs.observation.CGM)
        next_delta = next_cgm - current_cgm
        cgm_history.append(next_cgm)

        if done and next_cgm < 40.0:
            reward = -100.0
        else:
            reward = glucose_reward(next_cgm, next_delta, basal, bolus,
                                    get_iob(env.patient))

        total_reward += reward

        agent.store(state_seq, action, log_p, reward, value, float(done))
        step += 1

    if not done:
        state = build_state(float(obs.observation.CGM),
                            list(cgm_history)[-2] if len(cgm_history) >= 2 else float(obs.observation.CGM),
                            list(cgm_history)[-3] if len(cgm_history) >= 3 else float(obs.observation.CGM),
                            get_iob(env.patient), prev_basal, prev_bolus, meal_history)
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
    log_path     = "training_log_ppo.txt"
    with open(log_path, "w") as f:
        f.write("PPO Training Log\n")

    demo_states, demo_actions = collect_bc_data(env, n_episodes=100)
    print(f"Mean BB basal={demo_actions[:,0].mean():.3f}  "
          f"bolus={demo_actions[:,1].mean():.3f}")
    behavioral_cloning(agent, demo_states, demo_actions, epochs=100)

    UPDATE_EVERY = 2048
    episode      = 0
    total_steps  = 0
    update_count = 0

    print("Starting PPO training...")

    while episode < 3000:

        steps_collected = 0
        last_val        = 0.0

        while steps_collected < UPDATE_EVERY:
            r, s, last_val = run_ppo_episode(env, agent)
            steps_collected += s
            total_steps     += s
            episode         += 1

            print(f"Episode {episode}: reward={r:.2f}, steps={s}, "
                  f"total_steps={total_steps}")

            with open(log_path, "a") as f:
                f.write(f"{episode},{r:.2f}\n")

            if r > best_reward:
                best_reward = r
                agent.save(SAVE_PATH + "/best")
                print(f"  New best: {best_reward:.2f}")

            if episode % 500 == 0:
                agent.save(SAVE_PATH + f"/checkpoint_{episode}")

            if episode >= 3000:
                break

        pl, vl       = agent.update(last_val)
        update_count += 1

        with torch.no_grad():
            test_cases = {
                "CGM=180 rising": [180/400,  3/50,  0,      0,    0,   0,   0  ],
                "CGM=120 stable": [120/400,  0,     0,      0.2,  0,   0.3, 0  ],
                "CGM=80 falling": [80/400,  -3/50, -1/50,   0.1,  0,   0.2, 0  ],
                "CGM=60 low":     [60/400,  -2/50,  0,      0.05, 0,   0.1, 0  ],
                "CGM=200 meal":   [200/400,  5/50,  1/50,   0.3,  0.8, 0.5, 0.2],
            }

            print(f"\n--- Sanity (update={update_count} | ep={episode} | "
                  f"steps={total_steps}) ---")
            print(f"  policy_loss={pl:.4f}  value_loss={vl:.4f}")
            print(f"  log_std={agent.net.log_std.detach().cpu().numpy()}")

            for label, state_vals in test_cases.items():
                state_seq = np.tile(
                    np.array(state_vals, dtype=np.float32), (SEQ_LEN, 1)
                )
                state_t = torch.tensor(state_seq).unsqueeze(0).to(device)
                m, std, v = agent.net(state_t)
                print(f"  {label:20s} → basal={m[0,0].item():.3f}  "
                      f"bolus={m[0,1].item():.3f}  "
                      f"V={v[0,0].item():.2f}")
            print()

    agent.save(SAVE_PATH + "/final")