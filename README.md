# Diabetes modelisation with Reinforcement Learning 
Using the simglucose Simulator https://github.com/jxx123/simglucose, trying to predict blood glucose with reinforcement learning by training an agent able to manipulate injections. 

DONE: 
- Coded full TD3 DDPG agent (twin critic delayed). Changed from 
initial simple DDPG architecture, because training was too unstable.
- Trained on patient `adult#001` with fixed meal scenarios, agent converges after ~2500 episodes
-Tested multiple reward functions, current one - Reward: time-in-range (70–180 mg/dL) 
with penalties for large swings and huge negative rewards for severe hypoglycemia
- Coded PPO agent to compare results to DPPG approach, considering DDPG
is prone to estimation bias between actor-critic, whereas PPO considers
its actual policy observation (on-policy).
- Change considerably reward function, initial weight and biases init, 
noise injection in actor actions but still, a major problem remains : 
insulin injection is driven to 0.0. I am trying to fix that.

TO DO : 

- Comment code files.
- Implement SAC (soft actor-critic) and test convergence speed and 
resulting policies.
- Possibly reconsider state encoding (current : cgm (BG level),
        delta (difference between last BG and current),
        delta2 (difference between BG and past two (review this)),
        iob (insulin active in body given by simglucose),
        meal_history (past 72 steps (maybe too much)),
        prev_basal (basal injected at last step))
- Tune hyperparameters with Pytorch Raytune.

EVENTUALLY : 

- Implement a physical exercice component in simglucose. Physical 
activity's impact on insulin and carbohydrates absorption is a big
challenge in patients affected by T1D, so it would be interesting to study 
how an RL agent learns to adjust BG management to it. 

