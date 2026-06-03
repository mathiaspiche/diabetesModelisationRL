# Diabetes modelisation with Reinforcement Learning 
Using the simglucose Simulator https://github.com/jxx123/simglucose, trying to predict blood glucose with reinforcement learning by training an agent able to manipulate injections. 

DONE: 
- Coded full TD3 DDPG agent (twin critic delayed). Changed from 
initial simple DDPG architecture, because training was too unstable.
- Trained on patient `adult#001` with fixed meal scenarios, agent converges after ~2500 episodes
-Tested multiple reward functions, current one - Reward: time-in-range (70–180 mg/dL) 
with penalties for large swings and huge negative rewards for severe hypoglycemia.

TO DO : 

- Solve problem with test file, actor always ouputs 0 
- Comment code files.
- Implement SAC (soft actor-critic) and test convergence speed and 
resulting policies.
- Reconsider state space encoding. Maybe not the best one we've chosen.
State: CGM, glucose trend, IOB, meal history, previous actions (7 features).
- Tune hyperparameters with Pytorch Raytune.

EVENTUALLY : 

- Implement a physical exercice component in simglucose. Physical 
activity's impact on insulin and carbohydrates absorption is a big
challenge in patients affected by T1D, so it would be interesting to study 
how an RL agent learns to adjust BG management to it. 

