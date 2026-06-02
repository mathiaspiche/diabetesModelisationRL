# Diabetes modelisation with Reinforcement Learning 
Using the simglucose Simulator https://github.com/jxx123/simglucose, trying to predict blood glucose with reinforcement learning by training an agent able to manipulate injections. 

DONE: 
- Coded full TD3 DDPG agent (twin critic delayed). Changed from 
initial simple DDPG architecture, because training was too unstable.
- Tested on multiple meal scenarios on patient 'adult_001', agent takes
about 2500 episodes to converge to a policy it considers optimal... complete

TO DO : 

- Comment code files.
- Implement SAC (soft actor-critic) and test convergence speed and 
resulting policies.
- Reconsider state space encoding. Maybe not the best one we've chosen.

EVENTUALLY : 

- Implement a physical exercice component in simglucose. Physical 
activity's impact on insulin and carbohydrates absorption is a big
challenge in patients affected by T1D, so it would be interesting to study 
how an RL agent learns to adjust BG management to it. 

