# Collection of DRL algorithms (under construction....)
 In this repository, I try to implement some RL algorithms of my interest, As a starter in reinforcement learning.
  
## DQN

The DQN algotrithm was implemented based on two papers:

 >Mnih, Volodymyr, et al. "Playing Atari with Deep Reinforcement Learning." _arXiv preprint arXiv:1312.5602 (2015).
 >Mnih, Volodymyr, et al. "Human-Level Control Through Deep Reinforcement Learning." _Nature_, vol. 518, no. 7540, 2015, pp. 529–533.
 
The algorithm is like an "upgraded" Q-learning to me, taking the TD error as loss function, i.e.:
	$$
	L=（r+maxQ(s',a)-Q(s,a))^2
	$$
using two network with different parameters as Online network $Q(s,a;\theta)$ and Target network $Q(s,a;\theta')$.
My implementation followed most of the detailed papers,  with some changes like: without gradient cilp, a smaller buffer size, missing no op etc., but it bascically shows how a DQN works properly (I hope so). More discussion about the experiment will be added later on.

The next algorithm to implemented is TRPO.



Plans:  PPO  DDPG  SAC  
