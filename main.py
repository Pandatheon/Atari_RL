import matplotlib.pyplot as plt
import gym
from DQN import DQNAgent, replay_memory
import Config


env = gym.make('ALE/Breakout-v5')
agent = DQNAgent(env,Config.discount_factor,
                 Config.epsilon, Config.lr, Config.packed, Config.updates)
memory_pool = replay_memory(Config.pool_cap)


return_list = []

for i in range(Config.epochs):
    for j in range(Config.episodes):
        state=memory_pool.preprocessor(env.reset(seed=Config.seed)[0])
        done = False
        G = 0
        while not done:
            # interact
            action = agent.take_action(state.repeat(4,1,1).unsqueeze(dim=0))
            next_state, reward, done, _, _ = env.step(action)
            memory_pool.add(state, action, reward, next_state, done)

            state = memory_pool.preprocessor(next_state)
            G += reward
            # update
            if len(memory_pool) > Config.threshold:
                memories = memory_pool.sample(Config.batch_size)
                agent.update(memories)

        print("epoch{}:episode{}: return:{}".format(i,j,G))
        return_list.append(G)

plt.figure(1)
plt.plot(range(Config.epochs*Config.episodes),return_list)
plt.xlabel("episodes")
plt.ylabel("scores")
plt.show(block=True)

