import matplotlib.pyplot as plt
import gym

import utilis
from DQN import DQNAgent, replay_memory
import Config

import argparse
import os

parser = argparse.ArgumentParser("Deep_Q_learning")
parser.add_argument('--episodes', type=float,
                    default=Config.episodes, help='Numbers of episodes')
parser.add_argument('--discount', type=float,
                    default=Config.discount_factor, help='Gamma in formula')
parser.add_argument('--threshold', type=float,
                    default=Config.threshold, help='threshold for starting update')
parser.add_argument('--epsilon_init', type=float,
                    default=Config.epsilon[0], help='initial epsilon for policy')
parser.add_argument('--epsilon_last', type=float,
                    default=Config.epsilon[1], help='epsilon')
parser.add_argument('--updates', type=float,
                    default=Config.updates, help='Gap for target')
parser.add_argument('--pool_cap', type=float,
                    default=Config.pool_cap, help='Capacity of memory pool')
parser.add_argument('--packed', type=float,
                    default=Config.packed, help='Number of channels of input')
parser.add_argument('--batch_size', type=int,
                    default=Config.batch_size, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=Config.lr, help='init learning rate')
parser.add_argument('--seed', type=float,
                    default=Config.seed, help='seed')
parser.add_argument('--device', type=str,
                    default=Config.device,help="CUDA oder CPU")
parser.add_argument('--save_dir', type=str,
                    default=Config.save_path,help="save exp folder name")
args = parser.parse_args()

utilis.save_params(args)
utilis.ensure_repo(args.seed)
logger = utilis.create_logger(os.path.join("exp", args.exp_name))

#################################################
logger.info('------Begin Training------')

env = gym.make('ALE/Breakout-v5',render_mode='human')
agent = DQNAgent(env,args.discount,
                 Config.epsilon, args.learning_rate, args.packed, args.updates)
memory_pool = replay_memory(args.pool_cap)

utilis.ensure_repo(args.seed)

episodes_return = []

for j in range(10):
    for i in range(int(Config.episodes/10)):
        state=memory_pool.preprocessor(env.reset(seed=args.seed)[0])
        done = False
        G = 0
        C = 0
        agent.episode += 1
        while not done:
            C +=1
            # interact
            action = agent.take_action(state.repeat(args.packed,1,1).unsqueeze(dim=0))
            next_state, reward, done, _, _ = env.step(action)
            memory_pool.add(state, action, reward, next_state, done)

            state = memory_pool.preprocessor(next_state)
            G += reward

            # update
            if len(memory_pool) > args.threshold:
                memories = memory_pool.sample(args.batch_size)
                agent.update(memories)

        episodes_return.append(G)
        logger.info('episode:  {}, episodes_record:  {}, episodes_return:  {}'.format(agent.episode, C, G))
    utilis.save_checkpoint(args, agent, memory_pool)

logger.info('------End Training------')
##############################################

plt.figure(1)
plt.plot(range(int(Config.episodes)),episodes_return)
plt.xlabel("episodes")
plt.ylabel("scores")
plt.savefig(os.path.join("exp", args.exp_name, "reward.png"))
plt.show(block=True)

