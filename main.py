import matplotlib.pyplot as plt
import gym

import utilis
from DQN import DQNAgent, replay_memory
import Config

import argparse
import os

parser = argparse.ArgumentParser("Deep_Q_learning")
parser.add_argument('--frames', type=int,
                    default=Config.frames, help='Numbers of episodes')
parser.add_argument('--discount', type=float,
                    default=Config.discount_factor, help='Gamma in formula')
parser.add_argument('--threshold', type=int,
                    default=Config.threshold, help='threshold for starting update')
parser.add_argument('--epsilon_init', type=float,
                    default=Config.epsilon[0], help='initial epsilon for policy')
parser.add_argument('--epsilon_last', type=float,
                    default=Config.epsilon[1], help='epsilon')
parser.add_argument('--updates', type=int,
                    default=Config.updates, help='Gap for target')
parser.add_argument('--exploration_stop', type=int,
                    default=Config.exploration_stop, help='init learning rate')

parser.add_argument('--buffer_size', type=int,
                    default=Config.buffer_size, help='Capacity of memory pool')
parser.add_argument('--input_channel', type=float,
                    default=Config.input_channel, help='Number of channels of input')
parser.add_argument('--batch_size', type=int,
                    default=Config.batch_size, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=Config.lr, help='init learning rate')

parser.add_argument('--interval', type=int,
                    default=Config.interval, help='init learning rate')
parser.add_argument('--seed', type=int,
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

env = gym.make('ALE/Breakout-v5', render_mode='human')
agent = DQNAgent(env, args.discount,
                 [args.epsilon_init,args.epsilon_last], args.exploration_stop,
                 args.input_channel,args.learning_rate, args.updates, args.device)
memory_pool = replay_memory(args.buffer_size,args.input_channel)

return_list = []
episode_count = 0

while agent.frame_count < args.frames:
    state = memory_pool.preprocessor(env.reset(seed=args.seed)[0])
    done = False
    episodes_return = 0
    episodes_frame = 0
    episode_count +=1
    while not done:
        episodes_frame +=1
        # interact
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        memory_pool.add(state, action, reward, next_state, done)

        state = memory_pool.preprocessor(next_state)
        episodes_return += reward
        agent.frame_count += 1

        # update
        if len(memory_pool) > args.threshold:
            memories = memory_pool.sample(args.batch_size)
            agent.update(memories)

    return_list.append(episodes_return)
    logger.info('process:  {:.2f}%, episode:  {}, episodes_record:  {}, episodes_return:  {}'
                .format((agent.frame_count/args.frames)*100, episode_count, episodes_frame, episodes_return))
    if agent.frame_count % args.interval == 0:
        utilis.save_checkpoint(args, agent, memory_pool)

logger.info('------End Training------')
##############################################

utilis.save_checkpoint(args, agent, memory_pool)
plt.figure(1)
plt.plot(range(episode_count), return_list)
plt.xlabel("episodes")
plt.ylabel("scores")
plt.savefig(os.path.join("exp", args.exp_name, "reward.png"))
plt.show(block=True)

