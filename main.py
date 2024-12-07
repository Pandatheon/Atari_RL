import gym

import utilis
from DQN import DQNAgent, replay_memory
from arguments import args

import os
from torch.utils.tensorboard import SummaryWriter


utilis.save_params(args)
utilis.ensure_repo(args.seed)
logger = utilis.create_logger(os.path.join("exp", args.exp_name))
writer = SummaryWriter()

#################################################
logger.info('------Begin Training------')

env = gym.make('ALE/Breakout-v5')
agent = DQNAgent(env, args.discount,
                 [args.epsilon_init,args.epsilon_last], args.exploration_stop, args.observance_stop,
                 args.input_channel,args.learning_rate, args.target_update, args.device)
memory_pool = replay_memory(args.buffer_size,args.input_channel)

return_list = []
episode_count = 0
update_count = 0
while agent.frame_count < args.frames:
    state = memory_pool.preprocessor(env.reset(seed=args.seed)[0])
    done = False
    action_count = [0, 0, 0, 0] # ad-hoc
    episode_return = 0
    episode_frame = 0
    episode_count +=1
    while not done:
        episode_frame +=1
        # interact
        action = agent.take_action(state)
        next_state, reward, done, _, _ = env.step(action)
        memory_pool.add(state, action, reward, next_state, done)

        action_count[action] += 1  # ad-hoc
        state = memory_pool.preprocessor(next_state)
        episode_return += reward
        agent.frame_count += 1

        # update
        if agent.frame_count > args.observance_stop:
            if update_count % args.online_update == 0:
                memories = memory_pool.sample(args.batch_size)
                agent.update(memories)
            update_count+=1

        if agent.frame_count % args.interval == 0:
            utilis.save_checkpoint(args, agent)

    return_list.append(episode_return)
    logger.info('process:  {:.2f}%, episode:  {}, episodes_record:  {}, episodes_return:  {},\n'
                '\t \t \tactions: no_op:{}, fire:{}, right:{}, left:{}'  # very ad-hoc
                .format((agent.frame_count/args.frames)*100, episode_count, episode_frame, episode_return,
                action_count[0],action_count[1],action_count[2],action_count[3]))
    writer.add_scalar('return/episodes', episode_return, episode_count)

logger.info('------End Training------')
##############################################
utilis.save_checkpoint(args, agent)