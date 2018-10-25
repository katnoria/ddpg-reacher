import os
from datetime import datetime
import logging
from collections import deque
import argparse
import numpy as np
import torch
from ddpg import DDPGAgent
from unityagents import UnityEnvironment
from utils import save_to_json, save_to_txt
from utils import VisWriter

# Create logger
logger = logging.getLogger('ddpg')
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
dirname = 'runs/{}'.format(now)
if not os.path.exists(dirname):
    os.makedirs(dirname)

filehandler = logging.FileHandler(filename='{}/trainer.log'.format(dirname))
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
# Uncomment to enable console logger
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.INFO)
logger.addHandler(steamhandler)

def ddpg(env, agent, brain_name, num_agents, writer, n_episodes=300, max_t=1000, print_every=50):
    best_score = -np.inf
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        for t in range(max_t):
            agent.reset()
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)            
            dones = env_info.local_done                        # see if episode finished
            score += env_info.rewards                          # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0])
            if np.any(dones):       
                logger.debug('Episode:{} finishe in {} steps'.format(i_episode, t))
                break

        scores_deque.append(score)
        scores.append(score)
        current_score = np.mean(scores_deque)
        writer.text('Episode {}/{}: Average score(100): {}'.format(i_episode, n_episodes, current_score), "Average 100 episodes")
        writer.push(np.mean(scores_deque), "Average Score")
        logger.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, current_score))
        
        if current_score >= best_score:
            logger.info('Best score found, old: {}, new: {}'.format(best_score, current_score))
            best_score = current_score
            agent.checkpoint()

#         torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
#         torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            logger.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    logger.info('Final Average Score: {:.2f}'.format(np.mean(scores_deque)))
    return scores

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=1000, help="Total number of episodes to train")
    parser.add_argument("--max_t", type=int, default=1000, help="Max timestep in a single episode")
    parser.add_argument("--vis", type=bool, default=True, help="Whether to use visdom to visualise training")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--info", type=str, default="", help="Use this to attach notes to your runs")

    args = parser.parse_args()
    print(args)

    # visualiser
    writer = VisWriter(vis=args.vis)

    # save info
    open('{}/info.txt'.format(dirname), 'w').write(args.info)

    env = UnityEnvironment(file_name='./../data/Reacher_Linux_NoVis1/Reacher.x86_64')    
    # brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    state = env_info.vector_observations
    state_shape = state.shape[1]
    action_size = brain.vector_action_space_size    

    agent = DDPGAgent(state_shape, action_size, writer=writer, dirname=dirname, random_seed=42, print_every=100, model_path=args.model)
    scores = ddpg(env, agent, brain_name, num_agents, writer=writer, n_episodes=args.num_episodes, max_t=args.max_t)
    # save all scores
    save_to_txt('\n'.join(scores), '{}/scores_full.txt'.format(dirname))

if __name__ == "__main__":    
    main()