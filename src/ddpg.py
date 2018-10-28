import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import os
import logging
import json
from datetime import datetime
# Pytorch
import torch
import torch.nn.functional as F
import torch.optim as optim
# Code
from models import Actor, Critic
from utils import OUNoise, ReplayBuffer
from utils import save_to_json, save_to_txt


# Hyper parameters
config = {
    "USE_BATCHNORM": False,
    "USE_XAVIER": False,
    "BUFFER_SIZE": int(1e5),    # replay buffer size
    "BATCH_SIZE": 256,         # minibatch size
    "GAMMA": 0.9,              # discount factor
    "TAU": 1e-3,                # for soft update of target parameters
    "LR_ACTOR": 1e-3,           # learning rate of the actor 
    "LR_CRITIC": 1e-3,          # learning rate of the critic
    "WEIGHT_DECAY": 0,        # L2 weight decay
    "SCALE_REWARD": 1.0,       # default scaling factor
    "UPDATE_TARGET_FREQ": 1,
    "SIGMA": 0.01,
    "FC1": 128,
    "FC2": 128,
}

# Create logger
logger = logging.getLogger('ddpg')

# map device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug('Device Info:{}'.format(device))


class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, writer, dirname, random_seed, print_every=100, model_path=None, eval_mode=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            writer (object): visdom visualiser for realtime visualisations            
            random_seed (int): random seed
            dirname (string): output directory to store config, losses
            print_every (int): how often to print progress
            model_path (string): if defined, load saved model to resume training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dirname = dirname
        self.seed = random.seed(random_seed)  
        self.print_every = print_every      

        # save config params
        save_to_json(config, '{}/hyperparams.json'.format(self.dirname))

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"], use_xavier=config["USE_XAVIER"]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"], use_xavier=config["USE_XAVIER"]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config["LR_ACTOR"])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"], use_xavier=config["USE_XAVIER"]).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"], use_xavier=config["USE_XAVIER"]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config["LR_CRITIC"], weight_decay=config["WEIGHT_DECAY"])

        # Load saved model (if available)
        if model_path:
            logger.info('Loading model from {}'.format(model_path))
            self.actor_local.load_state_dict(torch.load('{}/checkpoint_actor.pth'.format(model_path)))
            self.actor_target.load_state_dict(torch.load('{}/checkpoint_actor.pth'.format(model_path)))
            self.critic_local.load_state_dict(torch.load('{}/checkpoint_critic.pth'.format(model_path)))
            self.critic_target.load_state_dict(torch.load('{}/checkpoint_critic.pth'.format(model_path)))
            if eval_mode:
                logger.info('Agent set to eval mode')
                self.actor_local.eval()


        # Noise process
        self.noise = OUNoise(action_size, random_seed, sigma=config['SIGMA'])

        # Replay memory
        self.memory = ReplayBuffer(action_size, config["BUFFER_SIZE"], config["BATCH_SIZE"], random_seed)

        # Record losses
        self.actor_losses = []
        self.critic_losses = []
        self.learn_count = []
        self.step_count = 0
        self.learn_step = 0
        self.writer = writer
        logger.info("Initialised with random seed: {}".format(random_seed))
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # self.memory.add(state, action, reward, next_state, done)        
        # scaled_rewards = [config['SCALE_REWARD'] * r for r in reward]
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > config["BATCH_SIZE"]:
            experiences = self.memory.sample()
            self.learn(experiences, config["GAMMA"])

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Resets the noise"""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        self.learn_step += 1

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        if self.learn_step % self.print_every == 0:
            save_to_txt(critic_loss, '{}/critic_loss.txt'.format(self.dirname))
            self.writer.text('critic loss: {}'.format(critic_loss), "Critic")
            self.writer.push(critic_loss.item(), "Loss(critic)")
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        if self.learn_step % self.print_every == 0:
            save_to_txt(actor_loss, '{}/actor_loss.txt'.format(self.dirname))
            self.writer.text('actor loss: {}'.format(actor_loss), "Actor")
            self.writer.push(actor_loss.item(), "Loss(actor)")

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step == 1:
            logger.info('Copying all parameters from local to target')
            self._copy_weights(self.critic_local, self.critic_target)
            self._copy_weights(self.actor_local, self.actor_target)
        else:
            if self.learn_step % config['UPDATE_TARGET_FREQ'] == 0:
                self.soft_update(self.critic_local, self.critic_target, config["TAU"])
                self.soft_update(self.actor_local, self.actor_target, config["TAU"])        

    def _copy_weights(self, source_network, target_network):
        """Copy source network weights to target"""        
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def checkpoint(self):
        """Checkpoint actor and critic models"""
        torch.save(self.actor_local.state_dict(), '{}/checkpoint_actor.pth'.format(self.dirname))
        torch.save(self.critic_local.state_dict(), '{}/checkpoint_critic.pth'.format(self.dirname))
        lr = {}
        lr['actor_lr'] = [param_group['lr'] for param_group in self.actor_optimizer.param_groups]        
        lr['critic_lr'] = [param_group['lr'] for param_group in self.critic_optimizer.param_groups]
        save_to_json(lr, '{}/best_lr.json'.format(self.dirname))