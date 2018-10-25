[//]: # (Image References)

[image1]: images/single_agent_avg_score_sm.png "single agent score"
[image2]: images/multi_agent_avg_score_sm.png "single agent score"

# Reacher

The Reacher environment requires the agent to learn from high dimensional state space and perform actions in continuous action space. 

The algorithm such as DQN can solve problems with high dimensional state space but only work on discrete and low-dimensional action spaces. Where as the algorithm such as REINFORCE can learn the policy to map state into actions but they are sample inefficient, noisy because we are sampling a trajectory (or a few trajectories) which may not truly represent the policy and could prematurely converge to local optima. 

The solution used in this repo makes use of actor-critic approach proposed by Lillicrap et al in deep deterministic policy gradient [paper](https://arxiv.org/abs/1509.02971).

The actor network takes state as input and returns the action whereas critic network takes state and action as input and returns the value. 

The critic in this case is DQN with fixed target network and replay buffer and it is trained by sampling experiences from the replay buffer. The sampled experiences are used to calculate Q estimate and targets and network is trained to minimise the mean squared error between target and estimate.

The actor is trained by using actor to generate actions for a state and maximising the value returned by critic for that state, action pair.


# Network Architecture

We use fully connected layers for both Actor and Critic network in pytorch 0.4.

<!-- <img src='https://g.gravizo.com/svg?
 digraph G {     
    subgraph cluster_actor {
    linear1 -> batchnorm1 -> relu1;
    relu1 -> linear2  -> batchnorm2 -> relu2;
    relu2 -> linear3 -> tanh;
    label="Actor Network";
    color=blue;
     }
     linear1[label="linear"];
     batchnorm1[label="batchnorm1d"];
     relu1[label="relu"];
     linear2[label="linear"];
     batchnorm2[label="batchnorm1d"];
     relu2[label="relu"];
     linear3[label="linear"];
    subgraph cluster_critic {
    linearc1 -> batchnormc1 -> reluc1;
    reluc1 -> linearc2  -> batchnormc2 -> reluc2;
    action -> linearc2;
    reluc2 -> linearc3;
    label="Critic Network";
    color=green;
     }
     linearc1[label="linear"];
     batchnormc1[label="batchnorm1d"];
     reluc1[label="relu"];
     linearc2[label="linear"];
     batchnormc2[label="batchnorm1d"];
     reluc2[label="relu"];
     linearc3[label="linear"];
 }
'/> -->
![Network Architecture](images/architecture.png)

# Hyper parameters

I tried several hyperparameters in order to solve the environment along with the ones specified by DDPG paper. In the end, the following worked the best.

## Single Agent
|Parameter|Value|Description|
|---------|-----|-----------|
|BUFFER_SIZE|1e5|Replay memory buffer size|
|BATCH_SIZE|256|Minibatch size|
|GAMMA|0.9|Discount factor|
|TAU|1e-3|Soft update of target parameters|
|LR_ACTOR|1e-3|Actor learning rate|
|LR_CRITIC|1e-3|Critic learning rate|
|WEIGHT_DECAY|0|L2 weight decay|
|SCALE_REWARD|1.0|Reward scaling|
|SIGMA|0.01|OU Noise std|
|FC1|128|input channels for 1st hidden layer|
|FC2|128|input channels for 1st hidden layer|

The agent was trained for 1000 timesteps per episode.

# Performance

The single agent environment was solved in **176 episodes**.

![Single Agent][image1]

Watch the single player play

![Single Player](images/singleplayer.gif)

The multi-agent environment was solved in **100 episodes**.

![Multi Agent][image2]

Watch the multi player play

![Multi Player](images/multiplayer.gif)


# Future Work

Some people were able to solve the environment in relatively fewer number of training episodes. I think a combination of following could help me reduce the training steps:

* reduce network size either by remove one hidden layer or decreasing the number of units as this would result in a lot less parameters.
* increase the learning rate and introduce weight decay (currently set to 0) to speed up the learning
* experiment with scaled rewards (see below)

The [paper](https://arxiv.org/abs/1604.06778) found that DDPG is less stable than batch algorithms such as REINFORC and the performance of policy could degrade significantly during the training phase. In my tests, I found that average score hit a plateau if I continued the training process even after solving the environment. They found that scaling the rewards seem to improve the stability.

While working on DDPG solution, there were a lot of moving parts such as network architecture, hyperparameters and it took be a long time, along with suggestions on the forum, to discover the combination that could solve the environment. Proximal policy optimization (PPO) has shown to achieve [state-of-the-art](https://blog.openai.com/openai-baselines-ppo/) results with very little hyperparamter tuning, greater sample efficiency and keeping the policy deviation under check (by forcing the ratio of old and new policy with in a small interval). 