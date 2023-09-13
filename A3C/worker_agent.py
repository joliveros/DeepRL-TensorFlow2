from collections import deque
from threading import Thread, Lock

import alog
import numpy as np
import optuna
import wandb
import gym
from optuna import TrialPruned

from A3C.actor import Actor
from A3C.critic import Critic

CUR_EPISODE = 0

class WorkerAgent(Thread):
    def __init__(self, global_actor, global_critic,
                 max_episodes,
                 gamma, update_interval, batch_size,
                 cache_len=1000,
                 env_name=None,
                 env_kwargs=None, trial=None, **kwargs):

        Thread.__init__(self)

        global CUR_EPISODE
        CUR_EPISODE=0

        self.env_state = dict()
        self.trial = trial
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.n_steps = 0
        self.lock = Lock()
        self.env = gym.make(env_name, **env_kwargs)

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim, **kwargs)
        self.critic = Critic(self.state_dim, **kwargs)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

        self.cache = deque(maxlen=cache_len)

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = []
        for elem in list:
            batch.append(elem[0])

        return np.asarray(batch)

    def train(self, *args, **kwargs):
        global CUR_EPISODE
        capital = None

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                if state is None:
                    state = np.zeros(self.state_dim)

                probs = self.actor.model.predict(np.asarray([state]))

                action = np.argmax(probs[0])

                # action = np.random.choice(self.action_dim, p=probs[0])
                # alog.info((action, probs))

                next_state, reward, done, _ = self.env.step(action)

                self.env_state = _

                state = np.asarray([state])
                action = np.reshape(action, [1, 1])
                next_state = np.asarray([next_state])
                reward = np.reshape(reward, [1, 1])

                self.cache.append([state, action, reward])

                if done:
                    wandb.log({'capital': self.env_state['capital']})

                if self.n_steps % self.update_interval == 0 or done:
                    states = []
                    actions = []
                    rewards = []
                    cache_len = len(self.cache)
                    for i in range(0, self.batch_size):
                        ix = np.random.randint(0, cache_len)
                        state, action , reward = self.cache[ix]
                        states.append(state[0])
                        actions.append(action[0])
                        rewards.append(reward[0])

                    states = np.asarray(states)
                    actions = np.asarray(actions)
                    rewards = np.asarray(rewards)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)

                    with self.lock:
                        actor_loss = self.global_actor.train(
                            states, actions, advantages)
                        critic_loss = self.global_critic.train(
                            states, td_targets)

                        self.actor.model.set_weights(
                            self.global_actor.model.get_weights())
                        self.critic.model.set_weights(
                            self.global_critic.model.get_weights())

                    wandb.log(dict(actor_loss=actor_loss.numpy(),
                                   critic_loss=critic_loss.numpy()))

                    self.n_steps += 1

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            wandb.log({'Reward': episode_reward})
            CUR_EPISODE += 1

            if self.trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish(quiet=True)
                raise TrialPruned()

        wandb.run.summary["final capital"] = self.env_state['capital']
        wandb.run.summary["state"] = "completed"
        wandb.finish(quiet=True)

    def run(self):
        return self.train()
