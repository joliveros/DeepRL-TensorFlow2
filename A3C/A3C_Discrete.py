#!/usr/bin/env python
from collections import deque
from exchange_data.models.resnet.model import Model
from tensorflow.keras.layers import Input, Dense
import alog
import tensorflow as tf
import tgym.envs
import wandb

import gym
import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
tf.keras.backend.set_floatx('float64')

wandb.init(name='A3C', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--actor_lr', type=float, default=0.00005)
parser.add_argument('--critic_lr', type=float, default=0.0005)

args = parser.parse_args()

CUR_EPISODE = 0
env_kwargs = dict(
    database_name='binance_futures',
    depth=24,
    sequence_length=48,
    interval='6h',
    symbol='UNFIUSDT',
    window_size='4m',
    group_by='30s',
    cache=True,
    leverage=2,
    offset_interval='0h',
    max_negative_pnl=-0.99,
    summary_interval=8,
    round_decimals=3,
    min_position_length = 0,
    min_flat_position_length = 0,
    short_class_str = 'ShortTrade',
    flat_class_str ='FlatTrade'
)

class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        # split_gpu()
        # return tf.keras.Sequential([
        #     Input(self.state_dim),
        #     Dense(32, activation='relu'),
        #     Dense(16, activation='relu'),
        #     Dense(self.action_dim, activation='softmax')
        # ])
        alog.info(self.state_dim)

        model = Model(
            input_shape=self.state_dim,
        )

        print(model.summary())
        return model

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        # split_gpu()

        model = Model(
            input_shape=self.state_dim,
        )
        model = Model(
            input_shape=self.state_dim,
        )

        print(model.summary())

        dense = Dense(1, activation='linear')(model.output)

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[dense])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name):
        env = gym.make(env_name, **env_kwargs)
        self.env_name = env_name

        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = args.num_workers

    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name, **env_kwargs)
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.n_steps = 0
        self.lock = Lock()
        self.env = env

        # self.state_dim = self.env.observation_space.shape
        self.state_dim = env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

        self.cache = deque(maxlen=1000)

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = []
        for elem in list:
            batch.append(elem[0])

        return np.asarray(batch)

    def train(self):
        global CUR_EPISODE

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                probs = self.actor.model.predict(np.asarray([state]))

                action = np.random.choice(self.action_dim, p=probs[0])

                next_state, reward, done, _ = self.env.step(action)

                # state = np.reshape(state, [1, self.state_dim])
                state = np.asarray([state])
                action = np.reshape(action, [1, 1])
                next_state = np.asarray([next_state])
                # next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                self.cache.append([state, action, reward])

                if self.n_steps % args.update_interval == 0 or done:
                    states = []
                    actions = []
                    rewards = []
                    cache_len = len(self.cache)
                    for i in range(0, 2):
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

                    self.n_steps += 1

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            wandb.log({'Reward': episode_reward})
            CUR_EPISODE += 1

    def run(self):
        self.train()

def split_gpu(memory=2400):
        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0 and memory > 0:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [
                    tf.config.
                    LogicalDeviceConfiguration(memory_limit=memory),
                ])


def main():
    env_name = 'orderbook-frame-env-v0'
    agent = Agent(env_name)
    agent.train()


if __name__ == "__main__":
    main()
