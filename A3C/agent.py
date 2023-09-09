from A3C.actor import Actor
from A3C.critic import Critic
from A3C.worker_agent import WorkerAgent

import alog
import gym
import tgym.envs


class Agent:
    def __init__(self, env_name, env_kwargs, num_workers, **kwargs):
        self.kwargs = kwargs
        self.env_kwargs = env_kwargs
        env = gym.make(env_name, **env_kwargs)
        self.env_name = env_name

        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n

        self.global_actor = Actor(self.state_dim, self.action_dim, **kwargs)
        self.global_critic = Critic(self.state_dim, **kwargs)
        self.num_workers = num_workers

    def train(self, max_episodes=10**10):
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name, **self.env_kwargs)
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes, **self.kwargs))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()
