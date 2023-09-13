from statistics import mean

from optuna import Trial

from A3C.actor import Actor
from A3C.critic import Critic
from A3C.worker_agent import WorkerAgent
import wandb
import alog
import gym
import tgym.envs


class Agent:
    def __init__(self, env_name, env_kwargs, num_workers,
                 max_episodes, offline, **kwargs):
        self.offline = offline
        self.global_critic = None
        self.global_actor = None
        self.max_episodes = max_episodes
        self.trial = None
        self.kwargs = kwargs
        self.env_kwargs = env_kwargs
        env = gym.make(env_name, **env_kwargs)
        self.env_name = env_name
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        self.num_workers = num_workers

        env.close()

    def train(self, trial: Trial, *args, **kwargs):
        self.trial = trial

        hparams = dict(
            block_kernel=trial.suggest_int('block_kernel', 1, 4),
            kernel_size=trial.suggest_int('kernel_size', 1, 4),
            actor_lr=trial.suggest_float('actor_lr', 0.0000001, 0.001),
            critic_lr=trial.suggest_float('critic_lr', 0.0000001, 0.001),
            cache_len=trial.suggest_int('cache_len', 10, 1000)
        )

        for key in hparams.keys():
            self.kwargs[key] = hparams[key]

        config = dict(trial.params)
        config["trial.number"] = trial.number

        wandb.init(
            name='A3C',
            project="deep-rl-tf2",
            config=config,
            mode='offline' if self.offline else 'online'
        )

        self.global_actor = Actor(self.state_dim, self.action_dim,
                                  **self.kwargs)
        self.global_critic = Critic(self.state_dim, **self.kwargs)

        workers = []

        for i in range(self.num_workers):
            workers.append(WorkerAgent(
                self.global_actor,
                self.global_critic,
                self.max_episodes,
                trial=trial,
                env_name=self.env_name,
                env_kwargs=self.env_kwargs,
                **self.kwargs))

        for worker in workers:
            worker.start()

        capital = []
        trades = []

        for worker in workers:
            worker.join()
            capital.append(worker.env_state['capital'])
            trades.append([trade for trade in
                           worker.env_state['trades'] if trade.pnl > 0])

        wandb.finish()

        if len(trades) > 0:
            trade_value = len(trades) ** (1/24)
        else:
            trade_value = 0

        return mean(capital) + trade_value
