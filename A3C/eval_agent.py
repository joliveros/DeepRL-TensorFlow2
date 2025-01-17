from copy import copy
from dataclasses import dataclass

import alog
import gym
import numpy as np
import wandb
from optuna import TrialPruned, Trial


@dataclass
class EvalStats:
    capital: float
    pos_trades: int

    @property
    def trade_capital_ratio(self):
        return self.capital + (self.pos_trades ** 1 / 24) - 1

    @property
    def __dict__(self):
        return dict(capital=self.capital,
                    pos_trades=self.pos_trades,
                    trade_capital_ratio=self.trade_capital_ratio)


class EvalAgent:
    def __init__(
            self,
            trial,
            test_interval,
            env_name,
            env_kwargs,
            actor,
            action_repetition,
            **kwargs):

        self.trial: Trial = trial
        self._env_kwargs = None
        self.steps = 0
        self.actor = actor
        self.stats: EvalStats = None
        self.test_interval = test_interval
        self.action_repetition = action_repetition
        self.last_action = None

        self.env_kwargs = copy(env_kwargs)

        self.env = gym.make(
            env_name,
            is_test=True,
            worker_name='eval_agent',
            custom_summary_keys=['worker_name'],
            **env_kwargs
        )
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

    def set_env_kwargs(self, value):
        self._env_kwargs = self.set_offset_interval(value)

    def get_env_kwargs(self):
        if self._env_kwargs is None:
            raise Exception('env_kwargs is not set.')
        return self._env_kwargs

    def set_offset_interval(self, env_kwargs):
        train_env_offset_interval = env_kwargs['offset_interval']
        if train_env_offset_interval == '0h':
            raise Exception('should not be zero. Eval data should be offset.')

        env_kwargs['offset_interval'] = env_kwargs['test_offset_interval']
        env_kwargs['interval'] = self.test_interval

        return env_kwargs

    def reset(self):
        self.steps = 0
        return self.env.reset()

    def eval(self, step):
        state = self.env.reset()
        eval_done = False

        while not eval_done:
            if state is None:
                state = np.zeros(self.state_dim)

            if self.steps % 10 == 0:
                probs = self.actor.model.predict(np.asarray([state]))
                alog.info(probs)
                action = np.random.choice(self.action_dim, p=probs[0])
                self.last_action = action
            else:
                action = self.last_action

            # probs = self.actor.model.predict(np.asarray([state]))
            # action = np.argmax(probs[0])

            next_state, reward, done, _ = self.env.step(action)

            state = next_state

            self.env_state = _
            self.steps += 1

            if done:
                eval_done = done
                pos_trades = [t for t in self.env_state['trades'] if t.pnl > 0]
                self.stats = EvalStats(capital=self.env_state['capital'],
                                       pos_trades=len(pos_trades))

                wandb.log(self.stats.__dict__)

                self.trial.report(self.stats.trade_capital_ratio, step)

                if self.trial.should_prune():
                    wandb.run.summary["state"] = "pruned"
                    wandb.finish()
                    raise TrialPruned()
