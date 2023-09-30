import alog
from exchange_data.models.study_wrapper import StudyWrapper
from .agent import Agent
from .convnext import model


class Tuner(StudyWrapper):
    current_lock_ix = 0
    hparams = None
    run_count = 0

    def __init__(self, env_name, env_kwargs, **kwargs):
        self._kwargs = kwargs.copy()

        super().__init__(**kwargs)

        StudyWrapper.__init__(self, **kwargs)

        self.agent = Agent(env_name, env_kwargs, model_fn=model, **kwargs)

        self.study.optimize(self.agent.train, n_trials=1000)
