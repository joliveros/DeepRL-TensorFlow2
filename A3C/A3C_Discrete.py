#!/usr/bin/env python

from A3C.tuner import Tuner
import alog
import tensorflow as tf
import wandb
import sec
import argparse

WANDB_API_KEY = sec.load('WANDB_API_KEY', lowercase=False)
wandb.login(key=WANDB_API_KEY)

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--max_episodes', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--actor_lr', type=float, default=0.00005)
parser.add_argument('--critic_lr', type=float, default=0.0005)

env_kwargs = dict(
    database_name='binance_futures',
    depth=24,
    sequence_length=48,
    interval='6h',
    symbol='UNFIUSDT',
    window_size='4m',
    group_by='1m',
    cache=True,
    leverage=2,
    offset_interval='0h',
    max_negative_pnl=-0.99,
    summary_interval=8,
    round_decimals=3,
    min_position_length=0,
    min_flat_position_length=0,
    min_change=-0.01,
    short_class_str='ShortRewardPnlDiffTrade',
    flat_class_str='NoRewardFlatTrade',
    random_frame_start=True,
)


def main():
    env_name = 'orderbook-frame-env-v0'
    args = parser.parse_args()

    Tuner(env_name, env_kwargs, **args.__dict__, **env_kwargs)


if __name__ == "__main__":
    main()
