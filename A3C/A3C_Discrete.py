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
parser.add_argument('--offline', '-o', action='store_true')

env_kwargs = dict(
    base_filter_size=64,
    cache=True,
    database_name='binance_futures',
    depth=24,
    flat_class_str='NoRewardFlatTrade',
    group_by='1m',
    interval='6h',
    leverage=2,
    max_negative_pnl=-0.99,
    min_change=-0.04,
    min_flat_position_length=0,
    min_position_length=0,
    offset_interval='1h',
    random_frame_start=True,
    round_decimals=3,
    sequence_length=48,
    short_class_str='ShortTrade',
    summary_interval=8,
    symbol='UNFIUSDT',
    test_interval='1h',
    test_offset_interval='0h',
    window_size='4m',
    kernel=1,
    block_kernel=2,
    cache_len=100,
    num_conv=16
)


def main():
    env_name = 'orderbook-frame-env-v0'
    args = parser.parse_args()

    Tuner(env_name, env_kwargs, **args.__dict__, **env_kwargs)


if __name__ == "__main__":
    main()
