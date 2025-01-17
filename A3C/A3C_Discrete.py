#!/usr/bin/env python

from A3C.tuner import Tuner
import alog
import argparse
import sec
import tensorflow as tf
import wandb

WANDB_API_KEY = sec.load('WANDB_API_KEY', lowercase=False)
wandb.login(key=WANDB_API_KEY)

tf.compat.v1.reset_default_graph()
tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--max_episodes', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--action_repetition', type=int, default=3)
parser.add_argument('--actor_lr', type=float, default=0.00005)
parser.add_argument('--critic_lr', type=float, default=0.0005)
parser.add_argument('--offline', '-o', action='store_true')

env_kwargs = dict(
    frame_width=219,
    base_filter_size=24,
    cache=False,
    database_name='binance_futures',
    # flat_class_str='FlatTrade',
    # flat_class_str='NoRewardFlatTrade',
    flat_class_str='FlatRewardPnlDiffTrade',
    group_by='1m',
    interval='12h',
    leverage=2,
    max_negative_pnl=-0.99,
    max_loss=-0.005,
    min_change=-0.99,
    min_flat_position_length=0,
    min_position_length=0,
    offset_interVAL='0h',
    random_frame_start=False,
    round_decimals=3,
    depth=20,
    sequence_length=35,
    short_class_str='ShortRewardPnlDiffTrade',
    # short_class_str='ShortTrade',
    summary_interval=20,
    symbol='UNFIUSDT',
    test_interval='1h',
    test_offset_interval='0h',
    window_size='30m',
    cache_len=216,
)


def main():
    env_name = 'orderbook-frame-env-v0'
    args = parser.parse_args()

    Tuner(env_name, env_kwargs, **args.__dict__, **env_kwargs)


if __name__ == "__main__":
    main()
