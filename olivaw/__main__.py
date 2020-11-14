import argparse
import datetime
import os

import gym
import torch
import yaml

from olivaw.dqn import train_dqn


def parse_args():
    parser = argparse.ArgumentParser('Olivaw Reinforcement Library, at your service.')

    # Environment
    parser.add_argument('--env', default='Pong-v0')
    parser.add_argument('--nb_stacked_frames', default=4, type=int)
    parser.add_argument('--update_frequency', default=1, type=int)
    parser.add_argument('--no_op_max', default=30, type=int)

    # DQN
    parser.add_argument('--double_dqn', default=False, action='store_true')
    parser.add_argument('--dueling_dqn', default=False, action='store_true')
    parser.add_argument('--srank_reg', default=0., type=float)
    # noisy dqn

    # Training
    parser.add_argument('--optimizer', default='rmsprop', choices=['rmsprop', 'adam', 'sgd'])
    parser.add_argument('--lr', default=0.00025, type=float)
    parser.add_argument('--grad_momentum', default=0.95, type=float)
    parser.add_argument('--sqr_grad_momentum', default=0.95, type=float)
    parser.add_argument('--min_sqr_grad_momentum', default=0.01, type=float)

    parser.add_argument('--nb_frames', default=10_000_000, type=int)
    parser.add_argument('--episode_max_step', default=5_000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--target_ckpt_frequency', default=10_000, type=int)

    parser.add_argument('--base_epsilon', default=1.0, type=float)
    parser.add_argument('--min_epsilon', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.00001, type=float)

    parser.add_argument('--gamma', default=0.99, type=float)

    parser.add_argument('--clip_rewards', default=True, action='store_true')

    # Testing
    parser.add_argument('--nb_test_episode', default=10, type=int)
    parser.add_argument('--test_epsilon', default=0.001, type=float)
    parser.add_argument('--test_frequency', default=500_000, type=int)

    # Experience Replay
    parser.add_argument('--pretrain_length', default=50_000, type=int)
    parser.add_argument('--memory_size', default=1_000_000, type=int)
    parser.add_argument('--prioritized_er', default=False, action='store_true')

    # Misc
    parser.add_argument('--print_frequency', default=50, type=int)
    parser.add_argument('--device', default=-1, type=int)
    parser.add_argument('--name', required=True)
    parser.add_argument('--save_dir', default='/local/douillard/olivaw_experiments')
    parser.add_argument('--options', default=[], nargs='*')

    return parser.parse_args()


def set_arguments(args):
    args.env_str = args.env
    args.env = gym.make(args.env)
    args.action_size = args.env.action_space.n

    args.state_size = [84, 84, args.nb_stacked_frames]

    args.device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')


def load_options(args):
    dict_args = vars(args)
    for path in args.options:
        with open(path, 'r') as f:
            dict_args.update(yaml.load(f))


def dump_metadata(args):
    args.log_dir = os.path.join(
        args.save_dir,
        f'{datetime.datetime.now().strftime("%Y-%m-%d-%H")}_{args.name}'
    )
    os.makedirs(args.log_dir, exist_ok=True)

    with open(os.path.join(args.log_dir, "config.yaml"), "w+") as f:
        yaml.dump(vars(args), f)


if __name__ == '__main__':
    args = parse_args()
    load_options(args)
    dump_metadata(args)
    os.system(f"echo '\ek{args.name} on {args.device}\e\\'")
    set_arguments(args)

    train_dqn(args)

