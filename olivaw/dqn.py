import copy
import random
from datetime import datetime
import os
import statistics as st

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from olivaw import policy, networks, factory, utils, replay, reg


def train_dqn(args):
    # Initializing all necessary objects
    behavior_policy = policy.DeepEpsilonGreedy(
        args.base_epsilon, args.min_epsilon, args.epsilon_decay
    )
    loss_fn = nn.MSELoss(reduction="none")#F.smooth_l1_loss

    if args.dueling_dqn:
        dqn = networks.AtariDuelingDQN(args.action_size, args.state_size[:2], args.nb_stacked_frames, args.device)
    else:
        dqn = networks.AtariDQN(args.action_size, args.state_size[:2], args.nb_stacked_frames, args.device)
    target_net = copy.deepcopy(dqn)
    dqn = dqn.to(args.device)
    target_net = target_net.to(args.device)
    opt = factory.get_optimizer(dqn.parameters(), args)

    stacked_frames = utils.StackedFrames(args.nb_stacked_frames)
    if args.prioritized_er:
        er = replay.PrioritizedExperienceReplay(args.memory_size, [84, 84], args.nb_stacked_frames)
    else:
        er = replay.ExperienceReplay(args.memory_size, [84, 84], args.nb_stacked_frames)
    er.prefill(stacked_frames, args.action_size, args.pretrain_length, args.env)

    all_rewards = []
    all_losses = []
    best_mean_reward = -float("inf")
    start_date = datetime.now()

    global_step = 0
    episode_index = 0

    action_names = args.env.unwrapped.get_action_meanings()

    print('Training starts now!')
    while True:
        state = args.env.reset()
        stacked_frames.on_new_episode(state)

        episode_reward = 0
        episode_loss = 0.
        action_distributions = [0 for _ in range(len(action_names))]

        nb_no_op = random.randint(0, args.no_op_max)
        for step_index in range(args.episode_max_step):
            state = stacked_frames.get()

            # ----------------------------
            # Behavior / Exploration phase
            # ----------------------------
            for _ in range (args.update_frequency):
                if args.no_op_max > 0:
                    action = 0
                else:
                    action = behavior_policy.sample(dqn, er.trsfs(state), args.env)
                behavior_policy.update()

                action_distributions[action] += 1
                next_state, reward, done, _ = args.env.step(action)
                if args.clip_rewards:
                    reward = max(-1., min(1., reward))
                episode_reward += reward

                if done:
                    next_state = np.zeros((210, 160, 3))  # to fix

                stacked_frames.on_new_step(next_state)

                er.add((state, action, reward, stacked_frames.get(), done))

                global_step += 1

            # ---------------------------
            # Target / Exploitation phase
            # ---------------------------
            opt.zero_grad()

            batch = er.sample(args.batch_size)

            batch_actions = batch["action"].to(args.device)
            batch_state = batch["state"].to(args.device)
            batch_next_state = batch["next_state"].to(args.device)
            batch_reward = batch["reward"].to(args.device)
            batch_done = batch["done"].to(args.device)

            with torch.no_grad():
                predicted_next_q = target_net(batch_next_state)["qvalues"]
                if args.double_dqn:
                    predicted_next_q_main_net = dqn(batch_next_state)["qvalues"]
            outputs = dqn(batch_state)
            predicted_q, emb = outputs["qvalues"], outputs["emb"]

            if args.double_dqn:
                next_q = predicted_next_q[torch.arange(args.batch_size), predicted_next_q_main_net.max(dim=1)[1]]
            else:
                next_q = predicted_next_q.max(dim=1)[0]
            batch_G = batch_reward + args.gamma * (1 - batch_done) * next_q
            batch_G = batch_G.to(args.device)

            loss = loss_fn(predicted_q[torch.arange(args.batch_size), batch_actions], batch_G)
            if args.prioritized_er:
                er.update_transition(loss.cpu().detach().numpy())
                loss = batch["is_weights"].to(loss.device) * loss
            loss = loss.mean()

            if args.srank_reg > 0.:
                for e in emb:
                    loss += reg.srank_penalty(e, factor=args.srank_reg)
            loss.backward()
            opt.step()

            episode_loss += loss.item()

            if global_step % args.target_ckpt_frequency == 0:
                target_net.load_state_dict(dqn.state_dict())
            if global_step % args.test_frequency:
                test_dqn(dqn, er, args)

            if done or step_index > args.episode_max_step:
                break

        all_rewards.append(episode_reward)
        all_losses.append(episode_loss / (step_index + 1))

        if episode_index > 0 and episode_index % args.print_frequency == 0:
            mean_reward = st.mean(all_rewards[:-args.print_frequency])
            mean_loss = st.mean(all_losses[:args.print_frequency])

            print(f'{datetime.now() - start_date}: {global_step}/{args.nb_frames}, episode: {episode_index}, reward: {round(mean_reward,2)}, loss: {mean_loss}, epsilon: {round(behavior_policy.epsilon, 4)}')
            total_action_taken = sum(action_distributions)
            action_distributions = {
                action_names[a]: round(100 * nb / total_action_taken, 2)
                for a, nb in enumerate(action_distributions)
            }
            print(f'  Actions: {action_distributions}')

            np.save(os.path.join(args.log_dir, "losses.npy"), all_losses)
            np.save(os.path.join(args.log_dir, "rewards.npy"), all_rewards)

            if mean_reward > best_mean_reward:
                print("Updated best reward, saving model")
                best_mean_reward = mean_reward
                torch.save(dqn.state_dict(), os.path.join(args.log_dir, "net.pth"))
        episode_index += 1

        if global_step >= args.nb_frames:
            break
    print(f"Finished in {datetime.now() - start_date}")



def test_dqn(dqn, er, args):
    dqn.eval()
    print('Testing DQN...', end=' ')
    mean_reward = 0
    stacked_frames = utils.StackedFrames(args.nb_stacked_frames)
    behavior_policy = policy.DeepEpsilonGreedy(
        args.test_epsilon, args.test_epsilon, 0.
    )

    for _ in range(args.nb_test_episode):
        state = args.env.reset()
        stacked_frames.on_new_episode(state)

        nb_no_op = random.randint(0, args.no_op_max)
        for step_index in range(args.episode_max_step):
            state = stacked_frames.get()

            for _ in range (args.update_frequency):
                if args.no_op_max > 0:
                    action = 0
                else:
                    action = behavior_policy.sample(dqn, er.trsfs(state), args.env)

                next_state, reward, done, _ = args.env.step(action)
                if args.clip_rewards:
                    reward = max(-1., min(1., reward))
                mean_reward += reward
                stacked_frames.on_new_step(next_state)

                if done:
                    break

    print(f'Reward: {int(mean_reward / args.nb_test_episode)}')
    dqn.train()
