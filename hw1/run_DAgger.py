#!/usr/bin/env python

import os
import pickle
import numpy as np
import pandas as pd
import gym
import load_policy

from imitation_learning import DAgger

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--n_dagger_iter', type=int, default=20,
                        help='number of iteration for DAgger')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs in training')
    parser.add_argument('--n_rollouts', type=int, default=20,
                        help='Number of roll outs in evaluation')
    args = parser.parse_args()

    print('loading expert data')
    expert_data = pickle.load(open(os.path.join('expert_data', args.envname + '.pkl'), 'rb'))

    returns_expert = expert_data['returns']
    # training data
    observations_expert = expert_data['observations']
    actions_expert = expert_data['actions']

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    n_hidden_units = env.observation_space.shape[0]

    print("build dagger policy")
    dagger = DAgger(env.observation_space.shape[0], env.action_space.shape[0], n_hidden_units=n_hidden_units)

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    states = observations_expert
    actions = actions_expert

    returns_df = pd.DataFrame({
        'expert_returns': returns_expert
    })

    for i in np.arange(args.n_dagger_iter):
        print("DAgger iter {0:d}".format(i))

        # training from {S,A}
        print("start training policy...")
        dagger.train(states, actions, n_epochs=args.n_epochs)

        # run policy and ask expert policy to label actions
        print("running policy and expert labeling...")
        new_S, new_A = dagger.run_policy_and_label_action(env, policy_fn)

        # aggregate
        print("aggreating...")
        states = np.vstack((states, new_S))
        actions = np.vstack((actions, new_A))

        print("Evaluating DAgger policy...")
        returns_dagger = dagger.evaluate(env, n_episodes=args.n_rollouts, max_steps=max_steps)
        returns_df['dagger_returns_{0:d}'.format(i)] = returns_dagger
        print("average return: {0:.6f}".format(np.mean(returns_dagger)))


    print("writing results to csv....")
    returns_df.to_csv(
        os.path.join('results/DAgger', args.envname + '_{0:d}_{1:d}.csv'.format(args.n_epochs, args.n_dagger_iter))
    )

if __name__ == '__main__':
    main()
