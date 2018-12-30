#!/usr/bin/env python

import os
import pickle
import pandas as pd
import gym

from imitation_learning import BehavioralCloning



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of expert roll outs')
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
    bc = BehavioralCloning(env.observation_space.shape[0], env.action_space.shape[0], n_hidden_units=n_hidden_units)

    # training
    bc.train(observations_expert, actions_expert)

    # predicting
    returns_predict = bc.evaluate(env, max_steps=max_steps)

    # writing results
    results_df = pd.DataFrame({
        'expert_returns': returns_expert,
        'bc_returns': returns_predict
    })

    results_df.to_csv(
        os.path.join('results/BC', args.envname+'_{0:d}.csv'.format(args.n_epochs))
    )


if __name__ == '__main__':
    main()
