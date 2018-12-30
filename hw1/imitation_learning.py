import tensorflow as tf
import numpy as np
import sys

class BehavioralCloning(object):
    def __init__(self, obs_dim, action_dim, n_hidden_units=None, lr=0.01):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_hidden_units = n_hidden_units
        self.lr = lr

        self.build_net()
        self.init_sess()

    def init_sess(self):
        self.sess = tf.Session()
        self.sess.__enter__()
        tf.global_variables_initializer().run()

    def build_net(self):
        # input
        self.observations = tf.placeholder(tf.float32, [None, self.obs_dim], name='observations')
        self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')

        # define parameters and  computations
        with tf.variable_scope('layer1'):
            self.w1 = tf.get_variable('w', [self.obs_dim, self.n_hidden_units],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable('b', [self.n_hidden_units],
                                      initializer=tf.constant_initializer(0.0))
            self.output_hidden = tf.nn.relu(tf.matmul(self.observations, self.w1) + self.b1)

        with tf.variable_scope('layer2'):
            self.w2 = tf.get_variable('w', [self.n_hidden_units, self.action_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
            self.b2 = tf.get_variable('b', [self.action_dim],
                                      initializer=tf.constant_initializer(0.0))
            self.actions_pred = tf.matmul(self.output_hidden, self.w2) + self.b2


        self.loss = tf.reduce_mean(tf.squared_difference(self.actions_pred, self.actions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def batch_fit(self, batch_obs, batch_actions):
        loss_value, _ = self.sess.run([self.loss, self.optimizer],
                                 feed_dict={self.observations: batch_obs,
                                            self.actions: batch_actions})
        return loss_value

    def predict(self, batch_obs):
        return self.sess.run(self.actions_pred,
                             feed_dict={self.observations: batch_obs})

    def train(self, observations, actions, n_epochs=10, batch_size=32, shuffle=True):
        """upsupervised learning using expert data
        """
        n_samples = observations.shape[0]
        n_batches = n_samples // batch_size
        sample_idx = np.arange(n_samples)

        for i in np.arange(n_epochs):
            if shuffle:
                # shuffle the samples
                np.random.shuffle(sample_idx)
                observations = observations[sample_idx]
                actions = actions[sample_idx]

            for j in np.arange(n_batches):
                start_idx = j * batch_size
                end_idx = (j + 1) * batch_size
                batch_obs = observations[start_idx: end_idx]
                batch_actions = actions[start_idx: end_idx]
                loss = self.batch_fit(batch_obs, np.squeeze(batch_actions))

            print("-----------Epoch {0:d} loss: {1:.6f}--------".format(i, loss))

    def evaluate(self, env, n_episodes=20, max_steps=1000):
        """evaluate policy using current environment
        """
        returns = []
        for i in np.arange(n_episodes):
            sys.stdout.flush()
            print('\riter {0:d}/{1:d}'.format(i+1, n_episodes), end="")

            obs = env.reset()
            done = False
            totalr = 0
            steps = 0

            while not done and steps < max_steps:
                action = self.predict(obs.reshape(1, -1))
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1

            returns.append(totalr)

        return returns


class DAgger(BehavioralCloning):
    def __init__(self, obs_dim, action_dim, n_hidden_units=None, lr=0.01):
        super(DAgger, self).__init__(obs_dim, action_dim, n_hidden_units=n_hidden_units, lr=lr)

    def run_policy(self, env, max_steps=1000):
        """
        :param env: environment
        :return: S: colleciong ot states following policy pi
        """
        obs = env.reset()
        done = False
        S = [obs]
        step = 0
        while not done and step < max_steps:
            action = self.predict(obs.reshape(1,-1))
            obs, _, _, _ = env.step(action)
            S.append(obs)
            step += 1
        return np.array(S)

    def label_actions(self, S, expert_policy_fn):
        """
        ask expert policy to label action for each observation
        :param expert_policy_fn: expert policy
        :return: A: a collection of labelled actions
        """
        A = [expert_policy_fn(obs.reshape(1, -1)) for obs in S]
        return np.array(A)

    def run_policy_and_label_action(self, env, expert_policy_fn):
        """
        run policy to get dataset D = {A, S},
        where S are the collections of states following policy pi,
        and A are the expert-labeled actions
        """
        S = self.run_policy(env)
        A = self.label_actions(S, expert_policy_fn)
        return S, A


