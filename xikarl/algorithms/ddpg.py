# coding: utf-8

import numpy as np
import tensorflow as tf
from xikarl.algorithms.base import DefaultAlgorithm
from xikarl.algorithms.net import MLP, Tanh
from xikarl.utilities.env_wrap import EnvWrap
from xikarl.utilities.update import soft_update

tk = tf.keras


class DDPG(DefaultAlgorithm):

    def __init__(
            self,
            *args,
            policy_units=(32, 32),
            policy_lr=1e-3,
            critic_units=(32, 32),
            critic_lr=2e-3,
            tau=0.05,
            discount=0.99,
            device="/cpu:0",
            **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "ddpg"
        super().__init__(*args, **kwargs)

        self.discount = discount
        self.tau = tau
        self.device = "/cpu:0"

        # critic
        self.critic = self._construct_critic(critic_units)
        self.target_critic = self._construct_critic(critic_units, name="critic_target")
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=critic_lr)
        soft_update(self.critic.weights, self.target_critic.weights, 1)

        # policy
        self.policy = self._construct_policy(policy_units)
        self.target_policy = self._construct_policy(policy_units, name="policy_target")
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=policy_lr)
        soft_update(self.policy.weights, self.target_policy.weights, 1)

    def _construct_policy(self, policy_units, name="policy"):
        return MLP(
            policy_units,
            self.obs_size,
            self.act_size,
            scale=self.env.action_space.high,
            output_activation=Tanh,
            name=self.name+"/"+name,
            before_step_func=self.policy_before_step_func
        )

    @staticmethod
    def policy_before_step_func(inputs):
        return tf.concat(inputs, axis=1)

    def _construct_critic(self, critic_units, name="critic"):
        return MLP(
            critic_units,
            self.obs_size+self.act_size,
            1,
            name=self.name+"/"+name
        )

    def get_cation(self, obs, is_test=False):
        action = self.policy(obs.reshape(1, -1))
        action = tf.squeeze(action).numpy()
        if not is_test:
            noise = np.random.normal(0, 0.1, action.shape)
            action += noise
        if self.act_size == 1:
            action = np.expand_dims(action, axis=0)
        return action

    def train(self, batch):
        with tf.device(self.device):
            # update critic
            with tf.GradientTape() as tape:
                tderror = self.compute_tderror(batch)
                critic_loss = tf.reduce_mean(tf.square(tderror) / 2)
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            # update policy
            with tf.GradientTape() as tape:
                act = self.policy(batch.state)
                Q = tf.concat([batch.state, act], axis=1)
                policy_loss = - tf.reduce_mean(Q)
            policy_grad = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_variables))

            # soft update
            soft_update(self.critic.weights, self.target_critic.weights, self.tau)
            soft_update(self.policy.weights, self.target_policy.weights, self.tau)

            score = tf.reduce_mean(tderror)

            # summary
            tf.summary.scalar("TDError", score)

            return np.squeeze(score.numpy())

    def compute_tderror(self, batch):
        target_Q  = self.discount * self.target_critic(np.concatenate([batch.next_state, self.target_policy(batch.next_state)], axis=1))
        target_Q += batch.reward
        c_value = self.critic(np.concatenate([batch.state, batch.action], axis=1))
        return tf.stop_gradient(target_Q) - c_value


if __name__ == '__main__':
    import gym
    # from xikarl.utilities.replaybuffer_old import ReplayBuffer, TimeStep
    from xikarl.utilities.buffer import ReplayBuffer
    env = gym.make("Pendulum-v0")
    model = DDPG(env)

    key_dims = {
        "state": model.obs_size,
        "action": model.act_size,
        "next_state": model.obs_size,
        "reward": 1,
        "done": 1
    }
    buf = ReplayBuffer(key_dims)

    step = 0
    for n in range(30):
        obs = env.reset()
        sum_reward = 0
        for t in range(200):
            act = model.get_cation(obs)
            obs_, reward, done, _ = env.step(act)
            env.render()
            sum_reward += reward

            buf.add(
                state=obs,
                action=act,
                next_state=obs_,
                reward=reward,
                done=done
            )

            if done:
                break

            obs = obs_
        print("reward:", sum_reward)

        score = 0
        for k in range(200):
            step += 1
            batch = buf.batch(10)
            score += model.train(batch)
            if (step % 100) == 0:
                print("Step: {: 6}".format(step), "TDError: {:10.6f}".format(score/100))
                score = 0

