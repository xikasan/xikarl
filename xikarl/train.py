# coding: utf-8

import os
import gym
import numpy as np
import datetime
import tensorflow as tf
from xikarl.algorithms.ddpg import DDPG
from xikarl.utilities.buffer import ReplayBuffer, RLDataDict


def main():
    env = gym.make("Pendulum-v0")
    info("env", env)

    model = DDPG(env)
    info("model", model)

    buffer = ReplayBuffer(RLDataDict(model), dtype=np.float32)
    info("buffer", buffer)

    writer = prepare_TensorBoard()
    train_body(env, model, buffer, writer)


def train_body(env, model, buf, writer):
    total_step = 0
    total_episode_step = 0

    while True:
        # collect trajectory
        episode_reward, episode_step = rollout(env, model, buf)
        total_episode_step += episode_step
        tf_episode_step = tf.constant(total_episode_step, dtype=tf.int64)
        tf.summary.scalar("PPDG/Episode_Reward", tf.constant(episode_reward), step=tf_episode_step)
        print("[RollOut] Step: {: 4}  Reward: {: 8.4f}".format(episode_step, episode_reward))

        while True:
            total_step += 1
            tderror = model.train(buf.batch(10), total_step)

            if (total_step % 50) == 0:
                print("[train] step {: 6}  TDError: {:10.6f}".format(total_step, tderror))

            if total_step == 30000:
                return

            if total_step >= total_episode_step:
                break

        print("-"*20)


def rollout(env, model, buf):
    episode_reward = 0
    episode_step = 0
    obs = env.reset()
    while True:
        obs = obs.astype(np.float32)
        act = model.get_action(obs)
        obs_, reward, done, _ = env.step(act)
        episode_reward += reward
        episode_step += 1

        buf.add(
            state=obs,
            action=act,
            next_state=obs_,
            reward=reward,
            done=done
        )

        if done:
            return episode_reward, episode_step

        obs = obs_


def prepare_TensorBoard():
    save_dir = datetime.datetime.now().strftime("%Y.%m.%d.%H%M%S")
    save_dir = "./results/" + save_dir
    os.makedirs(save_dir, exist_ok=False)

    writer = tf.summary.create_file_writer(save_dir)
    writer.set_as_default()
    return writer


def info(msg, obj=None):
    print("[info]", msg, end="")
    if obj is None:
        print("")
        return
    print(":", obj)

# def get_arguments(parser=None):


if __name__ == '__main__':
    main()
