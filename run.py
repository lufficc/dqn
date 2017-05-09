import numpy as np
import gym
import sys
import matplotlib.pyplot as plt
from core.env import *
from core.dqn import DeepQNetwork
from core.model import SimpleNeuralNetwork
from core.model import CNN
from gym import wrappers
sys.path.append("game/")
import random
import wrapped_flappy_bird as game
import cv2
import pickle


class CartPoleEnv(Env):
    def __init__(self, monitor=False):
        self.env = gym.make('CartPole-v0')

    def step(self, action_index):
        s, r, t, i = self.env.step(action_index)
        return s, r, t, i

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()


image_size = 64


class FlappyBirdEnv(Env):
    def __init__(self):
        self.env = game.GameState()
        self.prev_image_data = None

    def step(self, action_index):
        # image_data: (288, 512, 3)
        image_data, reward, terminal = self.env.frame_step(
            self.get_action(action_index))
        image_data = self.process_image_data(image_data)
        if self.prev_image_data is None:
            state = np.stack(
                (image_data, image_data, image_data, image_data), axis=2)
        else:
            image_data = np.reshape(image_data, (image_size, image_size, 1))
            state = np.append(
                image_data, self.prev_image_data[:, :, :3], axis=2)
        self.prev_image_data = state
        return state, reward, terminal, {}

    def reset(self):
        self.env.reset()
        image_data, reward, terminal = self.env.frame_step(self.get_action(0))
        image_data = self.process_image_data(image_data)
        state = np.stack(
            (image_data, image_data, image_data, image_data), axis=2)
        self.prev_image_data = state
        return state

    def process_image_data(self, image_data):
        image_data = image_data[:, :410, :]
        image_data = cv2.cvtColor(
            cv2.resize(image_data, (image_size, image_size)),
            cv2.COLOR_BGR2GRAY)
        _, image_data = cv2.threshold(image_data, 1, 255, cv2.THRESH_BINARY)
        return image_data

    def render(self):
        # self.env.render()
        pass

    def get_action(self, action_index):
        action = np.zeros(2)
        action[action_index] = 1
        return action


def runGame(env, network):
    state = env.reset()
    while True:
        env.render()
        action = network.action(state)
        state, reward, terminal, _ = env.step(action)
        if terminal:
            state = env.reset()


def train_CartPole():
    model = SimpleNeuralNetwork([4, 24, 2])
    env = CartPoleEnv()
    qnetwork = DeepQNetwork(
        model=model, env=env, learning_rate=0.0001, logdir='./tmp/CartPole/')
    qnetwork.train(4000)

    # runGame(env, qnetwork)


def train_FlappyBirdEnv(train=True):
    model = CNN(img_w=image_size, img_h=image_size, num_outputs=2)
    env = FlappyBirdEnv()

    def explore_policy(epsilon):
        if random.random() < 0.95:
            action_index = 0
        else:
            action_index = 1
        return action_index

    qnetwork = DeepQNetwork(
        model=model,
        env=env,
        learning_rate=1e-6,
        initial_epsilon=1,
        final_epsilon=0,
        decay_factor=0.999999,
        explore_policy=explore_policy,
        save_per_step=1000,
        logdir='./tmp/FlappyBird/')
    if train:
        qnetwork.train(10000)
    else:
        runGame(env, qnetwork)


if __name__ == '__main__':
    train_CartPole()
