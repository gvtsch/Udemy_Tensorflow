# %%
import gym
import numpy as np
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# %%
class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.model = self.build_model

    def build_model(self):
        model=Sequential()
        model.add(Dense(units=100, input_dim=self.num_observations))
        model.add(Activation("relu"))
        model.add(Dense(units=self.num_actions))
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            opt="Adam",
            metrics=["accuracy"]
        )
        return model

    def get_action(self, state):
        pass

    def get_sample(self, num_episodes):
        pass

    def filter_episodes(self, rewards, episodes, percentile):
        pass

    def train(self, percentile, num_iterations, num_episodes):
        pass

    def play(self, num_episodes, render=True):
        pass

# %%
# Main
env = gym.make("CartPole-v1")
agent = Agent(env)
# agent.train()
# agent.play()


