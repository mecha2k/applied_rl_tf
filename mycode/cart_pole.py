import gym
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers


def main():
    batch_size = 64
    n_episodes = 10000

    lr = 1e-3
    gamma = 0.99

    env = gym.make("CartPole-v1")
    obs = env.reset()

    model = models.Sequential()
    model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    #
    # def create_policy_model(self, input_shape):
    #     input_layer = layers.Input(shape=input_shape)
    #     advantages = layers.Input(shape=[1])
    #
    #     hidden_layer = layers.Dense(
    #         units=self.n_units,
    #         activation=self.hidden_activation,
    #         use_bias=False,
    #         kernel_initializer=glorot_uniform(seed=42),
    #     )(input_layer)
    #
    #     output_layer = layers.Dense(
    #         units=self.n_outputs,
    #         activation=self.output_activation,
    #         use_bias=False,
    #         kernel_initializer=glorot_uniform(seed=42),
    #     )(hidden_layer)
    #
    #     def log_likelihood_loss(actual_labels, predicted_labels):
    #         log_likelihood = backend.log(
    #             actual_labels * (actual_labels - predicted_labels)
    #             + (1 - actual_labels) * (actual_labels + predicted_labels)
    #         )
    #         return backend.mean(log_likelihood * advantages, keepdims=True)
    #
    #     if self.loss_function == "log_likelihood":
    #         self.loss_function = log_likelihood_loss
    #     else:
    #         self.loss_function = "categorical_crossentropy"
    #
    #     policy_model = Model(inputs=[input_layer, advantages], outputs=output_layer)
    #     policy_model.compile(loss=self.loss_function, optimizer=Adam(self.learning_rate))
    #     model_prediction = Model(input=[input_layer], outputs=output_layer)
    #     return policy_model, model_prediction


if __name__ == "__main__":
    main()
