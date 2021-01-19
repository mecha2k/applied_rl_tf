import gym


def cartpole():
    env = gym.make("CartPole-v1")
    env.reset()
    for _ in range(10):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("Step {}:".format(_))
        print("action: {}".format(action))
        print("observation: {}".format(observation))
        print("reward: {}".format(reward))
        print("done: {}".format(done))
        print("info: {}".format(info))


if __name__ == "__main__":
    cartpole()
