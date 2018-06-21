import os

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CartPoleQLearningAgent:
    def __init__(self,
                 learning_rate=1.0,
                 discount_factor=0.0,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None
        self._num_actions = 2
        self.num_discrete_states = 10

        self.__boundaries = [
            (-2.5, 2.5),
            (-4, 4),
            (-0.3, 0.3),
            (-4, 4)
        ]

        self._discrete_states = [np.linspace(low, up, self.num_discrete_states) for (low, up) in self.__boundaries]
        self._len_discrete_states = self.num_discrete_states ** len(self._discrete_states)

        self.q = np.zeros((self._len_discrete_states, self._num_actions))

    def _build_state(self, observation):
        states = [np.digitize(val, self._discrete_states[i]) * (len(self._discrete_states)**i) for i,val in enumerate(observation)]
        return sum(states)

    def begin_episode(self, observation):
        self.state = self._build_state(observation)

        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        #   Based on the Q-Table, get the best action for our current state.
        action = np.argmax(self.q[self.state])

        return action

    def act(self, observation, reward):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)

        #   If we choose exploration (enable_exploration == True), we perform a random action.
        next_action = np.random.randint(0, self._num_actions)

        #   If we choose exploitation, we perform the best possible action for this state.
        if not enable_exploration:
            next_action = np.argmax(self.q[next_state])

        self.q[self.state, self.action] = (1 - self.learning_rate) * self.q[self.state, self.action]\
                                          + (self.learning_rate * (reward + self.discount_factor
                                              * self.q[next_state, np.argmax(self.q[next_state])]))

        self.state = next_state
        self.action = next_action
        return next_action


class EpisodeHistory:
    def __init__(self,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100):

        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.point_plot = None
        self.mean_plot = None
        self.fig = None
        self.ax = None

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History")

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)

        self.ax.set_title("Episode Length History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")

        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        rolling_means = pd.rolling_mean(
            rolling_mean_data,
            window=mean_kernel_size,
            min_periods=0
        )[mean_kernel_size:]
        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        avg = np.average(self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1])
        return avg >= self.goal_avg_episode_length


def log_timestep(index, action, reward, observation):
    format_string = "   ".join([
        "Timestep: {0:3d}",
        "Action: {1:2d}",
        "Reward: {2:5.1f}",
        "Cart Position: {3:6.3f}",
        "Cart Velocity: {4:6.3f}",
        "Angle: {5:6.3f}",
        "Tip Velocity: {6:6.3f}"
    ])
    print(format_string.format(index, action, reward, *observation))


def run_agent(env, verbose=False):
    max_episodes_to_run = 5000
    max_timesteps_per_episode = 200

    goal_avg_episode_length = 195
    goal_consecutive_episodes = 100

    plot_episode_count = 200
    plot_redraw_frequency = 10

    agent = CartPoleQLearningAgent(
        learning_rate=0.15,
        discount_factor=0.9,
        exploration_rate=0.6,
        exploration_decay_rate=0.96
    )

    episode_history = EpisodeHistory(
        capacity=max_episodes_to_run,
        plot_episode_count=plot_episode_count,
        max_timesteps_per_episode=max_timesteps_per_episode,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes
    )
    episode_history.create_plot()

    for episode_index in range(max_episodes_to_run):
        observation = env.reset()

        action = agent.begin_episode(observation)

        for timestep_index in range(max_timesteps_per_episode):
            # Perform the action and observe the new state.
            observation, reward, done, info = env.step(action)


            # Update the display and log the current state.
            if verbose:
                env.render()
                log_timestep(timestep_index, action, reward, observation)

            # If the episode has ended prematurely, penalize the agent.
            if done and timestep_index < max_timesteps_per_episode - 1:
                reward = -max_episodes_to_run

            # Get the next action from the agent, given our new state.
            action = agent.act(observation, reward)

            # Record this episode to the history and check if the goal has been reached.
            if done or timestep_index == max_timesteps_per_episode - 1:
                print("Episode {} finished after {} timesteps.".format(episode_index + 1, timestep_index + 1))

                episode_history[episode_index] = timestep_index + 1
                if verbose or episode_index % plot_redraw_frequency == 0:
                    episode_history.update_plot(episode_index)

                if episode_history.is_goal_reached(episode_index):
                    print()
                    print("Goal reached after {} episodes!".format(episode_index + 1))
                    return episode_history

                break

    print("Goal not reached after {} episodes.".format(max_episodes_to_run))
    return episode_history


def save_history(history, experiment_dir):
    # Save the episode lengths to CSV.
    filename = os.path.join(experiment_dir, "episode_history.csv")
    dataframe = pd.DataFrame(history.lengths, columns=["length"])
    dataframe.to_csv(filename, header=True, index_label="episode")

def __find_boundaries(env):
    obs = []

    agent = CartPoleQLearningAgent(

    )

    for episode_index in range(10000):
        observation = env.reset()
        obs.append(observation)

        action = agent.begin_episode(observation)

        for timestep_index in range(200):
            observation, reward, done, info = env.step(action)
            obs.append(observation)
            action = agent.act(observation, reward)

            if done:
                break

    obs = np.array(obs)
    print('Min values: {}'.format(obs.min(axis = 0)))
    print('Max values: {}'.format(obs.max(axis=0)))
    print('Mean values: {}'.format(obs.mean(axis=0)))
    print()



def main():
    random_state = 0
    experiment_dir = "cartpole-qlearning-1"

    env = gym.make("CartPole-v0")
    env.seed(random_state)
    np.random.seed(random_state)

    # __find_boundaries(env)

    env.monitor.start(experiment_dir, force=True, resume=False, seed=random_state)
    episode_history = run_agent(env, verbose=False)   # Set verbose=False to greatly speed up the process.
    save_history(episode_history, experiment_dir)
    env.monitor.close()


if __name__ == "__main__":
    main()
