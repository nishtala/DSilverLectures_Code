#!/usr/bin/env python3

import numpy as np
import gym
#from gym import wrappers
import deeprl_hw1

class value_iteration:
    def __init__(self):
        self.env = gym.make(env_name)
        self.env_state_size = self.env.nS #observation_space
        self.env_action_size = self.env.nA #action_space
        # Initialize value function to some random value
        self.value_func = np.zeros(self.env_state_size)
        self.policy = np.zeros(self.env_state_size)
        self.max_iterations = 100000
        self.eps = 1e-20

    def find_optimal_value_function(self):
        for iters in range(self.max_iterations):
            prev_v = np.copy(self.value_func)
            for s in range(self.env_state_size):
                # Imagining this as a graph from DSilver backward graph: Take one step back from each step you take.
                q_sa = [sum(\
                        [p * (r * prev_v[s_])\
                        for p, s_, r, _ in self.env.P[s][a]])\
                        for a in range(self.env_action_size)]
                self.value_func[s] = max(q_sa)
            if (np.sum(np.fabs(prev_v - self.value_func))\
                        <= self.eps):
                print ("value iter has converged after {}".format(iters+1))
                break

    def extract_optimal_policy(self):
        for s in range(self.env_state_size):
            q_sa = np.zeros(self.env_action_size)
            for a in range(self.env_action_size):
                for p, s_, r, _ in self.env.P[s][a]:
                    q_sa[a] += (p * (r + gamma * self.value_func[s_]))
            self.policy[s] = np.argmax(q_sa)


    def evaluate_policy(self):
        # Let's play 100 games
        games = 100
        for game in range(games):
            obs = self.env.reset()
            total_reward = 0
            step_idx = 0
            while True:
                obs, reward, done, _ = self.env.step(int(self.policy[obs]))
                total_reward += (gamma ** step_idx * reward)
                step_idx +=1
                if done:
                    break
            print ("reward: {}".format(total_reward))

class policy_iter:
    def __init__(self):
        self.env = gym.make(env_name)
        self.env_state_size = self.env.nS #observation_space
        self.env_action_size = self.env.nA #action_space
        # Initialize random policy
        self.policy = np.random.choice(self.env_action_size, \
                size = (self.env_state_size))
        self.max_iterations = 100000
        self.eps = 1e-10
        for iters in range(self.max_iterations):
            self.policy_evaluation()
            self.policy_improvement()

    def policy_evaluation(self):
        while True:
            prev_v = np.copy(self.policy)
            for s in range(self.env_state_size):
                policy_a = self.policy[s]
                self.policy[s] = sum(\
                        [p * (r * gamma * prev_v[s_]) for p, s_,r, _ in self.env.P[s][policy_a]])
            if (np.sum(np.fabs(prev_v - self.policy)) <= self.eps):
                print ("value iter has converged")
                break

    def policy_improvement(self):
        best_policy = np.zeros(self.env_state_size)
        for s in range(self.env_state_size):
            q_sa = np.zeros(self.env_action_size)
            for a in range(self.env_action_size):
                q_sa[a] = sum([p * (r + gamma * self.policy[s_]) for p, s_, r, _ in self.env.P[s][a])
            self.policy[s] = np.argmax(q_sa)

    def evaluate_policy(self):
        # Let's play 100 games
        games = 100
        for game in range(games):
            obs = self.env.reset()
            total_reward = 0
            step_idx = 0
            while True:
                obs, reward, done, _ = self.env.step(int(self.policy[obs]))
                total_reward += (gamma ** step_idx * reward)
                step_idx +=1
                if done:
                    break
            print ("reward: {}".format(total_reward))




def main():
    global env_name, gamma
    # Initiate the game
    env_name = 'Deterministic-4x4-FrozenLake-v0'#FrozenLake8x8-v0'
    # Discount factor
    gamma = 0.9
    vl_iter = value_iteration()
    vl_iter.find_optimal_value_function()
    vl_iter.extract_optimal_policy()
    vl_iter.evaluate_policy()




if __name__ == '__main__':
    main()
