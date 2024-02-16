import gym
import numpy as np
import random


# create a new instance of taxi, and get the initial state

learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate = 0.005
q_table = np.zeros((gym.make("Taxi-v3").observation_space.n,gym.make("Taxi-v3").action_space.n))

def training(epochs,max_steps):
    # create Taxi environment
    env = gym.make("Taxi-v3",render_mode="ansi")
    for epoch in range(epochs):
        epsilon = np.exp(-decay_rate*epoch)
        state = env.reset()[0]
        for step in range(max_steps):
            action = env.action_space.sample()
            if random.uniform(0,1) < epsilon :
                action = np.argmax(q_table[state,:])
            new_state, reward, terminated, truncated, info = env.step(action)
            q_table[state,action] += learning_rate * (reward + discount_rate * max(q_table[new_state,:]) - q_table[state,action])         
            print('step {} in epoch {}'.format(step, epoch))
            state = new_state 
            if terminated:
                break
def play():
   max_steps = 50
   env = gym.make("Taxi-v3",render_mode = "human")
   state = env.reset()
   for step in range(max_steps):
        print(state)
        action  = np.argmax(q_table[state[0],:])
        new_state = env.step(action)
        state = new_state
        env.render()

training(10000,100)
play()
