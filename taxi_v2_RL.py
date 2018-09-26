from IPython.display import clear_output
from time import sleep
import numpy as np
import subprocess
import random
import gym

def print_frames(frames):
    for i,frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print("Timestep: %i" %(i + 1))
        print("State: %i" %(frame['state']))
        print("Action: %i" %(frame['action']))
        print("Reward: %i" %(frame['reward']))
        sleep(1.0)

def main():
    # Load the game environment and render it
    env = gym.make("Taxi-v2").env
    env.render()

    # Reset environment and set a state manually
    env.reset()
    state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
    env.s = state
    env.render()

    # IMPLEMENTING Q-LEARNING ALGORITHM
    # Q-table: n_states x n_actions matrix
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1     # Learning rate (0 <= gamma <= 1)
    gamma = 0.6     # Discount factor (0 <= gamma <= 1)
    epsilon = 0.1   # Eplore x exploit tradeoff factor (0 <= gamma <= 1)

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        frames = [] # for animation

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            # Put each rendered frame into dict for animation
            if i >= 100000:
                frames.append({
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward
                    })

            state = next_state
            epochs += 1
        
        if i % 100 == 0:
            clear_output(wait=True)
            print("Episode: %i" %i)

    print("Training finished.\n")
    print_frames(frames)


if __name__ == '__main__':
    main()

