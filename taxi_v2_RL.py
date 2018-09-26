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

    # A reward table P (states x actions matrix) is created with the taxi environment.
    # Each element of this table is a dictionary that has the structure 
    #           {action: [(probability, next_state, reward, done)]}
    # An action can be encoded from 0 to 5, corresponding to (south, north, east, west, pickup, dropoff).
    env.P[state] # Check state-action structure for a given "state"


    # IMPLEMENTING Q-LEARNING ALGORITHM
    # Q-table is a n_states x n_actions matrix representing state-action values.
    # Formally,
    #    Q(s,a) = (1-alpha)*Q(s,a) + alpha*[r + gama*max_a(Q(next_s, all_a))]
    # Where:
    #   @ alpha is the learning rate (0 < alpha < 1)
    #   @ gama is the discount factor (0 < gama < 1)
    #   @ s is the state
    #   @ a is the action
    #   @ r is the reward
    # It's first initialized to 0, and then values are updated after training.
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

