from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers import PReLU
from keras.layers import LeakyReLU
import keras as ks
import matplotlib.pyplot as plt
import math
import gym
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from riddle_solvers import *
import time
import traceback
import numpy as np
import tensorflow as tf
import copy
import warnings
warnings.filterwarnings("ignore")

def create_q_model():
    # Network defined by the Deepmind paper
    esize = np.prod(MAZE_SIZE)
    inputs = ks.layers.Input(shape=(VECT_SIZE,))
    layer1 = ks.layers.Dense(esize, activation=LeakyReLU(alpha=0.24))(inputs)
    layer2 = ks.layers.Dense(esize, activation=LeakyReLU(alpha=0.24))(layer1)
    action = ks.layers.Dense(NUM_ACTIONS, activation="linear")(layer2)
    return ks.Model(inputs=inputs, outputs=action)


def obv_to_vector(obv):
    return np.array([i for i in obv[0]]+[i for i in obv[1]]+[i for ar in obv[2] for i in ar])


def calculate_final_score(agent_id, rescued_items, riddlesTimeDictionary, manager_):
    rescue_score = (1000*rescued_items)/(manager_.maze_map[agent_id].steps)
    riddles_score = 0
    riddles_score_dict = dict()
    for riddle in manager_.riddles_dict[agent_id].riddles.values():
        riddle_score = manager_.riddle_scores[riddle.riddle_type]*riddle.solved()
        if riddle_score > 0:
            riddle_score = riddle_score / (riddlesTimeDictionary.get(riddle.riddle_type,1)*100)
        riddles_score += riddle_score
        riddles_score_dict[riddle.riddle_type] = riddle_score
        
    total_score = (rescue_score + riddles_score)
    # print(">>>>> rescue_score: ", rescue_score, "   riddles_score: ", riddles_score, riddles_score_dict)

    if(not tuple(manager_.maze_map[agent_id].maze_view.robot)==(9,9) or not manager_.maze_map[agent_id].terminated):
        total_score = 0.8 * total_score
        # print(">>>>> total_score: ", total_score)
    
    return total_score, riddles_score_dict

def check_valid_actions(state, prev_state, prev_action, visited_cells):
    col, row = state[0]
    actions = [0, 1, 2, 3]
    if row == 0:
        actions.remove(1)
    elif row == MAZE_SIZE[0]-1:
        actions.remove(3)

    if col == 0:
        actions.remove(0)
    elif col == MAZE_SIZE[1]-1:
        actions.remove(2)

    if prev_action != None:
        if row>0 and row<MAZE_SIZE[0]-1 and np.array_equal(state[0], prev_state[0]):
            if prev_action in actions: actions.remove(prev_action)

        if col>0 and col<MAZE_SIZE[1]-1 and np.array_equal(state[0], prev_state[0]):
            if prev_action in actions: actions.remove(prev_action)
    
    actions = list(set(actions) - visited_cells[tuple(state[0])][1])
    return actions


def get_reward(state, prev_state, visited_cells, action, rescued_itesm, info):
    if not info['riddle_type'] == None:
        return rewards_dic['item']
    valid_actions = check_valid_actions(state, prev_state, action, visited_cells)
    if not valid_actions:
        mode = 'blocked'
    elif action in valid_actions:
        mode = 'valid'
    else:                  # invalid action, no change in agent position
        mode = 'invalid'

    if np.array_equal(state[0], (9, 9)): # reduce the reward if the agent is not rescued all items
        return 1.0 - ((len(riddle_solvers.keys())-rescued_itesm) / len(riddle_solvers.keys()))
    if mode == 'blocked':
        return rewards_dic['blocked']
    elif mode == 'invalid':
        return rewards_dic['invalid']
    elif mode == 'valid':
        return rewards_dic['valid'] #* (1 + 0.1*self.visited[tuple(state[0])][0] ** 2)
    


def train_model(model, model_target):
    global EPSILON
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = ks.optimizers.Adam(learning_rate=MIN_LEARNING_RATE, clipnorm=1.0)

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    epsilon = EPSILON
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    epsilon_greedy_frames = 100000
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 100000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    update_target_network = 400
    # Using huber loss for stability
    loss_function = ks.losses.Huber()
    winiing_count = 0
    continous_best_winiing_count = 0
    max_episode_reward=-np.inf

    while True:  # Run until solved
        current_state = manager.reset(agent_id)
        prev_state = copy.deepcopy(current_state)
        visited_cells = dict(((r,c), (0, set()))for r in range(MAZE_SIZE[0]) for c in range(MAZE_SIZE[1]))
        current_state_vector = obv_to_vector(current_state)
        episode_reward = 0
        riddles_solving_time = dict()
        visited_items = 0
        rescued_itesm = 0
        finish = False
        prev_action = None
        timestep = 0
        
        while True:
            current_state = copy.deepcopy(current_state)
            timestep += 1
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            frame_count += 1

            # Use epsilon-greedy for exploration
            valid_actions = check_valid_actions(current_state, prev_state, prev_action, visited_cells)
            if epsilon > np.random.rand(1)[0]:
                # Take random action
                # print("Random action  ", random_num, "  ", epsilon)
                action = random.choice(valid_actions)
            else:
                # print("Greedy action")
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(current_state_vector)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = list(filter(lambda x: x in valid_actions, tf.argsort(action_probs[0], direction='DESCENDING').numpy()))[0]
                # action = tf.argmax(action_probs[0]).numpy()
            
            # Decay probability of taking random action
            epsilon -= EPSILON_INTERVAL / epsilon_greedy_frames
            epsilon = max(epsilon, EPSILON_MIN)
            
            # Apply the sampled action in our environment
            # execute the action
            new_state, _, _, _, info = manager.step(agent_id, ACTIONS_DICT[action])
            
            if RENDER_MAZE:
                manager.render(agent_id)
                
            # in case the last action was invalid like a go throw a wall
            valid_actions = check_valid_actions(current_state, new_state, action, visited_cells)
            if not valid_actions or action not in valid_actions:
                visited_cells[tuple(current_state[0])][1].add(action)

            prev_action = action
            current_reward = 0
            if not info['riddle_type'] == None:
                visited_items += 1
                solution_time = time.time()

                solution = riddle_solvers[info['riddle_type']](info['riddle_question'])

                solution_time = time.time() - solution_time
                if info['riddle_type'] in riddles_solving_time:
                    riddles_solving_time[info['riddle_type']] = riddles_solving_time[info['riddle_type']] + solution_time
                else:
                    riddles_solving_time[info['riddle_type']] = solution_time

                new_state, _, _, truncated, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)

                rescued_itesm = info['rescued_items']
                current_score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time, manager)
                current_reward += current_score

            current_reward += get_reward(new_state, current_state, visited_cells, action, rescued_itesm, info)


            done = 0
            if np.array_equal(new_state[0], (9, 9)):
                done =  1
                winiing_count += 1
                finish = True
                if continous_best_winiing_count < SOLVED_T:
                    continous_best_winiing_count += 1
                else:
                    continous_best_winiing_count = 0

            if timestep > MAX_STEPS_PER_EPISODE:
                done=0
                finish = True
                continous_best_winiing_count = 0

            episode_reward += current_reward
            state_next_vector = obv_to_vector(new_state)

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(current_state_vector)
            state_next_history.append(state_next_vector)
            done_history.append(done)
            rewards_history.append(current_reward)
            prev_state = current_state
            current_state = new_state
            current_state_vector = state_next_vector

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > BATCH_SIZE:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=BATCH_SIZE)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample, verbose=0)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + GAMMA * tf.reduce_max(future_rewards, axis=1)

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, NUM_ACTIONS)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #if frame_count % update_target_network == 0:
            if max_episode_reward < episode_reward and episode_count > 1:
                max_episode_reward = episode_reward
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "max_episode_reward: {:.2f} at episode {}, time_step {}"
                print(template.format(max_episode_reward, episode_count, timestep))
                if timestep > 1:
                    validate(model_target, manager)
                    print("========================= validating =============================")

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if finish:
                score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time, manager)
                template = "state: {:d} | total_reward: {:.2f} | score: {:.2f} | episode {:d} | steps {:d} | visited_item: {:d} | rescued_items {:d} | winning_count {:d} | best_winning_count {:d} | epsilon {:.2f}"
                print(template.format(done, episode_reward, score, episode_count, timestep, visited_items, rescued_itesm, winiing_count, continous_best_winiing_count, epsilon))
                break

        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]

        episode_count += 1
        if continous_best_winiing_count >= STREAK_TO_END:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time, manager)
            template = "state: {:d} | total_reward: {:.2f} | score: {:.2f} | episode {:d} | steps {:d} | visited_item: {:d} | rescued_items {:d} | winning_count {:d} | best_winning_count {:d} | epsilon {:.2f}"
            print(template.format(done, episode_reward, score, episode_count, timestep, visited_items, rescued_itesm, winiing_count, continous_best_winiing_count, epsilon))
            break




def validate(pred_model, _manager):
    current_state = _manager.reset(agent_id)
    prev_state = copy.deepcopy(current_state)
    visited_cells = dict(((r,c), (0, set()))for r in range(MAZE_SIZE[0]) for c in range(MAZE_SIZE[1]))
    current_state_vector = obv_to_vector(current_state)
    episode_reward = 0
    riddles_solving_time = dict()
    visited_items = 0
    rescued_itesm = 0
    finish = False
    prev_action = None
    timestep = 0
    epsilon = EPSILON
    epsilon_greedy_frames = 1000
        
    while True:
        current_state = copy.deepcopy(current_state)
        timestep += 1
        # Use epsilon-greedy for exploration
        valid_actions = check_valid_actions(current_state, prev_state, prev_action, visited_cells)
        if epsilon > np.random.rand(1)[0]:
            # Take random action
            # print("Random action  ", random_num, "  ", epsilon)
            action = random.choice(valid_actions)
        else:
            # print("Greedy action")
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(current_state_vector)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = pred_model(state_tensor, training=False)
            # Take best action
            action = list(filter(lambda x: x in valid_actions, tf.argsort(action_probs[0], direction='DESCENDING').numpy()))[0]
            # action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        epsilon -= EPSILON_INTERVAL / epsilon_greedy_frames
        epsilon = max(epsilon, EPSILON_MIN)

        
        # Apply the sampled action in our environment
        # execute the action
        prev_action = action
        new_state, _, _, _, info = _manager.step(agent_id, ACTIONS_DICT[action])


        # in case the last action was invalid like a go throw a wall
        valid_actions = check_valid_actions(current_state, new_state, action, visited_cells)
        if not valid_actions or action not in valid_actions:
            visited_cells[tuple(current_state[0])][1].add(action)

        current_reward = 0
        if not info['riddle_type'] == None:
            visited_items += 1
            solution_time = time.time()

            solution = riddle_solvers[info['riddle_type']](info['riddle_question'])

            solution_time = time.time() - solution_time
            if info['riddle_type'] in riddles_solving_time:
                riddles_solving_time[info['riddle_type']] = riddles_solving_time[info['riddle_type']] + solution_time
            else:
                riddles_solving_time[info['riddle_type']] = solution_time

            new_state, _, _, truncated, info = manager.solve_riddle(info['riddle_type'], agent_id, solution)

            rescued_itesm = info['rescued_items']
            current_score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time, _manager)
            current_reward += current_score

        current_reward += get_reward(new_state, current_state, visited_cells, action, rescued_itesm, info)


        if np.array_equal(new_state[0], (9, 9)):
            done = 1
            finish = True
        if timestep > MAX_STEPS_PER_EPISODE:
            done=0
            finish = True

        episode_reward += current_reward
        state_next_vector = obv_to_vector(new_state)
        prev_state = current_state
        current_state = new_state
        current_state_vector = state_next_vector

        if finish:
            score, _ = calculate_final_score(agent_id, rescued_itesm, riddles_solving_time, _manager)
            template = "state: {:d} | total_reward: {:.2f} | score: {:.2f} | steps {:d} | visited_item: {:d} | rescued_items {:d}"
            print(template.format(done, episode_reward, score, timestep, visited_items, rescued_itesm))
            break





if __name__ == "__main__":

    sample_maze = np.load("sample_maze.npy") #np.load("hackathon_sample.npy")
    agent_id = "9" # add your agent id here
    
    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=sample_maze)
    env = manager.maze_map[agent_id]

    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver, 'pcap': pcap_solver, 'server': server_solver}
    maze = {}
    states = {}

    maze['maze'] = env.maze_view.maze.maze_cells.tolist()
    maze['rescue_items'] = list(manager.rescue_items_dict.keys())


    # Configuration paramaters for the whole setup
    SEED = 42
    GAMMA = 0.99  # Discount factor for past rewards
    EPSILON = 1.0  # Epsilon greedy parameter
    EPSILON_MIN = 0.1  # Minimum epsilon greedy parameter
    EPSILON_MAX = 1.0  # Maximum epsilon greedy parameter
    EPSILON_INTERVAL = (
        EPSILON_MAX - EPSILON_MIN
    )  # Rate at which to reduce chance of random action being taken
    BATCH_SIZE = 32  # Size of batch taken from replay buffer
    MAX_STEPS_PER_EPISODE = 5000


    # Number of discrete states (bucket) per state dimension
    obv = manager.reset(agent_id)
    VECT_SIZE = len(obv_to_vector(obv))
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    min_reward = -0.5 * np.prod(MAZE_SIZE, dtype=float)
    base = np.sqrt(MAZE_SIZE[0] * MAZE_SIZE[1])
    rewards_dic = {
        'blocked':  min_reward,
        'invalid': -4.0/base,
        'valid':   -1.0/np.prod(MAZE_SIZE, dtype=float),
        'item' : 0.8
    }

    print(rewards_dic)

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.002
    MIN_LEARNING_RATE = 0.1
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)*3
    RENDER_MAZE = False

    # Actions dictionary
    ACTIONS_DICT = {
        0: 'W',
        1: 'N',
        2: 'E',
        3: 'S',
    }

    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = create_q_model()
    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model()

    train_model(model, model_target)