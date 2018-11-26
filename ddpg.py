import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import copy


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OtterTuneEnv import OtterTuneEnv
from Parser import Parser
import timeit


min_vals = np.array([1, '64MB', '1GB', '2MB'])
max_vals = np.array([8, '30GB', '30GB', '1GB'])
default_vals = np.array([2, '128MB', '4GB', '4MB'])
knob_names = ['max_parallel_workers_per_gather', 'shared_buffers', 'effective_cache_size', 'work_mem']
knob_types = ['integer', 'size', 'size', 'size']


def epsilon_greedy_policy(policy_action, epsilon):
    # Creating epsilon greedy probabilities to sample from.
    is_random = np.random.choice([0, 1], p=[1 - epsilon, epsilon])
    if is_random:
        # [0,1) uniform random
        action = np.random.uniform()
    else:
        action = policy_action
    return action

def get_range_raw(min_vals, max_vals, default_vals):
    min_raw_vals = []
    max_raw_vals = []
    default_raw_vals = []
    for i in range(len(min_vals)):
        min_raw_val = Parser().get_knob_raw(min_vals[i], knob_types[i])
        max_raw_val = Parser().get_knob_raw(max_vals[i], knob_types[i])
        default_raw_val = Parser().get_knob_raw(default_vals[i], knob_types[i])
        min_raw_vals.append(min_raw_val)
        max_raw_vals.append(max_raw_val)
        default_raw_vals.append(default_raw_val)
    return min_raw_vals, max_raw_vals, default_raw_vals

min_raw_vals, max_raw_vals, default_raw_vals = get_range_raw(min_vals, max_vals, default_vals)

print(default_raw_vals)
print(min_raw_vals)
print(max_raw_vals)


def get_config_knobs(knob_vals):
    config_knobs = [] 
    for i in range(len(knob_vals)):
        knob_val = Parser().get_knob_readable(knob_vals[i], knob_types[i])
        config_knobs.append(knob_val)
    return config_knobs


def load_weights(actor, critic, actor_file="actor_weights.json",
                 critic_file="critic_weights.json"):
    try:
        actor.model.load_weights(actor_file)
        critic.model.load_weights(critic_file)
        actor.target_model.load_weights(actor_file)
        critic.target_model.load_weights(critic_file)
        print("Weight load successfully")
    except:
        print("Cannot find the weight")


def ddpgTune(): 
    DEBUG = True    #Print intermediate results for debugging
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    GAMMA = 1.0     #Discount Factor
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor  1/32 ~ 0.03
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 1  # set value (0,1)
    tuning_knobs_num = 4 # of tuning knobs
    # state = (tuning_knobs_value, current_knob_id)
    state_dim = tuning_knobs_num + 1 # of state

    np.random.seed(1234)


    episode_count = 2 #000
    
    # Exploration
    step = 0
    epsilon_begin = 0.5
    epsilon_end = 0.05
    epsilon_iters = 1000 # 4 * episode


    default_latency = 0.  # ms

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a OtterTune environment
    env = OtterTuneEnv(min_raw_vals, max_raw_vals, default_raw_vals, knob_names)

    # Now load the weight
    #print("Now we load the weight")
    # load_weights(actor, critic)

    print("OtterTune Experiment Start.")
    for episode_i in range(episode_count):

        print("Episode : " + str(episode_i) )

        initial_state = env.reset()
        current_state = initial_state
        episode_reward = 0
        buffs = []
        while True:

            if step > epsilon_iters:
                epsilon = epsilon_end
            else:
                epsilon = epsilon_begin - (epsilon_begin - epsilon_end) * 1.0 * step / epsilon_iters

            policy_action = actor.model.predict(np.array([current_state]), batch_size=1)[0]
            # epsilon greedy
            action = epsilon_greedy_policy(policy_action, epsilon)
            nextstate, reward, is_terminal, debug_info = env.step(action, current_state)
            transition = [current_state, action, reward, nextstate, is_terminal]
            X_samples, X_currstates, X_nextstates, X_rewards, X_actions = buff.sample_batch()
            # print(X_samples)    
            if len(X_samples) > 0:
                X_nextactions = actor.target_model.predict(np.array(X_nextstates), batch_size=len(X_nextstates))
                Y_nextstates = critic.target_model.predict([np.array(X_nextstates), np.array(X_nextactions)], batch_size=len(X_nextstates))
                Y_minibatch = np.zeros([len(X_samples), 1])
                i = 0
                for x_state, x_action, x_reward, x_nextstate, x_is_terminal in X_samples:
                    if x_is_terminal:
                        y = x_reward
                    else:
                        y = x_reward + GAMMA * Y_nextstates[i]
                    Y_minibatch[i][0] = y
                    i += 1

                # update critic network
                critic.model.fit([np.array(X_currstates), np.array(X_actions)], np.array(Y_minibatch), batch_size=len(X_currstates), epochs=1, verbose=0)
                
                a_for_grad = actor.model.predict([np.array(X_currstates)], batch_size=len(X_currstates))
                grads = critic.gradients(X_currstates, a_for_grad)
                mean_grads = 1.0 / len(X_currstates) * grads

                actor.train(X_currstates, mean_grads)
                actor.target_train()
                critic.target_train()
            else: # initial default config
                print ('initial state')
                tmp_current_state = copy.copy(initial_state)
                tmp_next_state = copy.copy(initial_state)
                tmp_current_state[tuning_knobs_num] = step 
                tmp_next_state[tuning_knobs_num] = step + 1 
                transition = [tmp_current_state, initial_state[step], 0, tmp_next_state, is_terminal]
                nextstate = initial_state


            buffs.append(transition)
            episode_reward += reward
            current_state = nextstate

            if DEBUG is True:
                print("Episode", episode_i, "Step", step,
                      "Reward", reward, "Sample Size", len(X_samples))
            step += 1
            if is_terminal:
                break


        # print (current_state)
        # convert current knob to config knob
        rescaled_vals = Parser().rescaled(min_raw_vals, max_raw_vals, current_state[:tuning_knobs_num])
        config_vals = get_config_knobs(rescaled_vals)

        print (config_vals)

        env.change_conf(config_vals)
        
        # run tpch query to get reward
        config_reward = env.run_experiment()
        if episode_i == 0:
            default_reward = config_reward

        # scaled reward, latency: 1 - new/default
        scaled_reward = 1 - config_reward * 1.0 / default_reward
        print('reward ', config_reward, 'scaled ', scaled_reward)
        for transition in buffs:
            transition[2] = scaled_reward
            buff.append(transition)
            print (transition)
    
        # save, udf and upload
        env.save_and_upload()

    actor.save_model_weights('actor_weights.json')
    critic.save_model_weights('critic_weights.json')
    print("Finish.")

if __name__ == "__main__":
    ddpgTune()
