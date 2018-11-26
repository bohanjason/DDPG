import numpy as np
import math
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN_SIZE = 32

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.model.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss='mse')
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(suffix)

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(weight_file)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        concat_input = Concatenate(axis=-1)([S, A])
        sequence= Sequential([
            Dense(HIDDEN_SIZE, activation='relu'),  # First hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Second hidden layer
            Dense(HIDDEN_SIZE, activation='relu'),  # Third hidden layer
            Dense(1, activation='linear'),  # Output layer
        ])#critic_model
        sequence_out = sequence(concat_input)
        model = Model([S, A], sequence_out)
        return model, A, S 
