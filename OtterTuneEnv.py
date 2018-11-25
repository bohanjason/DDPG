import numpy as np
import copy



class OtterTuneEnv(object):
    def __init__(self, min_vals, max_vals, knob_names):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
        self.knob_names = np.array(knob_names)
        self.N = len(knob_names)
        self.knob_id = 0
        assert(self.N > 0)
        assert(len(min_vals) == self.N)
        assert(len(max_vals) == self.N)


    def reset(self):
        self.knob_id = 0
        return np.zeros(self.N)

    def step(self, action, state):
        '''
        action: (0, 1)
        '''
        knob_id = self.knob_id
        #print(state)
        nextstate = copy.copy(state)
        nextstate[knob_id] = action
        #print(nextstate)
        debug_info = {}
        if knob_id < self.N - 1 and knob_id >= 0:
            reward = 0
            is_terminal = False
        elif knob_id == self.N - 1:
            is_terminal = True
            # TO DO: get reward
            reward = 1
        else:
            raise Exception("Invalid Knob ID {}. ".format(knob_id))

        self.knob_id += 1
        return (nextstate, reward, is_terminal, debug_info)

    def change_conf(self, config_vals):
        for i in range(len(self.knob_names)):
            s = str(self.knob_names[i]) + ' = ' + str(config_vals[i])
            print (s)
