import numpy as np
import copy
from fabric.api import (env, local, task, lcd)
import json
from multiprocessing import Process
import time
from Parser import Parser



with open('../driver/driver_config.json', 'r') as f:
    CONF = json.load(f)

@task
def free_cache():
    cmd = 'sync; sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"'
    local(cmd)

@task
def run_tpch():
    cmd = 'PGPASSWORD=bohan psql --cluster 10/main -U bohan -d tpch40 -a -f ./q13.sql -o tpch.log > ./query.log' #2>&1 &'
    local(cmd)

@task
def run_controller():
    cmd = 'sudo gradle run -PappArgs="-c {} -d output/" --no-daemon'.\
          format(CONF['controller_config'])
    with lcd("../controller"):  # pylint: disable=not-context-manager
        local(cmd)

@task
def stop_controller():
    pid = int(open('../controller/pid.txt').read())
    cmd = 'sudo kill -2 {}'.format(pid)
    with lcd("../controller"):  # pylint: disable=not-context-manager
        local(cmd)

@task
def get_latency():
    with open('tpch.log', 'r') as f:
        lines = f.readlines()
        line = lines[-3]
        latency = line.split(':')[1].split('.')[0].strip()  # ms
        print(line)
        print(latency)
        return latency

@task
def restart_database():
    cmd = 'sudo service postgresql restart'
    local(cmd)

@task
def save_dbms_result():
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    for f_ in files:
        f_prefix = f_.split('.')[0]
        cmd = 'cp ../controller/output/{} {}/{}__{}.json'.\
              format(f_, CONF['save_path'], t, f_prefix)
        local(cmd)

    cmd = 'cp {} {}/{}__{}.txt'.\
            format('tpch.log', CONF['save_path'], t, 'queryplan')
    local(cmd)

@task
def upload_result():
    cmd = 'python3 ../../server/website/script/upload/upload.py \
           ../controller/output/ {} {}/new_result/'.format(CONF['upload_code'],
                                                           CONF['upload_url'])
    local(cmd)

@task
def add_udf():
    cmd = 'sudo python3 ../driver/LatencyUDF.py ../controller/output/ {}'.format('tpch.log')
    local(cmd)


class OtterTuneEnv(object):
    def __init__(self, min_vals, max_vals, default_vals, knob_names):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
        self.default_vals = np.array(default_vals)
        self.knob_names = np.array(knob_names)
        self.N = len(knob_names)
        self.knob_id = 0
        assert(self.N > 0)
        assert(len(min_vals) == self.N)
        assert(len(max_vals) == self.N)


    def reset(self):
        self.knob_id = 0
        # initial state is default config
        initial_state = np.zeros(self.N + 1)
        scaled_vals = Parser().scaled(self.min_vals, self.max_vals, self.default_vals)
        initial_state[:self.N] = scaled_vals
        return initial_state

    def step(self, action, state):
        '''
        action: (0, 1)
        '''
        knob_id = self.knob_id
        #print(state)
        nextstate = copy.copy(state)
        nextstate[knob_id] = action
        nextstate[self.N] = knob_id + 1
        #print('action', action)
        #print('nextstate', nextstate)
        debug_info = {}
        reward = 0
        if knob_id < self.N - 1 and knob_id >= 0:
            is_terminal = False
        elif knob_id == self.N - 1:
            is_terminal = True
        else:
            raise Exception("Invalid Knob ID {}. ".format(knob_id))

        self.knob_id += 1
        return (nextstate, reward, is_terminal, debug_info)


    def run_experiment(self):
        
        # free cache
        free_cache()
 
        # restart database 
        restart_database()
 
        p = Process(target=run_controller, args=())
        p.start()

        # sleep some time to wait for the contorller.
        time.sleep(5)

        run_tpch()
        stop_controller()
        p.join()

        reward = int(get_latency())
        print (reward)
        return reward
        
        
    def change_conf(self, config_vals):
        
        conf_path = '/etc/postgresql/10/main/postgresql.conf' 
        with open(conf_path, "r+") as postgresqlconf:
            lines = postgresqlconf.readlines()
            settings_idx = lines.index("# Add settings for extensions here\n")
            postgresqlconf.seek(0)
            postgresqlconf.truncate(0)

            lines = lines[0:(settings_idx + 1)]
            for line in lines:
                postgresqlconf.write(line)

            for i in range(len(self.knob_names)):
                s = str(self.knob_names[i]) + ' = ' + str(config_vals[i]) + "\n"
                print (s)
                postgresqlconf.write(s)
        
    
    def save_and_upload(self):
        add_udf()
        save_dbms_result()
        upload_result()


