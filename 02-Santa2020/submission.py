
import numpy as np
import pandas as pd
import random, os, datetime, math
from collections import defaultdict

total_reward = 0
bandit_dict = {}

def set_seed(my_seed=42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)

def get_next_bandit():
    best_bandit = 0
    best_bandit_expected = 0
    for bnd in bandit_dict:
        expect = (bandit_dict[bnd]['win'] - bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'] - (bandit_dict[bnd]['opp']>0)*1.7 + bandit_dict[bnd]['op_continue']) \
                 / (bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp']) \
                * math.pow(0.97, bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'])
        if expect > best_bandit_expected:
            best_bandit_expected = expect
            best_bandit = bnd
    return best_bandit

my_action_list = []
op_action_list = []

op_continue_cnt_dict = defaultdict(int)

def multi_armed_probabilities(observation, configuration):
    global total_reward, bandit_dict

    my_pull = random.randrange(configuration['banditCount'])
    if 0 == observation['step']:
        set_seed()
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration['banditCount']):
            bandit_dict[i] = {'win': 1, 'loss': 0, 'opp': 0, 'my_continue': 0, 'op_continue': 0}
    else:
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']
        
        my_idx = observation['agentIndex']
        my_last_action = observation['lastActions'][my_idx]
        op_last_action = observation['lastActions'][1-my_idx]
        
        my_action_list.append(my_last_action)
        op_action_list.append(op_last_action)
        
        if 0 < last_reward:
            bandit_dict[my_last_action]['win'] = bandit_dict[my_last_action]['win'] +1
        else:
            bandit_dict[my_last_action]['loss'] = bandit_dict[my_last_action]['loss'] +1
        bandit_dict[op_last_action]['opp'] = bandit_dict[op_last_action]['opp'] +1
        
        if observation['step'] >= 3:
            if my_action_list[-1] == my_action_list[-2]:
                bandit_dict[my_last_action]['my_continue'] += 1
            else:
                bandit_dict[my_last_action]['my_continue'] = 0
            if op_action_list[-1] == op_action_list[-2]:
                bandit_dict[op_last_action]['op_continue'] += 1
            else:
                bandit_dict[op_last_action]['op_continue'] = 0
        
        if last_reward > 0:
            my_pull = my_last_action
        else:
            if observation['step'] >= 4:
                if (my_action_list[-1] == my_action_list[-2]) and (my_action_list[-1] == my_action_list[-3]):
                    if random.random() < 0.5:
                        my_pull = my_action_list[-1]
                    else:
                        my_pull = get_next_bandit()
                else:
                    my_pull = get_next_bandit()
            else:
                my_pull = get_next_bandit()
    
    return my_pull
