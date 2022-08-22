import os
from pathlib import Path

def ini_training_log(file):
    # iterations = epochs
    path = Path(f"{file}")
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
        print(path.parent)
    f = open(file, 'w')
    return f

def store_training_log(f, log_dict):
    f.write(f"Step: {log_dict['steps']}; Reward: {log_dict['avg_reward']}; Real Concrete Safe Reward: {log_dict['real_mean_safety']}; Concrete Safe Reward: {log_dict['mean_safety']}\n")
    f.flush()
    return
