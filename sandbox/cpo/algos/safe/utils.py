import os

def ini_training_log(file):
    # iterations = epochs
    if not os.path.exists(file.parent):
        os.makedirs(file.parent)
    f = open(file, 'w')
    return f

def store_training_log(f, log_dict):
    f.write(f"Step: {log_dict['steps']}; Reward: {log_dict['avg_reward']}; Real Concrete Safe Reward: {log_dict['real_mean_safety']}; Concrete Safe Reward: {log_dict['mean_safety']}\n")
    f.flush()
    return
