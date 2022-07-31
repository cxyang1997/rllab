
def ini_training_log(file):
    # iterations = epochs
    f = open(file, 'w')
    return f

def store_training_log(f, log_dict):
    f.write(f"Step: {log_dict['steps']}; Reward: {log_dict['avg_reward']}; Concrete Safe Reward: {log_dict['mean_safety']}\n")
    f.flush()
    return
