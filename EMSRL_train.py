import ray
from EMSRL_env import BRLEnv as env
from ray import tune
from ray.tune import Stopper  # added
import time  # added
import matplotlib.pyplot as plt  # added
import numpy as np  #   added

# added for early stop of the execution
class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 60 * 3  # define the execution time # added

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline
# added for early stop of the execution

episode = "EMSRL"

def register_env(env_name, env_config={}):
    # env = create_env(env_name)
    tune.register_env(env_name,
                      lambda env_name: env(env_name,
                                           env_config=env_config))

env_name = 'BRLEnv'
env_config = {}  # Change environment parameters here
rl_config = dict(
    env=env_name,
    num_workers=1,
    env_config=env_config,
    lr=3e-5,
    framework='torch',
    train_batch_size=128,  #   8000
    model = dict(
        fcnet_hiddens=[256,256,256],
    )
)

# Register environment
register_env(env_name, env_config)

training_start_time = time.time()   # added running time display

# Initialize Ray and Build Agent
ray.init(ignore_reinit_error=True)

analysis = tune.run("PPO", config=rl_config, verbose=2, local_dir=f'./ray_results/rl_{episode}', stop=TimeStopper(),
                    checkpoint_freq=5)  # verbose=1

run_time_training = time.time() - training_start_time   # added running time display

# add result visualization
dfs = analysis.fetch_trial_dataframes()

list_reward_mean = []
list_reward_max = []
list_reward_min = []
for d in dfs.values():
    reward_mean_raw = d["episode_reward_mean"]
    for r_m in range(len(reward_mean_raw)):
        list_reward_mean.append(np.nansum(reward_mean_raw[1:r_m]))

    reward_max_raw = d["episode_reward_max"]
    for r_max in range(len(reward_max_raw)):
        list_reward_max.append(np.nansum(reward_max_raw[1:r_max]))

    reward_min_raw = d["episode_reward_min"]
    for r_min in range(len(reward_min_raw)):
        list_reward_min.append(np.nansum(reward_min_raw[1:r_min]))

    plt.plot(list_reward_mean, label='mean reward')
    plt.plot(list_reward_max, label='max reward')
    plt.plot(list_reward_min, label='min reward')
    # ax = d.plot("training_iteration", "episode_reward_mean", ax=ax, legend=False)

plt.xlabel("iterations")
plt.ylabel("accumulated reward")
plot_title = 'run time: ' + str(run_time_training) + ' seconds'
plt.title(plot_title)
plt.legend()
plt.show()
# add result visualization


ray.shutdown()
