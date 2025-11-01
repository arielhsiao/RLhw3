import warnings
import time
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Custom features extractor to use with CnnPolicy without requiring NatureCNN/image obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AdjacentTileFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 6144):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0] # 應為 16
        
        # 檢查輸入維度是否符合
        if n_input_channels != 16 or observation_space.shape[1:] != (4, 4):
            print(f"Warning: This network (CNN22) is designed for 16x4x4 observations, "
                  f"but got {observation_space.shape}. Make sure this is intended.")

        # CNN layers
        self.cnn = nn.Sequential(
            # Input: (N, 16, 4, 4)
            # 1. conv (2x2) 256 -> ReLU
            nn.Conv2d(n_input_channels, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # Output: (N, 256, 3, 3)
            
            # 2. conv (2x2) 512 -> ReLU但
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # Output: (N, 512, 2, 2)
            
            # 3. Flatten
            nn.Flatten(),
            # Output: (N, 512 * 2 * 2) = (N, 2048)
        )

        # ---------------------------------------------------------------------
        # 這部分用於動態計算展平後的大小，確保 `n_flatten` 正確
        # 即使我們已經手動算過 (2048)，這樣做更保險
        with th.no_grad():
            # 建立一個符合 observation_space 的假輸入
            sample_input = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1] # 應為 2048
        # ---------------------------------------------------------------------

        # Fully connected layers
        self.linear = nn.Sequential(
            # 4. FC 1024 -> ReLU
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            # 5. FC 256 -> ReLU (This is the features_dim)
            nn.Linear(1024, 256),
            nn.ReLU(),
            # Output: (N, 256)
        )


    def forward(self, obs: th.Tensor) -> th.Tensor:
       
        B = obs.shape[0]
        dev = obs.device

        # Convert one-hot to discrete tile indices (0-15)
        tile_id = th.argmax(obs, dim=1)  # (B, 4, 4)

        # # ----- Pairwise adjacency (horizontal + vertical) -----
        # horiz_l = tile_id[:, :, :-1].reshape(B, -1)
        # horiz_r = tile_id[:, :, 1:].reshape(B, -1)
        # vert_t = tile_id[:, :-1, :].reshape(B, -1)
        # vert_b = tile_id[:, 1:, :].reshape(B, -1)

        # first_pair = th.cat([horiz_l, vert_t], dim=1)
        # second_pair = th.cat([horiz_r, vert_b], dim=1)
        # n_pairs = first_pair.shape[1]

        # pair_matrix = th.zeros(B, n_pairs, 16, 16, device=dev)
        # b_idx = th.arange(B, device=dev).repeat_interleave(n_pairs)
        # p_idx = th.arange(n_pairs, device=dev).repeat(B)
        # a_flat, b_flat = first_pair.reshape(-1), second_pair.reshape(-1)
        # pair_matrix[b_idx, p_idx, a_flat, b_flat] = 1

        # # ----- Triplet adjacency (horizontal + vertical) -----
        # L = tile_id[:, :, :-2].reshape(B, -1)
        # C = tile_id[:, :, 1:-1].reshape(B, -1)
        # R = tile_id[:, :, 2:].reshape(B, -1)
        # T = tile_id[:, :-2, :].reshape(B, -1)
        # M = tile_id[:, 1:-1, :].reshape(B, -1)
        # D = tile_id[:, 2:, :].reshape(B, -1)

        # first_tri = th.cat([L, T], dim=1)
        # mid_tri = th.cat([C, M], dim=1)
        # last_tri = th.cat([R, D], dim=1)
        # n_triplets = first_tri.shape[1]

        # tri_tensor = th.zeros(B, n_triplets, 16, 16, 16, device=dev)
        # b_tri_idx = th.arange(B, device=dev).repeat_interleave(n_triplets)
        # seq_idx = th.arange(n_triplets, device=dev).repeat(B)
        # triples = th.stack([first_tri, mid_tri, last_tri], dim=2).reshape(-1, 3)
        # t1, t2, t3 = triples[:, 0], triples[:, 1], triples[:, 2]
        # tri_tensor[b_tri_idx, seq_idx, t1, t2, t3] = 1

        # # ----- Flatten all features -----
        # flat_pairs = pair_matrix.reshape(B, -1)
        # flat_triplets = tri_tensor.reshape(B, -1)
        flat_obs = self.linear(self.cnn(obs))
        flat_obs = flat_obs.flatten(start_dim=1)

        # features = th.cat([flat_pairs, flat_triplets, flat_obs], dim=1)
        return flat_obs



# Set hyper params (configurations) for training
my_config = {
    "run_id": "PPO-for-wandb-report-Q1",
    "algorithm": PPO,
    "policy_network": "CnnPolicy",
    "save_path": "models/PPO-for-wandb-report-Q1",
    "num_train_envs": 4,
    "epoch_num": 1500,
    "timesteps_per_epoch": 20000,
    "eval_episode_num": 10,
    }

'''
### DQN  
"batch_size": 32,
"learning_rate": 2e-4,
#"learning_rate": 1e-4,
#"learning_starts": 100,
"learning_starts": 0,
"buffer_size": 1000000,
#"buffer_size": 200000,        
"train_freq": (4, "step"),
"gradient_steps": 1,
"gamma": 0.99,
"target_update_interval": 10000,
#"target_update_interval":6000,
"exploration_fraction": 0.1,
"exploration_final_eps": 0.05,
#"exploration_fraction": 0.04,
#"exploration_final_eps": 0.02
'''



def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    maxv = 0
    highest_list = []
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        maxv = max(maxv, info[0]['highest'])
        highest_list.append(info[0]['highest'])
        avg_score   += info[0]['score']
    
    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num

    # INSERT_YOUR_CODE
    from collections import Counter
    distribution = Counter(highest_list)
    print("Distribution of highest tiles:")
    for tile, count in sorted(distribution.items()):
        print(f"Tile {tile}: {count} times")
    
    return avg_score, avg_highest, maxv

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_score = 0
    current_best_highest = 0
    maxc=0
    lowest_maxc=0
    min_eps=0
    start_time = time.time()
    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        # Uncomment to enable logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                verbose=2,
            ),
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        ### Evaluation
        eval_start = time.time()
        avg_score, avg_highest, maxv = eval(eval_env, model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        # Print training progress and speed
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance:")
        print(f"   - Avg Score: {avg_score:.1f}")
        print(f"   - Avg Highest Tile: {avg_highest:.1f}")


        wandb.log(
            {"avg_highest": avg_highest,
             "avg_score": avg_score}
        )
        
        ## Save best model
   
        if current_best_score < avg_score or current_best_highest < avg_highest:
            print("Saving New Best Model")
            if current_best_score < avg_score:
                current_best_score = avg_score
                print(f"   - Previous best score: {current_best_score:.1f} → {avg_score:.1f}")
            elif current_best_highest < avg_highest:
                current_best_highest = avg_highest
                print(f"   - Previous best tile: {current_best_highest:.1f} → {avg_highest:.1f}")

            save_path = config["save_path"]
            model.save(f"{save_path}/bestavgscorebyTA")
        print("-"*60)

        ### Save best model avg score or avg highest tile max value的model
        # episode最大的tile最大值
        if maxc <= maxv:
            maxc = maxv
            save_path = config["save_path"]
            model.save(f"{save_path}/bestbymaxc")
            max_eps = epoch
        if lowest_maxc < maxv:
            lowest_maxc = maxv
            min_eps = epoch
        print("-"*60)
        print("maxc=", maxc, "min_eps=", min_eps, "max_eps=", max_eps)

        print("---------------")
        

    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print("final maxc",maxc)

if __name__ == "__main__":

    #Create wandb session (Uncomment to enable wandb logging)
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        #sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"]
    )

    train_env = SubprocVecEnv([make_env for _ in range(my_config["num_train_envs"])])

    eval_env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    policy_kwargs = dict(
        features_extractor_class= AdjacentTileFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # features_extractor_kwargs=dict(features_dim=71936),
        net_arch=[] 
    )

    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env, 
        verbose=0,
        tensorboard_log=my_config["run_id"],
        policy_kwargs=policy_kwargs,
    )
    
    # model = my_config["algorithm"](
    #     my_config["policy_network"],
    #     train_env,
    #     verbose=0,
    #     tensorboard_log=my_config["run_id"],
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=my_config["learning_rate"],
    #     batch_size=my_config["batch_size"],
    #     buffer_size=my_config["buffer_size"],
    #     train_freq=my_config["train_freq"],
    #     gradient_steps=my_config["gradient_steps"],
    #     gamma=my_config["gamma"],
    #     target_update_interval=my_config["target_update_interval"],
    #     exploration_fraction=my_config["exploration_fraction"],
    #     exploration_final_eps=my_config["exploration_final_eps"],
    #     #device="cuda"
    # )

    '''
    # ========= 修改開始 =========
    from pathlib import Path
    model_path = Path("models/myenvmodified_arieltrain_evalchangeto10_gpu_addmonotonicityreward/bestavgscorebyTA.zip")
    
    if model_path.exists():
        print(f"✅ Loading model from {model_path}")
        model = DQN.load(model_path, env=train_env, device = "cuda")
        model.set_env(train_env)  # 確保 VecEnv 正確綁定
    else:
        print("🚀 Starting new training from scratch")
        model = my_config["algorithm"](
            my_config["policy_network"],
            train_env,
            verbose=1,
            tensorboard_log=my_config["run_id"],
            policy_kwargs=policy_kwargs,
            learning_rate=my_config["learning_rate"],
            batch_size=my_config["batch_size"],
            buffer_size=my_config["buffer_size"],
            train_freq=my_config["train_freq"],
            gradient_steps=my_config["gradient_steps"],
            gamma=my_config["gamma"],
            target_update_interval=my_config["target_update_interval"],
            exploration_fraction=my_config["exploration_fraction"],
            exploration_final_eps=my_config["exploration_final_eps"],
            device="cuda"
        )
    # ========= 修改結束 =========
    '''


    print(model.policy)

    train(eval_env, model, my_config)