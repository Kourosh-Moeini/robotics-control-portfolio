import os
import numpy as np
import gym
import matlab.engine
from scipy.io import loadmat
from sklearn.neighbors import LocalOutlierFactor

class MatlabEnv(gym.Env):
    def __init__(self, matter='PDMS2', mag_set='Set3', dataset_path='Catheter_DL_Dataset.mat'):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.abspath(os.path.join(os.getcwd(), '1) Environment-mex-3d')))
        self.sim = self.eng.Simulation(matter, mag_set)

        L0, T0, mag_l = self.eng.feval('limits', self.sim, nargout=3)
        L0 = np.array(L0).astype(np.float32)
        T0 = np.array(T0).astype(np.float32)
        mag_l = np.array(mag_l).astype(np.float32)
        mag_l = np.array(0.09).astype(np.float32)
        self.sim_freq = 100  # Assuming the simulation runs at 100 Hz
        self.control_freq = 10  # Desired control frequency in Hz
        # self.control_freq = 20  # Desired control frequency in Hz
        self.hold_steps = int(self.sim_freq / self.control_freq)  # Steps to hold each action

        # Action space
        self.action_space = gym.spaces.Box(
            low=np.array([-L0, -L0, L0 + mag_l], dtype=np.float32),
            high=np.array([L0, L0, 3*L0], dtype=np.float32),
            dtype=np.float32
        )
       
        # Observation space includes previous action
        pos_low =  np.array([-L0, -L0, 0.0], dtype=np.float32)
        pos_high = np.array([+L0, +L0, +L0], dtype=np.float32)
        error_low = pos_low - pos_high
        error_high = pos_high - pos_low
        vel_low =  np.array([-L0/T0, -L0/T0, -L0/T0], dtype=np.float32)
        vel_high = np.array([L0/T0, L0/T0, L0/T0], dtype=np.float32)
       
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([pos_low, error_low, vel_low]),
            high=np.concatenate([pos_high, error_high, vel_high]),
            dtype=np.float32
        )
       
        self.threshold_error_pos = 0.001
        self.threshold_error_pos = 0.0035
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])

        # --- DATASET LOADING AND CLEANING ---
        print("Loading dataset...")
        mat_data = loadmat(dataset_path) # Pass PATHS['dataset'] here if managed externally
        dataset = mat_data['dataset']
        
        # dataset_s1 = dataset
        
        # # We use Local Outlier Factor (LOF) on the 3D positions to find points 
        # # that are technically within bounds but are sparse/unstable artifacts.
        # print("Running Spatial Density analysis (this may take a moment)...")
        # current_positions = dataset_s1[:, 2:5]
        
        # # n_neighbors=50 provides a good look at local regions. 
        # # contamination is set to 'auto' or a strict percentage to shave off the fuzz.
        # lof = LocalOutlierFactor(n_neighbors=50, contamination=0.03, n_jobs=-1)
        # spatial_inlier_mask = lof.fit_predict(current_positions) == 1
        
        # dataset_s2 = dataset_s1[spatial_inlier_mask]
        # print(f"Stage 2 (Spatial Density LOF) removed {dataset_s1.shape[0] - dataset_s2.shape[0]} points.")
        
        # self.cleaned_dataset = dataset_s2
        self.cleaned_dataset =  dataset
        self.dataset_size = self.cleaned_dataset.shape[0]
        self.current_dataset_idx = 0
        
        # Precompute start indices for each episode for quick lookup during wrap-around
        self.ep_start_indices = {}
        for i in range(self.dataset_size):
            ep_num = int(self.cleaned_dataset[i, 0])
            if ep_num not in self.ep_start_indices:
                self.ep_start_indices[ep_num] = i
        print("Dataset ready.")


    def rand_setpoint(self):
        # Initialize traversal direction if it doesn't exist (1 for forward, -1 for backward)
        # if not hasattr(self, 'step_direction'):
        #     self.step_direction = 1
        current_ep = int(self.cleaned_dataset[self.current_dataset_idx, 0])
        potential_next_idx = self.current_dataset_idx + self.step_direction

        # Check if the potential next index is out of bounds or crosses an episode boundary
        if (potential_next_idx >= self.dataset_size or 
            potential_next_idx < 0 or 
            int(self.cleaned_dataset[potential_next_idx, 0]) != current_ep):
            
            # Reverse the direction
            self.step_direction *= -1
            
            # Calculate the new next index with the reversed direction
            potential_next_idx = self.current_dataset_idx + self.step_direction
            
            # Edge case fallback: if reversing also pushes us out of bounds 
            # (e.g., episode has only 1 step), just stay on the current index.
            if (potential_next_idx >= self.dataset_size or 
                potential_next_idx < 0 or 
                int(self.cleaned_dataset[potential_next_idx, 0]) != current_ep):
                potential_next_idx = self.current_dataset_idx

        self.current_dataset_idx = potential_next_idx

        # Assign targets from cols 2:5 (p_tip)
        self.p_target = self.cleaned_dataset[self.current_dataset_idx, 2:5].astype(np.float32)

        # Update metrics
        self.error_tip_t = self.p_tip_t - self.p_target

    def reset(self, action0=None, render=False):
        self.error_tip_t = np.zeros(3)
        self.step_direction = 1
        while True:
            p_ext_init = self.action_space.sample() if action0 is None else action0
            mag_config0 = np.concatenate([p_ext_init, self.quat])
            p_tip0, _, v_tip0, _, reset_term = self.eng.feval('reset', self.sim, matlab.double([mag_config0.tolist()]), nargout=5)
            if not bool(reset_term) or action0 is not None:
                break     

        self.p_tip_t = np.ravel(p_tip0).astype(np.float32)
        self.v_tip_t = np.ravel(v_tip0).astype(np.float32)
        self.p_ext_t = p_ext_init

        # --- FIND CLOSEST STATE IN DATASET ---
        dataset_p_tip = self.cleaned_dataset[:, 2:5]
        
        pos_dist = np.linalg.norm(dataset_p_tip - self.p_tip_t, axis=1)
        
        # find nearest overall neighbor
        self.current_dataset_idx = int(np.argmin(pos_dist))

        self.render = render
        if not self.render:
            self.rand_setpoint()

        return self._get_state()

    def create_video(self, video_path):
        # Call the create_video method in MATLAB with the specified video path
        self.eng.feval('create_video', self.sim, video_path, nargout=0)

    def step(self, action):
        p_ext_t1 = action
        mat_env_action = matlab.double(np.concatenate([p_ext_t1, self.quat]).tolist())

        func = 'test_model' if self.render else 'step'

        step = 0

        while step < self.hold_steps:
            # Only use 'test_model' (which saves history) on the very last hold step if rendering is on.
            # Otherwise, use the standard 'step' function for intermediate physics calculations.
            if self.render and (step == self.hold_steps - 1):
                func = 'test_model'
            else:
                func = 'step'
                
            p_tip_t1, _, v_tip_t1, _, step_term = self.eng.feval(func, self.sim, mat_env_action, nargout=5)
            step += 1

        p_tip_t1 = np.ravel(p_tip_t1).astype(np.float32)
        v_tip_t1 = np.ravel(v_tip_t1).astype(np.float32)

        reward, success, metrics = self._compute_reward_error(p_tip_t1, v_tip_t1, p_ext_t1)

        self.p_tip_t, self.v_tip_t = p_tip_t1, v_tip_t1
        self.p_ext_t = p_ext_t1

        if not self.render:
            self.rand_setpoint()

        info = {**metrics, 'is_success': success, 'step_term': bool(step_term)}
        return self._get_state(), reward, info

    def _compute_reward_error(self, p_tip_t1, v_tip_t1, p_ext_t1):
        pos_norm = np.linalg.norm(p_tip_t1 - self.p_target)
        r_pos =  100 * np.exp(-100 * pos_norm)

        # des_v_tip_t1 = (self.p_target - self.p_tip_t) / 0.01
        des_v_tip_t1 = (self.p_target - self.p_tip_t) / 0.1
        v_norm = np.linalg.norm(v_tip_t1 - des_v_tip_t1)
        r_v = 10 * np.exp(-10 * v_norm)

        a_norm = np.linalg.norm(p_ext_t1 - self.p_ext_t)

        tot_reward = float(r_pos + r_v)

        success = (pos_norm < self.threshold_error_pos)

        return tot_reward, success, {'pos_norm': pos_norm, 'v_norm': v_norm, 'a_norm': a_norm, 'r_pos': r_pos, 'r_v': r_v}

    def _get_state(self):
        return np.concatenate([self.p_tip_t, self.error_tip_t, self.v_tip_t], dtype=np.float32)

    def create_video(self, video_path):
        self.eng.feval('create_video', self.sim, video_path, nargout=0)

    def close(self):
        self.eng.quit()
