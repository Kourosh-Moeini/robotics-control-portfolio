import os
import numpy as np
import gymnasium as gym
import matlab.engine
from collections import deque

class MatlabEnv_Dynamic(gym.Env):
    def __init__(self, matter='PDMS2', mag_set='Set2'):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.path.abspath(os.path.join(os.getcwd(), '../../2) Matlab_Codes/2) Environment-mex-3d-bvp4c-dynamic')))

        self.sim = self.eng.Simulation(matter, mag_set)
        
        # Assume MATLAB returns bounds for force/torque and n0/m0
        L0, F0, Tau0 = self.eng.feval('limits', self.sim, nargout=3)
        F0 = np.array(F0).astype(np.float32).flatten()
        Tau0 = np.array(Tau0).astype(np.float32).flatten()
        L0 = np.array(L0).astype(np.float32).flatten()

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([-L0, -L0, np.array([0.0], dtype=np.float32)]),
            high=np.concatenate([L0, L0, L0]),
            dtype=np.float32
        )

        Fxmin = np.array([-0.08], dtype=np.float32)
        Fxmax = np.array([ 0.08], dtype=np.float32)
        Fymin = np.array([-0.06], dtype=np.float32)
        Fymax = np.array([ 0.1], dtype=np.float32)
        Fzmin = np.array([-0.0], dtype=np.float32)
        Fzmax = np.array([ 0.1], dtype=np.float32)

        Txmin = np.array([-0.0004], dtype=np.float32)
        Txmax = np.array([ 0.0004], dtype=np.float32)
        Tymin = np.array([-0.0004], dtype=np.float32)
        Tymax = np.array([ 0.0004], dtype=np.float32)
        Tzmin = np.array([-0.00002], dtype=np.float32)
        Tzmax = np.array([0.0], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.concatenate( [Fxmin, Fymin, Fzmin, Txmin, Tymin, Tzmin]),
            high=np.concatenate([Fxmax, Fymax, Fzmax, Txmax, Tymax, Tzmax]),
            dtype=np.float32
        )

        self.state_range = self.observation_space.high - self.observation_space.low
        self.action_range = self.action_space.high - self.action_space.low
        self.threshold_error = 0.005
        self.previous_action = None  # To correlate successive actions for related wrenches

    def rand_pos(self):
        action_low = self.action_space.low
        action_high = self.action_space.high
        
        # Sample action related to previous if available
        if self.previous_action is None:
            rand_act = np.random.uniform(action_low, action_high)
        else:
            # Perturb previous action to generate related wrench
            perturbation_std = 0.1 * self.action_range  # 10% of action range as std for perturbation
            rand_act = np.clip(self.previous_action + np.random.normal(0, perturbation_std), action_low, action_high)
        
        rand_act = np.array([-0.05, -0.0, 0.0, 0.0, 0.0, 0.0])
        self.previous_action = rand_act.copy()  # Update for next reset
        
        mat_action = matlab.double(rand_act.tolist())
        _, _, rand_pos, rand_ang, _ = self.eng.feval('rand_pos', self.sim, mat_action, nargout=5)
        rand_pos = np.array(rand_pos).flatten().astype(np.float32)
        return rand_pos
    
    def reset(self, new_ep):
        if new_ep:
           pos, _ = self.eng.feval('reset', self.sim, nargout=2)
           pos = np.array(pos).flatten().astype(np.float32)

        rand_pos = self.rand_pos()
        self.current_state = rand_pos
        self.prev_error = 0.0
        self.sum_error = 0.0
        self.prev_act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self.current_state

    def create_video(self, video_path):
        # Call MATLAB video creation if needed (adapt for BVP visualization)
        self.eng.feval('create_video', self.sim, video_path, nargout=0)

    def step(self, action, target):
        # target is the current state (target force/torque)
        mat_action = matlab.double(action.tolist())

        # Call MATLAB to solve BVP with n0_m0 (action) and target
        F_ext, T_ext, next_pos, next_ang, next_q, next_w, terminated = self.eng.feval('step', self.sim, mat_action, nargout=7)
        truncated = 1

        next_pos = np.array(next_pos).flatten().astype(np.float32)

        next_state = next_pos

        done = bool(truncated)
        terminated = bool(terminated)

        # Compute NRMSE
        error_vec = next_state - target
        error_rmse = np.sqrt(np.mean((error_vec ** 2)))

        # Error summation
        self.sum_error += error_rmse

        # Error summation
        diff_error = abs(self.prev_error - error_rmse)
        self.prev_error = error_rmse

        # Action error
        action_rmse = np.sqrt(np.mean((action ** 2)))
        self.prev_act = action.copy()

        k = 30
        reward = np.exp(-k * error_rmse) # Reward is now between 0 (high error) and 1 (zero error)

        success = 0

        if error_rmse < self.threshold_error:
            success = 1

        if terminated:
            reward = 0.0

        return (next_state, float(reward), done, terminated, success, error_rmse,
                self.sum_error, diff_error, action_rmse)
