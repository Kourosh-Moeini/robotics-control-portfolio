"""Neural-network definitions extracted from the archived research notebook.

CPU is selected here so the forward-pass example is portable. No weights, training
run or robot connection is loaded by importing this file.
"""
import numpy as np
import torch
from torch import nn
from collections import deque

device = torch.device("cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, action_min, hidden_sizes, state_min, state_max):
        super().__init__()
        self.register_buffer('action_max', torch.tensor(action_max, dtype=torch.float32))
        self.register_buffer('action_min', torch.tensor(action_min, dtype=torch.float32))
        self.register_buffer('state_min', torch.tensor(state_min, dtype=torch.float32))
        self.register_buffer('state_max', torch.tensor(state_max, dtype=torch.float32))
        layers = []
        input_dim = state_dim  # State only (6D: force + torque)
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, size), nn.LayerNorm(size), nn.ReLU()])
            input_dim = size
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers).to(device)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    def forward(self, state):
        state_norm = 2 * (state - self.state_min) / (self.state_max - self.state_min) - 1
        act_norm = torch.tanh(self.net(state_norm))
        # act_norm = torch.tanh(self.net(state))
        act = 0.5 * (act_norm + 1) * (self.action_max - self.action_min) + self.action_min
        return act

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_sizes, input_min, input_max):
        super().__init__()
        self.register_buffer('input_min', torch.tensor(input_min, dtype=torch.float32))
        self.register_buffer('input_max', torch.tensor(input_max, dtype=torch.float32))
        layers = []
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, size), nn.LayerNorm(size), nn.ReLU()])
            input_dim = size
        layers.append(nn.Linear(input_dim, 1))
        self.q_net = nn.Sequential(*layers).to(device)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x_norm = 2 * (x - self.input_min) / (self.input_max - self.input_min) - 1
        return self.q_net(x_norm)
        # return self.q_net(x)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        
        self.states = torch.zeros((capacity, state_dim), device=device, dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), device=device, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), device=device, dtype=torch.float32)
        self.dones = torch.zeros(capacity, device=device, dtype=torch.float32)
    
    def add(self, state, action, reward, next_state, done):
        idx = self.idx
        
        self.states[idx] = torch.as_tensor(state, device=device, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(action, device=device, dtype=torch.float32)
        self.rewards[idx] = torch.as_tensor(reward, device=device, dtype=torch.float32)
        self.next_states[idx] = torch.as_tensor(next_state, device=device, dtype=torch.float32)
        self.dones[idx] = torch.as_tensor(done, device=device, dtype=torch.float32)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)
    
    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.idx
        indices = torch.randint(0, max_idx, (batch_size,), device=device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
