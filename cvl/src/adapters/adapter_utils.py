"""Implementation of different utility functions for adapter layers."""

import torch
import torch.nn as nn
from transformers.activations import get_activation
import torch.nn.functional as F
import math
import numpy as np

class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear

class SmoothStep(nn.Module):
    def __init__(self, gamma=1.0):
        super(SmoothStep, self).__init__()
        self.lower_bound = -gamma / 2
        self.upper_bound = gamma / 2
        self.a3 = -2 / (gamma ** 3)
        self.a1 = 3 / (2 * gamma)
        self.a0 = 0.5

    def forward(self, x):
        return torch.where(
            x <= self.lower_bound, torch.zeros_like(x),
            torch.where(x >= self.upper_bound, torch.ones_like(x),
                        self.a3 * (x ** 3) + self.a1 * x + self.a0))

class Router(nn.Module):
    def __init__(self,config, in_dim=None):
        super().__init__()
        self.config = config
        self.model_dim = self.config.model_dim
        self.n_routers = self.config.num_routers
        self.num_adapters = self.config.num_adapters
        if in_dim is None:
            self.in_dim = self.model_dim
        else:
            self.in_dim = in_dim

        if self.config.routing_estimator == "soft_input_routing":
            self.router_weights = nn.Parameter(torch.zeros(self.in_dim, self.num_adapters, self.config.num_slots_per_adapter))
            self.scales = nn.Parameter(torch.ones(self.num_adapters, self.config.num_slots_per_adapter))
            nn.init.normal_(self.router_weights, std=config.router_init_scale)
        else:
            self.router_weights = nn.Parameter(torch.zeros(self.n_routers, self.in_dim, self.num_adapters))
            #print(f"Router weights shape: {self.router_weights.shape}")
            nn.init.normal_(self.router_weights, std=config.router_init_scale)#正态分布初始化
        if not config.no_router_bias:
            # (M,N)
            self.router_bias = nn.Parameter(torch.zeros(self.n_routers, self.num_adapters))
        self.multi_router_layer_norms = nn.ModuleList([nn.BatchNorm2d(self.in_dim) for _ in range(self.n_routers)])
        self.multi_router_weights_layer_norms = []
        for i in range(self.n_routers):
            self.multi_router_weights_layer_norms.append(nn.LayerNorm(self.in_dim))
        self.multi_router_weights_layer_norms = nn.ModuleList(self.multi_router_weights_layer_norms)
        for i in range(self.n_routers):
            self.multi_router_weights_layer_norms[i].weight = nn.Parameter(torch.ones(self.in_dim)*config.router_init_scale)

    def forward_soft_input_routing(self,x):
        # normalize x
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
        # normalize routing weights
        router_weights = self.router_weights / (torch.norm(self.router_weights, dim=0, keepdim=True) + 1e-6)
        router_weights = router_weights * self.scales[None,:,:]
        logits = torch.einsum('bmd,dnp->bmnp', x, router_weights)
        return logits


    def forward(self, x):
        # 确保输入的形状为 (batch_size, channels, height, width)
        #print('路由层接受的形状：', x.shape)
        batch_size, channel_dim, height, width = x.shape
        if self.config.routing_estimator == "soft_input_routing":# 计算soft_input_routing的logit
            return self.forward_soft_input_routing(x)

        # 数据格式转换
        if self.config.use_load_balancing:
            new_x = x
        else:
            new_x = []
            for i in range(self.n_routers):
                new_x.append(self.multi_router_layer_norms[i](x))  # x 形状 [batch_size, 512, 4, 4]

        # (M, B, C)
        new_x = torch.cat([y.unsqueeze(0) for y in new_x], dim=0)  # 将不同路由器的输出拼接
        # 对 height 和 width 维度进行全局平均池化
        new_x = torch.mean(new_x, dim=[3, 4])  # 池化后的形状为 [M, batch_size, 512]
        # 确保与 router_weights 维度匹配
        #print(f"new_x 的形状：{new_x.shape}")  # 调试信息

        #计算路由权重
        if self.config.normalize_router_weights:
            router_weights = []
            for i in range(self.n_routers):#循环处理每个路由器的权重
                router_weights.append(self.multi_router_weights_layer_norms[i](self.router_weights[i].T).T)#获取第 i 个路由器的权重并进行归一化
            router_weights = torch.cat([y.unsqueeze(0) for y in router_weights], dim=0)
        elif self.config.cosine_router:
            router_weights = []
            for i in range(self.n_routers):
                router_weights.append(self.router_weights[i] / torch.norm(self.router_weights[i], dim=0, keepdim=True))
            router_weights = torch.cat([y.unsqueeze(0) for y in router_weights], dim=0)
        else:
            router_weights = self.router_weights

        #print(f"router_weights 的形状：{router_weights.shape}")  # 调试信息

        if not self.config.no_router_bias:
            adapter_logits = torch.matmul(new_x.view(batch_size, -1), router_weights) + self.router_bias[:, None, :]
        else:
            #print(f"new_x.view(batch_size, -1) 的形状：{new_x.view(batch_size, -1).shape}")  # 调试信息
            #print(batch_size)
            adapter_logits = torch.matmul(new_x.view(batch_size, -1), router_weights)#将 new_x（经过全局平均池化后的输入特征）与 router_weights（专家权重）进行矩阵乘法得到的

        # (M, B, N)
        adapter_probs = F.softmax(adapter_logits, dim=-1)#表示每个专家在当前输入下被选择的概率
        #print(f"adapter_probs 的形状：{adapter_probs.shape}")  # 调试信息


        if self.training and self.config.same_init_then_branch > 0:
            adapter_probs = torch.ones(adapter_probs.shape).to(adapter_probs.device)
            adapter_probs = adapter_probs / torch.sum(adapter_probs, dim=-1, keepdim=True)
            return new_x, adapter_logits, adapter_probs

        if self.training and self.config.epsilon_greedy != 0:
            p = torch.rand(1).item()
            if p < self.config.epsilon_greedy:
                adapter_probs = torch.ones(adapter_probs.shape).to(adapter_probs.device)
                adapter_probs = adapter_probs / torch.sum(adapter_probs, dim=-1, keepdim=True)

        if self.training and self.config.expert_dropout != 0:
            ones = torch.ones(adapter_probs.shape)
            zeros = torch.zeros(adapter_probs.shape)
            random_tensor = torch.rand(adapter_probs.shape)
            mask = torch.where(random_tensor < self.config.expert_dropout, zeros, ones)
            mask = mask.to(adapter_probs.device)
            adapter_probs = adapter_probs * mask
            if self.config.renormalize_adapter_probs:
                adapter_probs = (adapter_probs + 1e-10) / torch.sum((adapter_probs + 1e-10), dim=-1, keepdim=True)

            neg_infs = - torch.ones(adapter_probs.shape) * 1e10
            mask2 = torch.where(random_tensor < self.config.expert_dropout, neg_infs, zeros)
            mask2 = mask2.to(adapter_probs.device)
            adapter_logits = adapter_logits + mask2

        return new_x, adapter_logits, adapter_probs


class Baseline(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, in_dim=8192):
        super().__init__()
        self.config = config
        self.model_dim = self.config.model_dim
        self.n_routers = self.config.num_routers
        self.num_adapters = self.config.num_adapters
        if in_dim is None:
            self.in_dim = self.model_dim
        else:
            self.in_dim = in_dim
        self.down_sample_size = self.in_dim // self.config.bl_reduction_factor
        #(M,C,down_size)
        self.down_samplers_weights = nn.Parameter(torch.zeros(self.n_routers, self.in_dim, self.down_sample_size))
        nn.init.normal_(self.down_samplers_weights, std=1e-2)
        #(M,down_size)
        self.down_samplers_bias = nn.Parameter(torch.zeros(self.n_routers, self.down_sample_size))

        #(M,down_size,1)
        self.up_samplers_weights = nn.Parameter(torch.zeros(self.n_routers, self.down_sample_size, 1))
        nn.init.normal_(self.up_samplers_weights, std=1e-2)
        #(M,1)
        self.up_samplers_bias = nn.Parameter(torch.zeros(self.n_routers, 1))
        
    def forward(self, new_x):
        z = torch.matmul(new_x, self.down_samplers_weights) + self.down_samplers_bias[:,None,:]
        # (M,B,1)
        val = torch.matmul(z, self.up_samplers_weights) + self.up_samplers_bias[:,None,:]
        return val
