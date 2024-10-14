"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from adapters.adapter_utils import Activations
import ipdb
import copy
import math
from sparsemax import Sparsemax
from .adapter_utils import Router, Baseline
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

#主要实现专家适配器层的逻辑。其核心功能是根据输入数据，通过下采样和上采样操作，动态地调整特征表示。
class Adapter(nn.Module):
    """Modified Adapter layer to incorporate clean and adversarial pre-trained experts."""

    def __init__(self, config, channel_dim=None, clean_model=None, adv_model=None):
        super().__init__()
        self.config = config
        self.channel_dim = channel_dim
        self.clean_model = clean_model  # 预训练的干净模型
        self.adv_model = adv_model  # 预训练的对抗模型
        self.alpha = 1.0  # 权重衰减参数

    def calculate_clean_weight_based_on_loss(self, clean_loss, alpha=1.0):
        clean_loss = clean_loss + 1e-6  # 避免除零错误
        clean_weight = alpha / clean_loss  # 损失越低，权重越高
        return clean_weight

    def calculate_adv_weight_based_on_loss(self, adv_loss, alpha=1.0):
        adv_loss = adv_loss + 1e-6  # 避免除零错误
        adv_weight = alpha / adv_loss  # 损失越低，权重越高
        return adv_weight

    def forward(self, x,  expert_selector,labels=None,):

        # 1. Parameter Averaging Routing
        #print("专家层输入的形状为：",x.shape)

        if self.config.routing_estimator == 'parameter_averaging_routing':

            criterion = nn.CrossEntropyLoss()
            clean_output = self.clean_model(x)
            #print("clean_output的形状:",clean_output.shape)
            clean_loss = criterion(clean_output, labels)
            #print("clean_loss:",clean_loss)
            clean_weight = self.calculate_clean_weight_based_on_loss(clean_loss)  # 自定义函数
            #print("clean_weight:",clean_weight)

            adv_output = self.adv_model(x)
            adv_loss = criterion(adv_output, labels)
            adv_weight = self.calculate_adv_weight_based_on_loss(adv_loss.item())  # 自定义函数

            output = clean_weight * clean_output + adv_weight * adv_output
            #print("output的形状:",output.shape)


        # 2. GS-ST Routing (Gumbel-Softmax routing)
        elif self.config.routing_estimator == 'gs_st_routing':
            # expert_selector 是经过 Gumbel-Softmax 采样后的概率，probability vector
            clean_weight = expert_selector[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            adv_weight = expert_selector[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # 使用 clean 模型和 adversarial 模型分别计算输出
            clean_output = self.clean_model(x)
            adv_output = self.adv_model(x)

            # 根据采样后的权重加权输出
            output = clean_weight * clean_output + adv_weight * adv_output

        # 3. Soft Input Routing
        elif self.config.routing_estimator == 'soft_input_routing':
            # expert_selector 是 logits，直接作为 softmax 输入来选择专家
            expert_probs = F.softmax(expert_selector, dim=-1)

            clean_weight = expert_probs[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            adv_weight = expert_probs[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # 使用 clean 模型和 adversarial 模型分别计算输出
            clean_output = self.clean_model(x)
            adv_output = self.adv_model(x)

            # 根据 softmax 权重组合输出
            output = clean_weight * clean_output + adv_weight * adv_output

        else:
            raise ValueError(f"Unsupported routing strategy: {self.config.routing_estimator}")

        return output


#作为一个控制器，负责管理和调用多个专家层。它根据不同的任务或数据特征动态选择和配置专家。
class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config, channel_dim, clean_model, adv_model):
        super().__init__()
        self.config = config
        self.n_routers = self.config.num_routers if self.config.num_adapters > 1 else 1

        self.clean_model = clean_model  # 传入的预训练干净模型
        self.adv_model = adv_model  # 传入的预训练对抗模型

        # 保留 ResNet 输出的通道数
        self.multi_adapters = Adapter(config, channel_dim, clean_model, adv_model)
        self.multi_routers = Router(config, channel_dim)

        # 转换为三通道的卷积层，用于将 ResNet 输出恢复为 3 通道
        self.channel_converter = nn.Conv2d(channel_dim, 3, kernel_size=1)

        self.add_batch_norm_before_adapter = True
        self.add_batch_norm_after_adapter = False
        if self.add_batch_norm_before_adapter:
            self.pre_batch_norm = nn.BatchNorm2d(channel_dim)
        if self.add_batch_norm_after_adapter:
            self.post_batch_norm = nn.BatchNorm2d(3)  # 转换后的通道数为 3

        if self.config.supervised_loss_weight != 0:
            self.supervised_loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """Retrieves the adapter layer corresponding to the given task."""
        # inputs (B,C,H,W)
        #print("控制层输入的形状1为：",inputs.shape)
        (batch_size, channel_dim, height, width) = inputs.shape
        z = self.pre_batch_norm(inputs) if self.add_batch_norm_before_adapter else inputs

        routing_estimator = self.config.routing_estimator
        load_loss = torch.tensor(0.0).cuda()
        supervised_loss = torch.tensor(0.0).cuda()
        #print("控制层输入的形状2为：",z.shape)


        # 计算 routing 的 logits 或者 adapter_probs
        if routing_estimator == "soft_input_routing":
            logits = self.multi_routers(z)#

        else:#修改了维度信息

            z_sent = z  # 不进行任何维度压缩，保留原始的 4D 形状
            if self.training and self.config.jitter_noise > 0:
                r1 = 1 - self.config.jitter_noise
                r2 = 1 + self.config.jitter_noise
                noise = (r1 - r2) * torch.rand(z_sent.shape).cuda() + r2
                z_sent = z_sent * noise
            # print("控制层输入的形状3为：", z_sent.shape)

            new_x, adapter_logits, adapter_probs = self.multi_routers(z_sent)   #获取数据，logits权重信息

            if self.training:
                if routing_estimator == 'gs_st_routing':
                    U = torch.rand(adapter_logits.shape).cuda()
                    y = adapter_logits + (-torch.log(-torch.log(U + 1e-20) + 1e-20))
                    y = F.softmax(y / self.config.adapter_temp, dim=-1)
                    probs, expert_index = y.max(dim=-1)
                    val = torch.ones_like(expert_index) - probs.detach() + probs
                else:
                    probs, expert_index = adapter_probs.max(dim=-1)
                    #print('probs:',probs)
                    #print('expert_index:',expert_index)

                if self.config.supervised_loss_weight != 0:
                    for router_index in range(self.config.num_routers):
                        supervised_loss += self.supervised_loss_fn(adapter_logits[router_index], labels)
            else:
                probs, expert_index = adapter_probs.max(dim=-1) #获取概率和索引

        # 转换高维特征为 3 通道
        z = self.channel_converter(z)  # 这里将输入转为 (B, 3, H, W)
        #print("控制层输入的形状4为：",z.shape)

        # 将转换后的三通道特征传递给专家模型
        if routing_estimator == 'parameter_averaging_routing':
            outputs = self.multi_adapters(z, adapter_probs,labels)
        elif routing_estimator == 'soft_routing':
            outputs = self.multi_adapters(z, adapter_probs)
        elif routing_estimator == "soft_input_routing":
            outputs = self.multi_adapters(z, logits)
        else:
            outputs = self.multi_adapters(z, expert_index)

        if self.config.analyze_model:
            self.config.analysis_list.append(expert_index)
            self.config.analysis_list.append(adapter_probs)

        if routing_estimator == 'gs_st_routing' and self.training:
            outputs = outputs * val[:, :, None, None]

        # print("控制层输入的形状5为：",outputs.shape)
        # multi_outputs = outputs + inputs
        # if self.add_batch_norm_after_adapter:
        #     multi_outputs = self.post_batch_norm(multi_outputs)

        if self.training:
            return outputs, load_loss, supervised_loss
        else:
            return outputs