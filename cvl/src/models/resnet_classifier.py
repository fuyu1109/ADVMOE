import os
import math
import random
import torch
import torch.nn as nn
import re
import numpy as np
import ipdb
from  resnet.modeling_resnet import resnet18, resnet50
import torch.nn.functional as F


class ResNet_Classifier(torch.nn.Module):
    def __init__(self, config, device, clean_model=None, adv_model=None):
        '''
        :param device:
        '''
        super(ResNet_Classifier, self).__init__()
        self.config = config
        self.device = device

        if self.config.pretrained_weight == 'resnet50':
            self.model = resnet50(pretrained=True, config=self.config, clean_model=clean_model, adv_model=adv_model)
        elif self.config.pretrained_weight == 'resnet18':
            self.model = resnet18(pretrained=True, config=self.config, clean_model=clean_model, adv_model=adv_model)

        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.huber_loss = nn.HuberLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def get_loss(self, images, labels):
        '''
        Get classification loss

        :param batch:
        :return:
        '''


        # 使用模型进行前向传播，仅传递图像
        label_logits = self.model(images,labels)

        # 计算交叉熵损失
        label_loss = self.loss(label_logits, labels)

        return label_loss, label_logits

    def forward(self, images, labels=None):
        '''
        Compute loss

        :param batch:
        '''
        # 自适应模块的初始化
        if labels is None:
            raise ValueError("Missing labels for loss computation")

        if self.config.train_adapters:
            self.config.load_loss_accm = torch.tensor(0.0).cuda()
            self.config.supervised_loss_accm = torch.tensor(0.0).cuda()

        # 计算损失和 logits
        label_loss, label_logits = self.get_loss(images, labels)

        if self.config.probe_input_features:
            loss = torch.tensor(0).cuda()
        else:
            loss = torch.mean(label_loss)

        # 如果启用自适应路由
        if self.training and self.config.train_adapters:
            dict_val = {"label_loss": loss, "load_loss": self.config.load_loss_weight * self.config.load_loss_accm,
                            "supervised_loss": self.config.supervised_loss_weight * self.config.supervised_loss_accm}
            loss = loss + self.config.load_loss_weight * self.config.load_loss_accm + self.config.supervised_loss_weight * self.config.supervised_loss_accm
            dict_val['loss'] = loss

        else:
            dict_val = {'loss': loss}

        return loss

    def predict(self, images, labels=None):
        '''
        Predict the lbl for batch

        :param batch:
        '''
        _, label_logits = self.get_loss(images, labels)
        lbl_prob = torch.softmax(label_logits, dim=-1)
        return torch.argmax(lbl_prob, dim=-1)
        #return torch.argmax(lbl_prob, dim=-1), lbl_prob


