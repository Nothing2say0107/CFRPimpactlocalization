'''
This program is just a related training program and is currently used to showcase the related work of the paper 
"Impact Localization in Composite Laminate with Retentive Network and Tree-Structured Bayesian Sensor Placement".
'''
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from retnet import RetNet
import matplotlib.pyplot as plt
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import init
num_batchsize = 8
device_count = torch.cuda.device_count()


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        # print('ltsdef', result.shape)
        output = self.conv1(result)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        # print('ltsdef',output.shape)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction).to("cuda")
        self.sa = SpatialAttention(kernel_size=kernel_size).to("cuda")
        self.li1 = nn.Linear(in_features=8*256, out_features=1000).to("cuda")
        self.li2 = nn.Linear(in_features=1000, out_features=2).to("cuda")
        self.fc = nn.Linear(in_features=6144 * 8, out_features=256 * 8).to("cuda")
        batch_size = 8
        hidden_size = 8
        sequence_length = 256
        heads = 2
        layers = 1
        ffn_size = 8
        self.retnet = RetNet(layers, hidden_size, ffn_size, heads, double_v_dim=True).to("cuda")

        # self.dropout = nn.Dropout(0.1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0,2,1).float().to("cuda")

        x = x.reshape(x.shape[0], -1)  
        x = self.fc(x)
        x = x.reshape(x.shape[0], 256, 8)

        '''retnet'''
        out = self.retnet(x)
        # print('out.shape', out.shape)
        '''KAN'''
        out = out.view(x.shape[0], -1)
        # print('out.shape', out.shape)
        out = self.li1(out).to("cuda")
        # out = self.dropout(out)
        out = self.li2(out).to("cuda")
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
# 模型实例化
rootdir_val = r'E:\...'
pattern = r'(\d+\.\d+-\d+\.\d+)'
# 创建模型实例
model = CBAMBlock(channel=1, reduction=1, kernel_size=8)

# 损失函数和优化器定义
criterion1 = nn.L1Loss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)  # 每7个epoch，学习率乘以0.1
# 训练数据
import pandas as pd
import numpy as np
import os

import re


def extract_number(s):
    match = re.search(r"(\d+\.\d+)", s)
    if match:
        return match.group(1)
    else:
        return None


result2_real = np.load('1.npy')[:,:-1,:]
result2 = result2_real

target2_real = np.load('1.npy')[:,-1,0:2]
target2 = target2_real



tenresult2 = torch.from_numpy(result2)
tentarget2 = torch.from_numpy(target2)
train_data = tenresult2[1:, :, :].to(device)
train_target = tentarget2[1:, :].to(device)

val_data = torch.zeros(1000, 8, 6144)
result2_val = np.zeros((1, 8, 6144))
target2_val = np.zeros((1, 2))
for xx, dirname, filenames in os.walk(rootdir_val):
    for filename in filenames:
        df = pd.read_csv(xx + '//' + filename)
        print(xx + '//' + filename)

        column_names = df.columns
        new_data = [extract_number(x) for x in np.array(column_names)]
        new_arr = [float(x) for x in new_data]
        new_arr = np.array(new_arr)
        new_arr = np.expand_dims(new_arr, axis=0)
        first_row = df.head(7)
        first_row = np.array(first_row)
        # print(new_arr.shape,first_row.shape)
        result_val = np.concatenate((new_arr, first_row))
        result_val = np.expand_dims(result_val, axis=0)
        result2_val = np.concatenate((result2_val, result_val))

        target_val = df.iloc[7][0:2]
        match = re.search(pattern, xx + '\\' + filename)
        name = match.group(1)
        parts = name.split('-')
        # 将分割后的字符串转换为浮点数
        float1 = float(parts[0])
        float2 = float(parts[1])
        target_val[0] = float1
        target_val[1] = float2

        target_val = np.array(target_val)
        # target_val = target_val.squeeze(0)
        target_val = np.expand_dims(target_val, axis=0)
        target2_val = np.concatenate((target2_val, target_val))

tenresult2_val = torch.from_numpy(result2_val)
tentarget2_val = torch.from_numpy(target2_val)
val_data = tenresult2_val[1:, :, :].to(device)
val_target = tentarget2_val[1:, :].to(device)
print('val_data.shape:', val_data.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_target),
                                           batch_size=num_batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_data, val_target), batch_size=num_batchsize,
                                         shuffle=True)

loss_values = []
loss_values_val = []

num_epochs = 1600
import time
for epoch in range(num_epochs + 1):
    starttime = time.time()

    print('epoch:', epoch)
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (data, target) in enumerate(train_loader):
        # print('train_data:',data.shape)
        output = model(data.to(device)).to(device)

        loss = criterion2(output, target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    # 计算平均损失和精度
    epoch_loss = running_loss / len(train_loader.dataset)
    loss_values.append(epoch_loss)
    if epoch % 200 == 0:
        torch.save(model.state_dict(), 'f8sensor_{}.pth'.format(epoch))
    print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    model.eval()  
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device)).to(device)
            loss = criterion2(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    epoch_loss_val = val_loss / len(val_loader.dataset)
    loss_values_val.append(epoch_loss_val)

    print('Epoch: {}/{}, VAL Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    print('time.with1epoch:',time.time()-starttime)

