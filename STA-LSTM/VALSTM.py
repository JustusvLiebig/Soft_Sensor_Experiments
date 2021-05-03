import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes, load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class Mydataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


class VALSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VALSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        self.Wa = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim * 2, input_dim), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)


        # 全连接层(做预测)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 权重初始化
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X, init_states=None):

        # 参数取得便于后续操作
        batch_size, seq_len, _ = X.size()

        # 参数命名
        x = X

        # 隐藏序列
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t, c_t = init_states
        # 序列长度的计算
        HS = self.hidden_dim

        t = 0
        # LSTM的计算
        while t < seq_len:

            # 取出当前的值
            x_t = x[:, t, :]

            # 计算注意力
            a_t = torch.tanh(x_t @ self.Wa + torch.cat((h_t, c_t), dim=1) @ self.Ua + self.ba) @ self.Va

            # softmax归一化
            alpha_t = self.Softmax(a_t)

            # 加权
            x_t = alpha_t * x_t

            # 计算门值
            gates = x_t @ self.W + h_t @ self.U + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS*2])
            g_t = torch.tanh(gates[:, HS*2:HS*3])
            o_t = torch.sigmoid(gates[:, HS*3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        final_feature = hidden_seq[:, seq_len-1, :].squeeze()



        final_feature = final_feature.view(batch_size, HS)

        # 全连接层做标签预测
        y_pred = self.fc(final_feature)


        return y_pred, hidden_seq, (h_t, c_t), alpha_t

class SLSTMModel(BaseEstimator, RegressorMixin):

    def __init__(self, dim_X, dim_y, dim_H, seq_len=30, n_epoch=200, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=1024):
        super(SLSTMModel, self).__init__()

        # 卡种子
        torch.manual_seed(seed)

        # 分配参数
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_H = dim_H
        self.seq_len = seq_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # 初始化模型
        self.loss_hist = []
        self.model = VALSTM(input_dim=dim_X, hidden_dim=dim_H, output_dim=dim_y).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        y = y.reshape(-1, self.dim_y)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i: i + self.seq_len, :])
            y_3d.append(y[i + self.seq_len - 1: i + self.seq_len, :])

        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = Mydataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.device), '3D')

        # 模型训练
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)
                batch_y = batch_y.squeeze(1)
                self.optimizer.zero_grad()
                output, _, _, _ = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self

    def predict(self, X, seq_len):

        # 转化为三维再预测
        X_3d = []

        for i in range(X.shape[0] - seq_len + 1):
            X_3d.append(X[i: i + seq_len, :])
        X_3d = np.stack(X_3d, 1)
        X = torch.tensor(X_3d, dtype=torch.float32, device=self.device).permute(1, 0, 2)

        self.model.eval()
        with torch.no_grad():

            y, _, _, alpha = self.model(X)

            # 放上cpu转为numpy
            y = y.cpu().numpy()
            alpha = alpha.cpu().numpy()

        return y, alpha

# 定义超参数
DIM_H = 60
SEQ_LEN = 20
EPOCH = 250
BATCH_SIZE = 90
LR = 0.014
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 1024

# 数据读取
data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')
data = data.values

TRAIN_SIZE = 1500

x_temp = data[:, 0:7]
y_temp = data[:, 7]

train_X = x_temp[:TRAIN_SIZE, :]
test_X = x_temp[TRAIN_SIZE-SEQ_LEN+1:, :]

y_train = y_temp[:TRAIN_SIZE]
y_test = y_temp[TRAIN_SIZE:]


train_y = y_train
test_y = y_test


mdl = SLSTMModel(dim_X=train_X.shape[1], dim_y=1, dim_H=DIM_H, seq_len=SEQ_LEN, n_epoch=EPOCH, batch_size=BATCH_SIZE, lr=LR, device=DEVICE, seed=SEED).fit(X=train_X, y=train_y)


y_pred, alpha = mdl.predict(X=test_X, seq_len=SEQ_LEN)


rmse = math.sqrt(mean_squared_error(test_y, y_pred))
print('\n测试集的MSE：', mean_squared_error(test_y, y_pred))
print('\n测试集的RMSE:', rmse)
print('\n测试集的相关系数：', r2_score(test_y, y_pred))

test_y = test_y.reshape(-1, 1)

# 存储预测结果与注意力
results = np.hstack((test_y, y_pred))
np.savetxt('VA_LSTM_DC.csv', results, delimiter=',')
np.savetxt('Attention_Value.csv', alpha, delimiter=',')