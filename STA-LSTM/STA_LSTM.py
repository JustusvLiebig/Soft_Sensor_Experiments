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

# 时间注意力
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim):
        super(TemporalLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, seq_len), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim , seq_len), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(seq_len), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(seq_len, seq_len), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        # LSTM参数
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)
        # y的权重是在门值的计算上(4个门都有y)
        self.Wy = nn.Parameter(torch.Tensor(output_dim, hidden_dim * 4), requires_grad=True)

        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, H, y_t_1):

        # 参数取得便于后续操作
        batch_size, seq_len, input_dim = H.size()
        # 序列长度的计算
        HS = self.hidden_dim

        # 参数命名
        h = H

        # 隐藏序列
        hidden_seq = []

        # 初始状态
        s_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_h_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)
        LSTM_c_t = torch.zeros(batch_size, self.hidden_dim).to(h.device)

        # 打循环开始
        t = 0
        # 注意力机制的计算
        while t < seq_len:
            # 取出当前的值
            h_t = h[:, t, :]

            # 计算注意力(第二个维度对应了是时间序列长度)
            beta_t = torch.tanh(h_t @ self.Wa + s_t @ self.Ua + self.ba) @ self.Va

            # softmax过一次
            beta_t = self.Softmax(beta_t)
            # 扩充对齐inpupt_dim维度(重复之后直接做哈达玛积运算)
            beta_t = beta_t.unsqueeze(2)
            beta_t = beta_t.repeat(1, 1, input_dim)

            # 合并掉时间序列的维度(全序列)
            h_t = torch.sum(input=beta_t * h, dim=1)

            # LSTM门值的计算(y加进去算)
            gates = h_t @ self.W + LSTM_h_t @ self.U + y_t_1 @ self.Wy + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS * 2])
            g_t = torch.tanh(gates[:, HS * 2:HS * 3])
            o_t = torch.sigmoid(gates[:, HS * 3:])

            # 隐藏层状态的计算
            LSTM_c_t = f_t * LSTM_c_t + i_t * g_t
            LSTM_h_t = o_t * torch.tanh(LSTM_c_t)
            hidden_seq.append(LSTM_h_t.unsqueeze(0))

            y_t_1 = self.fc(LSTM_h_t)

            # 时刻加一
            t = t + 1
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return y_t_1, hidden_seq

# 空间注意力
class SpatialLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialLSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)

        # 注意力的参数
        self.Wa = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(hidden_dim * 2, input_dim), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        # 权重初始化
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X):

        # 参数取得便于后续操作
        batch_size, seq_len, _ = X.size()

        # 参数命名
        x = X

        # 隐藏序列
        hidden_seq = []

        # 初始值计算
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 序列长度的计算
        HS = self.hidden_dim

        # 打循环开始
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

        return hidden_seq, (h_t, c_t), alpha_t

# 时空注意力
class STA_LSTM(nn.Module):
    def __init__(self, input_dim, sa_hidden, ta_hidden, seq_length, output_dim):
        super(STA_LSTM, self).__init__()

        # 超参数继承
        self.input_dim = input_dim
        self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = seq_length
        self.output_dim = output_dim

        # 预测模型
        self.SA = SpatialLSTM(input_dim=input_dim, hidden_dim=sa_hidden)
        self.TA = TemporalLSTM(input_dim=sa_hidden, hidden_dim=ta_hidden, seq_len=seq_length, output_dim=output_dim)

    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        # 切分输入与输出关系
        x = X[:, :, 0: self.input_dim]
        y = X[:, :, self.input_dim - 1:]

        # 重新对齐tensor维度
        y = y.view(batch_size, seq_len, self.output_dim)

        # 参数的预测
        hidden_seq, (_, _), _ = self.SA(X=x)
        y_pred, _ = self.TA(H=hidden_seq, y_t_1=y[:, 0, :])

        return y_pred

class STALSTMModel(BaseEstimator, RegressorMixin):

    def __init__(self, input_dim, sa_hidden, ta_hidden, seq_length, output_dim, n_epoch=200, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=1024):
        super(STALSTMModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.input_dim = input_dim
        self.sa_hidden = sa_hidden
        self.ta_hidden = ta_hidden
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # Initialize Scaler
        self.scaler_X = StandardScaler()


        # Model Instantiation
        self.loss_hist = []
        self.model = STA_LSTM(input_dim=input_dim, sa_hidden=sa_hidden, ta_hidden=ta_hidden, seq_length=seq_length, output_dim=output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)

        y = y.reshape(-1, self.output_dim)


        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_length + 1):
            X_3d.append(X[i: i + self.seq_length, :])
            y_3d.append(y[i + self.seq_length - 1: i + self.seq_length, :])

        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = Mydataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.device), '3D')

        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)

                batch_y = batch_y.squeeze(1)

                self.optimizer.zero_grad()

                output = self.model(batch_X)



                loss = self.criterion(output, batch_y)

                self.loss_hist[-1] += loss.item()

                loss.backward()

                self.optimizer.step()

            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self


    def predict(self, X, seq_length):

        X = self.scaler_X.transform(X)

        # 转化为三维再预测
        X_3d = []

        for i in range(X.shape[0] - seq_length + 1):
            X_3d.append(X[i: i + seq_length, :])
        X_3d = np.stack(X_3d, 1)
        X = torch.tensor(X_3d, dtype=torch.float32, device=self.device).permute(1, 0, 2)

        self.model.eval()
        with torch.no_grad():

            y = self.model(X)

            # 放上cpu转为numpy
            y = y.cpu().numpy()

        return y


SEQ_LEN = 4

data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')

data = data.values

TRAIN_SIZE = 1556

x_temp = data[:, 0:7]
y_temp = data[:, 7]

x_new = np.zeros([data.shape[0], 8])
x_new[:, :7] = x_temp
x_new[:, 7] = y_temp



train_X = x_new[:TRAIN_SIZE, :]
y_train = y_temp[:TRAIN_SIZE]
train_y = y_train

test_X = x_new[TRAIN_SIZE-SEQ_LEN+1:, :]
y_test = y_temp[TRAIN_SIZE:]
test_y = y_test


mdl = STALSTMModel(input_dim=data.shape[1], sa_hidden=60, ta_hidden=60, seq_length=SEQ_LEN, output_dim=1, n_epoch=100, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=1024).fit(X=train_X, y=train_y)

y_pred = mdl.predict(test_X, seq_length=SEQ_LEN)
rmse = math.sqrt(mean_squared_error(test_y, y_pred))
print('\n测试集的MSE：', mean_squared_error(test_y, y_pred))
print('\n测试集的RMSE:', rmse)
print('\n测试集的相关系数：', r2_score(test_y, y_pred))



