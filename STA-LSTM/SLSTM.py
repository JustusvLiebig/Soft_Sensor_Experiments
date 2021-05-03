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


class SLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 向量化
        self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4), requires_grad=True)
        # y的权重是在门值的计算上(4个门都有y)
        self.Wy = nn.Parameter(torch.Tensor(output_dim, hidden_dim * 4), requires_grad=True)

        # 全连接层(做预测)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 权重初始化
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, X, init_states=None):
        batch_size, seq_len, _ = X.size()
        # 切分输入与输出关系
        x = X[:, :, 0: self.input_dim]
        y = X[:, :, self.input_dim-1:]

        # 重新对齐tensor维度
        y = y.view(batch_size, seq_len, self.output_dim)


        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        else:
            h_t, c_t = init_states
        # 序列长度的计算
        HS = self.hidden_dim
        for t in range(seq_len):
            x_t = x[:, t, :]

            y_t = y[:, t, :]

            y_t = y_t.view(batch_size, -1)

            # LSTM门值的计算(y加进去算)
            gates = x_t @ self.W + h_t @ self.U + y_t @ self.Wy + self.bias

            i_t = torch.sigmoid(gates[:, :HS])
            f_t = torch.sigmoid(gates[:, HS:HS*2])
            g_t = torch.tanh(gates[:, HS*2:HS*3])
            o_t = torch.sigmoid(gates[:, HS*3:])

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        # 隐藏状态的计算
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        final_feature = hidden_seq[:, seq_len-1, :].squeeze()



        final_feature = final_feature.view(batch_size, HS)

        # 全连接层做特征预测

        y_pred = self.fc(final_feature)


        return y_pred, hidden_seq, (h_t, c_t)

class SLSTMModel(BaseEstimator, RegressorMixin):

    def __init__(self, dim_X, dim_y, dim_H, seq_len=30, n_epoch=200, batch_size=64, lr=0.001, device=torch.device('cuda:0'), seed=1024):
        super(SLSTMModel, self).__init__()

        # Set seed
        torch.manual_seed(seed)

        # Parameter assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.dim_H = dim_H
        self.seq_len = seq_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        # Initialize Scaler
        self.scaler_X = StandardScaler()


        # Model Instantiation
        self.loss_hist = []
        self.model = SLSTM(input_dim=dim_X, hidden_dim=dim_H, output_dim=dim_y).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)

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

        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle = True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(0, 1, 2)
                batch_y = batch_y.permute(0, 1, 2)

                batch_y = batch_y.squeeze(1)

                self.optimizer.zero_grad()

                output, _, _ = self.model(batch_X)



                loss = self.criterion(output, batch_y)

                self.loss_hist[-1] += loss.item()

                loss.backward()

                self.optimizer.step()

            print('Epoch:{}, Loss:{}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished')

        return self

    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.device).unsqueeze(1)

        self.model.eval()
        with torch.no_grad():

            y, _, _ = self.model(X)

            y = y.cpu().numpy()

        return y


data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')


# print(data)
# print(data.shape)
data = data.values

TRAIN_SIZE = 1556

x_temp = data[:, 0:7]
y_temp = data[:, 7]

x_new = np.zeros([2394, 8])
x_new[:, :7] = x_temp

y_train = y_temp[:TRAIN_SIZE]
y_test_init = y_temp[TRAIN_SIZE-1:-1]


y_test = y_temp[TRAIN_SIZE:]

x_new[:TRAIN_SIZE, 7] = y_train
x_new[TRAIN_SIZE:, 7] = y_test_init




train_X = x_new[:TRAIN_SIZE, :]
train_y = y_train

test_X = x_new[TRAIN_SIZE:, :]
test_y = y_test


mdl = SLSTMModel(dim_X=data.shape[1], dim_y=1, dim_H=90, seq_len=10, n_epoch=55, batch_size=64, lr=0.020, device=torch.device('cuda:0'), seed=1024).fit(X=train_X, y=train_y)

y_pred = mdl.predict(test_X)
rmse = math.sqrt(mean_squared_error(test_y, y_pred))
print('\n测试集的MSE：', mean_squared_error(test_y, y_pred))
print('\n测试集的RMSE:', rmse)
print('\n测试集的相关系数：', r2_score(test_y, y_pred))


test_y = test_y.reshape(-1, 1)

results = np.hstack((test_y, y_pred))
np.savetxt('SLSTM_DC.csv', results, delimiter=',')