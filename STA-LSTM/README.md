#  Spatio-Temporal Attention-Based LSTM
## STA-LSTM
利用了时-空注意力LSTM的输入进行加权从而进行预测
## VALSTM
变量注意力LSTM
## SLSTM
监督LSTM

## 测试集性能评估
 软测量模型  | RMSE  | R2
 ----- | ----- | ------  
STA-LSTM  | 0.0046 | 0.9994
 SLSTM  | 0.0415 | 0.9525
VA-LSTM  | 0.12087 | 0.80993
 
 ## 参考文献
 * [Deep learning with spatiotemporal attention-based LSTM for industrial soft sensor model development](https://ieeexplore.ieee.org/abstract/document/9062588/)  
 * [Dual Attention-Based Encoder–Decoder: A Customized Sequence-to-Sequence Learning for Soft Sensor Development](https://ieeexplore.ieee.org/document/9174767)
