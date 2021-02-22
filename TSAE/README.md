# Target Related Stacked Auto Encoder
## T-SAE
在与训练的过程中,加入标签数据,将无监督预训练转化为有监督预训练问题。其中引入的超参数λ需要自行调整
## SS-SAE
不采用变量加权的形式进行脱丁烷精馏塔做软测量(半监督形式)

## 测试集性能评估
 软测量类型  | RMSE  | R2
 ----- | ----- | ------  
 T-SAE  | 0.03571 | 0.9518
 SS-SAE  | 0.04648 | 0.9206  
 
 ## 参考文献
 [Gated Stacked Target-Related Autoencoder: A Novel Deep Feature Extraction and Layerwise Ensemble Method for Industrial Soft Sensor Application](https://ieeexplore.ieee.org/abstract/document/9174659/)   
