# Gate Stacked Target-Related Stacked Auto Encoder
## T-SAE
在与训练的过程中,加入标签数据,将无监督预训练转化为有监督预训练问题。其中引入的超参数λ需要自行调整
## GS-TAE
利用门控逻辑单元对中间的抽象特征进行集成进而得到y的数据

## 测试集性能评估
 软测量模型  | RMSE  | R2
 ----- | ----- | ------  
 T-SAE  | 0.03571 | 0.9518
 GS-TAE  | 0.03046 | 0.9688  
 
 ## 参考文献
 * [Gated Stacked Target-Related Autoencoder: A Novel Deep Feature Extraction and Layerwise Ensemble Method for Industrial Soft Sensor Application](https://ieeexplore.ieee.org/abstract/document/9174659/)   
 * [基于自编码器的工业过程软测量建模方法研究](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1020072123.nh&v=vKAO1sAGlT%25mmd2BXXYkTwEe9uA%25mmd2FMEJVIsmY3qw1RpN7gFo%25mmd2BHmX3oM%25mmd2BuvG8UlyI42Bp4r)
