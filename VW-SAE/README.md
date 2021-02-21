# Variable-Wise Weighted Stacked Auto-Encoder
## VW-SAE
采用了变量加权(加权由Pearson系数计算)的堆叠自编码器形式并针对脱丁烷精馏塔进行了对应的软测量
## SS-SAE
不采用变量加权的形式进行脱丁烷精馏塔做软测量(半监督形式)

## 测试集性能评估
 软测量类型  | RMSE  | R2
 ----- | ----- | ------  
 VW-SAE  | 0.03571 | 0.00467
 SS-SAE  | 0.04648 | 0.00565  
 
 ## 参考文献
 [Deep learning-based feature representation and its application for soft sensor modeling with variable-wise weighted SAE](https://ieeexplore.ieee.org/abstract/document/8302941)   
