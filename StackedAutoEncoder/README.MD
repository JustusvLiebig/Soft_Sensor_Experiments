# Semi-Supervised Stacked Auto-Encoder 
## 半监督堆叠自编码器
算法的基本机制
* 无监督预训练  
堆叠自编码器逐层进行无监督预训练策略进行训练(layer-wise pretrain)
* 有监督微调  
加入标签数据,丢掉解码器,进行有监督微调

## 参考文献
[A novel semi-supervised pre-training strategy for deep networks and its application for quality variable prediction in industrial processes](https://www.sciencedirect.com/science/article/pii/S0009250920300415#!)