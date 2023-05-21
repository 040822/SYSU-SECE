# ppo的改进措施

参考链接:<https://zhuanlan.zhihu.com/p/512327050>  
  

##已完成部分  
- Trick 6 Learning Rate Decay  
  使用adam优化器内置操作  
- Trick 7 Gradient clip
  原始代码已有
- Trick 9 Adam Optimizer Epsilon Parameter  
  已将a和c的adam优化器参数均改为eps=1e-5s
- Trick10 Tanh Activation Function  
  已将network.py中所有ReLU换成Tanh
  出现问题

