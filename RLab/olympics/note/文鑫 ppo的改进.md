# ppo的改进措施

参考链接:<https://zhuanlan.zhihu.com/p/512327050>  
  

## 已完成部分  
- Trick 6 Learning Rate Decay  
  使用adam优化器内置操作  
- Trick 7 Gradient clip
  原始代码已有
- Trick 9 Adam Optimizer Epsilon Parameter  
  已将a和c的adam优化器参数均改为eps=1e-5s
- Trick10 Tanh Activation Function  
  已将network.py中所有ReLU换成Tanh
  出现问题


## Advangtage Actor-Critic(A2C) 
$Advangtage(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$  
  

$Q^\pi(s_t,a_t) = r_t + \gamma*r_{t+1} + \gamma^2*r_{t+2}+...+\gamma^n*r_{t+n}+\gamma^{n+1}*V^\pi (s_{t+n+1})$  
  
其中V由critic网络估计  

  
其目标为优化奖励函数$J(\pi_\theta)$,其中$\nabla J(\pi_\theta) = E[A^\pi_t\nabla_\theta log\pi_\theta(a_t|s_t)]$,
即在动作的对数概率乘上A
  
## GAE of A2C
$A^\pi _{GAE} (s_t,a_t)=\Sigma(\gamma\lambda)^l * \delta_{t+l}$  
$\delta = r_t+\gamma*V(s_{t+1})-V(s_t)$
  

$\gamma,\lambda$均为超参数。$\lambda$和指数加权平均值有关，$A_(GAE)$被定义为n步前向回报的指数加权平均值。  
  
## PPO  
已知强化学习的优化目标$J(\pi_\theta)$,那么如何判断新学习到的策略比原先的策略好多少呢？  
$J(\pi_{new})-J(\pi_{old})=E_{\tau - \pi_{new}}(\Sigma(\gamma^t*A^{\pi_{old}}(s_t,a_t)))$    (1)  
我们需要使这玩意大于等于0，不然就负优化了；并且每步优化中最合适的新的策略$\pi_{new}$就是使这个值最大的策略。  
(1)式的问题是，$E_{\tau - \pi_{new}}$需要基于新的策略，而我们需要(1)式才能找到最好的新的策略，这个矛盾解决的办法就是基于旧策略近似出一个新策略，近似后为：  
$J(\pi_{new})-J(\pi_{old})=E_{\tau - \pi_{new}}(\Sigma(\gamma^t*A^{\pi_{old}}(s_t,a_t)))\approx E_{\tau - \pi_{old}}[\Sigma(A^{\pi_{old}}(s_t,a_t))*\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}]=J^{CPI}_{\pi_{old}}(\pi_{new})$    (2)  
其中，  $J^{CPI}_{\pi_{old}}(\pi_{new})$被称为替代目标函数，CPI表示“保守的策略替代”，$\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}$表示重要性抽样权重。  
因为是近似值，所以说要衡量误差（这里暂略）。  
  
下边到了PPO，这里定义$r_t(\theta)=\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}$