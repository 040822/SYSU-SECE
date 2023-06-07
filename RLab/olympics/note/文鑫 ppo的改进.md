# ppo的改进措施

参考链接:<https://zhuanlan.zhihu.com/p/512327050>  


#

## Advantage Actor-Critic(A2C) 
最初版本：n-step Advantage (李宏毅的Version 3.5)  
$Advantage(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$  
  

$Q^\pi(s_t,a_t) = r_t + \gamma*r_{t+1} + \gamma^2*r_{t+2}+...+\gamma^n*r_{t+n}+\gamma^{n+1}*V^\pi (s_{t+n+1})$  
  
其中V由critic网络估计  

  
其目标为优化奖励函数$J(\pi_\theta)$,其中$\nabla J(\pi_\theta) = E[A^\pi_t\nabla_\theta log\pi_\theta(a_t|s_t)]$,
即在动作的对数概率乘上A
  
#

## GAE of A2C
$A^\pi _{GAE} (s_t,a_t)=\Sigma(\gamma\lambda)^l * \delta_{t+l}$  
$\delta = r_t+\gamma*V(s_{t+1})-V(s_t)$


$\gamma,\lambda$均为超参数。$\lambda$和指数加权平均值有关，$A_(GAE)$被定义为n步前向回报的指数加权平均值。  
  
#

## PPO  
A2C算法是基于Policy Gradient（策略梯度算法）实现的，而PG面临的一个问题就是performance collapse(性能崩溃)，为解决这一问题，人们先后提出NPG、TRPO、CPO、PPO等算法，其中PPO便是比赛中所使用的算法。  
  在改进PG中，人们主要做了两项措施：1.修改目标函数 2.约束目标函数
    
### 修改目标函数  
首先，我们需要知道为什么会产生性能崩溃问题。  
在PG中，我们通过策略梯度$\nabla_{\theta}J(\pi_\theta)$来更新策略参数$\theta$，从而间接优化策略$\pi_\theta$。即$\theta_{t+1} = \theta_t + \alpha *\nabla_{\theta}L(\theta)$  
我们定义所有策略的集合为策略空间，每个策略$\pi_\theta$的参数$\theta$的集合为参数空间。显然，PG的参数更新是在参数空间中更新的，并通过从参数空间到策略空间的映射$\pi(\theta)$来间接更新策略。然而，这两个空间之间的映射并非一个良好的映射：在参数空间中非常小的更新可能在策略空间中是一个非常大的更新，反之，在参数空间中非常大的更新可能对策略几乎没有影响。这就造成两个问题：1，策略的更新太过缓慢 2.策略的更新太激烈，导致负优化，更差的新策略又会导致下一次的更新效果同样糟糕。  
因此，我们需要一种方法去避免过为缓慢的更新和过为激烈的更新，这种方法首先要做的就是去衡量更新前后的策略的性能差异，去寻找性能更好的新策略，并避免负优化现象。


已知强化学习的优化目标$J(\pi_\theta)$,那么如何判断新学习到的策略比原先的策略好多少呢？  
$J(\pi_{new})-J(\pi_{old})=E_{\tau - \pi_{new}}(\Sigma(\gamma^t*A^{\pi_{old}}(s_t,a_t)))$    (1)  
我们需要使这玩意大于等于0，不然就负优化了；并且每步优化中最合适的新的策略$\pi_{new}$就是使这个值最大的策略。    

(1)式的问题是，$E_{\tau - \pi_{new}}$需要基于新的策略，而我们需要(1)式才能找到最好的新的策略，这个矛盾解决的办法就是基于旧策略近似出一个新策略，近似后为：  
$J(\pi_{new})-J(\pi_{old})=E_{\tau - \pi_{new}}(\Sigma(\gamma^t*A^{\pi_{old}}(s_t,a_t)))\approx E_{\tau - \pi_{old}}[\Sigma(A^{\pi_{old}}(s_t,a_t))*\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}]=J^{CPI}_{\pi_{old}}(\pi_{new})$    (2)  
其中，  $J^{CPI}_{\pi_{old}}(\pi_{new})$被称为替代目标函数，CPI表示“保守的策略替代”，$\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}$表示重要性抽样权重。  
因为是近似值，所以说要衡量误差，这里使用$\pi$和$\pi_{new}$之间的KL来限制误差。  
$|J(\pi_{new})-J(\pi_{old})-J^{CPI}_{\pi_{old}}(\pi_{new})\ge C \sqrt{E_t[KL(\pi_{new}(a_t|s_t)||\pi_{old}(a_t|s_t))]}|$  
  
  不等号前面为绝对误差，后边即为KL散度。连续分布中，KL散度定义为:  
$KL(\pi_{new}(a_t|s_t)||\pi_{old}(a_t|s_t))=\int_{-\infty}^\infty \pi_{new}*log\frac{\pi_{new}}{\pi_{old}}dx$    
  
由于我们要确保每次更新后不能负优化，因此有$J(\pi_{new})-J(\pi_{old})\ge0$ ,即    
  
$J(\pi_{new})-J(\pi_{old})\ge J^{CPI}_{\pi_{old}}(\pi_{new})- C \sqrt{E_t[KL(\pi_{new}(a_t|s_t)||\pi_{old}(a_t|s_t))]}$   

  我们的优化目标便是找到一个$J(\pi_{new})$，使右项最大化。在实践中，为了达到这种最大化，人们提出信任域以限制KL散度。  
  ${E_t[KL(\pi_{new}(a_t|s_t)||\pi_{old}(a_t|s_t))]}\le \delta$   
  其中$\delta$ 是超参数。  
  NPG、TRPO、NPG等算法的提出均是为了解决信任域优化问题，但这些方法有很多缺点，比如理论复杂、算法实现难、梯度计算代价高、合适的$\delta$ 难以确定。
  

### PPO
  
下边到了PPO，这里定义$r_t(\theta)=\frac{\pi^{new}(a_t|s_t)}{\pi^{old}(a_t|s_t)}$  

PPO的理论和实现要比上边的算法简单很多（什么KL什么信任域都可以扔掉了），计算代价较低，且不需要选定$\delta$，因此PPO算法非常流行。

  PPO的两种常见变体分别是 1.自适应KL 2.PPO clip，其中第二种的效果更好，也是比赛代码中所使用的PPO，因此这里忽略第一种，直接介绍第二种。  
  
  PPO clip直接抛弃了KL约束而使用另一种更”简单“的方式去限制目标函数。  
  clip的基本思想是把$r_t(\theta)$限制于一个邻域$[1-\epsilon,1+\epsilon]$中，超出这个领域的值直接裁剪掉（即用邻域的上界或下界进行替换，可以类比ReLU）   
  $J' = clip(r_t(\theta),1-\epsilon,1+\epsilon)*A$

  而原始的目标函数是$J^{CPI}(\theta)=r_t(\theta)*A$  
  最终的目标函数为：
  $J^{CLIP}=E({min(J^{CPI},J')})$，即 将上面的两个目标函数取最小值然后取期望。
  

  双裁剪PPO：
  $J^{DC} = max(min() , c*A)$
  其中c>1, 以确定$r_t(\theta)$的下界值
  
    
  

## Trick  
- Trick 1—Advantage Normalization  
  分为两种，第一种是batch adv norm，对advantage作如下的Normalization
  $advantage_GAE = ((advantage_GAE - advantage_GAE.mean()) / (advantage_GAE.std() + 1e-5))  
  第二种是minibatch adv norm，仅对大batch中的一个小batch（minibatch）做normalization，效果相对第一种会变差，故采用第一种
  
- Trick 2—State Normalization
  核心为两块，一块是对state做normalization，一块是动态计算一组state的mean和std，减少计算量（即用n个state的mean和std和一个新的state，计算n+1个state的mean和std）  
  $\overline{x'} *n =(n-1)*\overline{x}+x'$  
  $\overline{x'} =((n-1)*\overline{x}+x')/n$  
    
  $s^{2'} = s^2 +(x-\overline{x'})+(x- \overline{x})$


- Trick 3 & Trick 4—— Reward Normalization & Reward Scaling  
  调整reward，防止reward过大或过小影响训练  
  Reward Normalization和State Normalization算法一致，并且通过同一个类实现  
  Reward Scaling是先计算accumulate decay Reward，然后再把它除以std  
  按照作者的说法，Reward Scaling效果很好，且使用Reward Scaling的同时使用Reward Normalization会导致问题，故仅采用Reward Scaling

- Trick 5—Policy Entropy  
  对actor的输出求熵，并在actor_loss中加上这个熵。  
  $H = -\Sigma_{a_t} \pi(a_t|s_t)*log(\pi(a_t|s_t))$  
  ` policy_entropy = -sum([action_prob[i] * torch.log(action_prob[i]) for i in range(len(action_prob))])`  
  `action_loss = -torch.min(surr1, surr2).mean()  - self.entropy_coef * policy_entropy`
   

- Trick 6—Learning Rate Decay  
  学习率衰减  
  `lr = lr * (1 - total_steps / max_train_steps)`

- Trick 7-Gradient clip   
  梯度裁剪，在pytorch中有内置的方法实现
 `torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) `  
 这一方法的原理是，将所有的梯度乘以一个clip_coef,其中：  
 $cip_coef = \frac{max-norm}{total-norm}$  
 max_norm是函数的第二个参数（0.5），total_norm是所有梯度的L2 norm（或者说是梯度向量的长度）  
   
- Trick 8—Orthogonal Initialization   
  完全没看懂  
    
    正交初始化（Orthogonal Initialization）是为了防止在训练开始时出现梯度消失、梯度爆炸等问题所提出的一种神经网络初始化方式。具体的方法分为两步：

  （1）用均值为0，标准差为1的高斯分布初始化权重矩阵，

  （2）对这个权重矩阵进行奇异值分解，得到两个正交矩阵，取其中之一作为该层神经网络的权重矩阵。

  使用正交初始化的Actor和Critic实现如下面的代码所示：

  注：
1. 我们一般在初始化actor网络的输出层时，会把gain设置成0.01，actor网络的其他层和critic网络都使用Pytorch中正交初始化默认的gain=1.0。
2. 在我们的实现中，actor网络的输出层只输出mean，同时采用nn.Parameter的方式来训练一个“状态独立”的log_std，这往往比直接让神经网络同时输出mean和std效果好。（之所以训练log_std，是为了保证std=exp(log_std)>0）  

- Trick 9—Adam Optimizer Epsilon Parameter   
  把eps从1e-8改成1e-5   

- Trick10—Tanh Activation Function  
  ReLU改成Tanh，效果存疑。
