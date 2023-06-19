线性变换，因为线性变换不注意表达这个复杂的真实世界，确切地说不足以拟合这个真实世界，所以引入非线性函数，增加函数表达能力，或者说拟合能力

$$
y = f(x,\theta)
$$

已知 $(y,x)$ 去求 $\theta$,通过数据去求参数这就是模型学习过程，如何学习，学习就是根据梯度 $\frac{\partial}{\partial w}C$反馈来更新参数$\theta$

$$
\frac{1}{n}\sum_{i=1}^n(y - w*x)^2
$$

$$
w_i= w_{i-1} + \epsilon\\
dw = \frac{cost(w_{i}) - cost(w_{i-1})}{\epsilon}\\
w_i = rate w_{i_1}
$$

### 简单函数
$$
(x_i,y_i) i \in \{1,n\}
$$
$$
f(x_i) 
$$

$$
c = \frac{1}{n} \sum_{i=1}^n  (x_iw - y_i)^2 
$$
$$
C(w)^{\prime} = \left(\frac{1}{n} \sum_{i=1}^n  (x_iw - y_i)^2 \right)^{\prime}
$$

$$
C(w)^{\prime} = \frac{1}{n} \sum_{i=1}^n \left(  (x_iw - y_i)^2 \right)^{\prime}
$$

$$
g(f(x))^{\prime} = g(f(x))^{\prime} f(x)^{\prime}
$$


$$
C(w)^{\prime} = \frac{1}{n} \sum_{i=1}^n  2(x_iw - y_i)(x_iw - y_i)^{\prime} 
$$

$$
C(w)^{\prime} = \frac{1}{n} \sum_{i=1}^n  2x_i(x_iw - y_i)
$$

$$
C = \frac{1}{n} \sum_{i=1}^n \left( 
    \sigma(x_iw + b) - y_i
\right)^2
$$

$$
a_i = \sigma(x_iw + b)
$$

$$
C = \frac{1}{n} \sum_{i=1}^n \left( 
    \sigma(x_iw + b) - y_i
\right)^2
$$
$$
\frac{\partial}{\partial w}C = \frac{1}{n} \sum_{i=1}^n \left( 
    \sigma(x_iw + b) - y_i
\right)^2
$$

## 多层感知机梯度推导
$$
a_i^{(1)} =\sigma( x_iw^{(1)} + b^{(1)})
$$

$$
a_i^{(2)} =\sigma( a_i^{(1)}w^{(2)} + b^{(2)})
$$


这里小标 $i$ 表示第 $i$ 个样本，而上标表示神经网络的第几层

### 损失函数
$$
C = \frac{1}{n}\sum_{i=1}^n (a_i^{(2)} - y_i)^2
$$

成本函数就是计算模型输出(叫预测也好)和真实值之间差平方，然后计算所有样本和真正值的差值平方，之所以平方就是为了消除负数问题，我们只看差距不关心方向。然后再求均值

### 偏导数


$$
\frac{\partial}{\partial w^{(2)}} C = \frac{\partial}{\partial w^{(2)}} \left( \frac{1}{n}\sum_{i=1}^n (a_i^{(2)} - y_i)^2 \right)
$$
$$
\frac{\partial}{\partial b^{(2)}} C = \frac{\partial}{\partial b^{(2)}} \left( \frac{1}{n}\sum_{i=1}^n (a_i^{(2)} - y_i)^2 \right)
$$
化简梯度
$$
\frac{\partial}{\partial w^{(2)}} C = \frac{1}{n}\sum_{i=1}^n 2(a_i^{(2)} - y_i)\frac{\partial}{\partial w^{(2)}}a_i^{(2)}
$$

$$
\frac{\partial}{\partial w^{(2)}} C = \frac{1}{n}\sum_{i=1}^n 2(a_i^{(2)} - y_i)\frac{\partial}{\partial w^{(2)}}a_i^{(2)}
$$


$$
\frac{\partial}{\partial w^{(2)}}a_i^{(2)} = \frac{\partial}{\partial w^{(2)}} \left( 
    \sigma(  a_i^{(1)}w^{(2)} + b^{(2)})
\right)
$$

$$
\frac{\partial}{\partial w^{(2)}}a_i^{(2)} = 
(1- a_i^{(2)})a_i^{(1)}
$$


$$
\frac{\partial}{\partial w^{(2)}} C = \frac{1}{n}\sum_{i=1}^n 2(a_i^{(2)} - y_i)a_i^{(2)}(1- a_i^{(2)})a_i^{(1)}
$$

当计算第 2 层相对于权重 $w^{(2)}$ 会用到上一层输出 $a_i^{(1)}$ 损失函数就像一块大蛋糕，有点参数分少有的参数分的多，或者 cost 函数，也就是做这件事的成本，每一个参数都产生一定量成本，为了降低成本，我们找到每一个参数调整后参数量变化对成本变化影响程度来调整参数降低成本。

网络每一层是由若干神经元组成，每一个神经元 $\sigma(xw + b)$ ,多层堆叠形成结构化来学习一个结构上信息。

$$
f(x_i) = \sigma(\sigma(x_iw^{(1)} + b^{(1)})w^{(2)} + b^{(2)})
$$

$$
a_i^{(2)} =\sigma( a_i^{(1)}w^{(2)} + b^{(2)})
$$

$$
\frac{\partial}{\partial a^{(1)}_i }C
$$

### 反向传播

$$
\frac{\partial}{\partial w_^{(2)}} 
$$