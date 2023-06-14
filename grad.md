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