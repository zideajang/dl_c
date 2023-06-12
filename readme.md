## C 语言实现深度学习库

首先我们来创建一个 header 文件 `nn.h` 在这里包含了头文件和对于函数
声明实现部分
```c
#ifndef NN_H_
#define NN_H_

#endif //NN_H_
```

```c
// stb howtodo 库写的规范，下面是对头文件实现的部分
// 也就是类似 c body
#ifdef NN_IMPLEMENTATION

#endif //NN_IMPLEMENTATION



```
### Mat 结构体
定义数据结构体 Mat 有点类似其他框架的 Tensor，也是深度学习一切基础

```c
typedef struct 
{
    //Mat的 shape (rows,cols)
    size_t rows;
    size_t cols;
    float *data;
} Mat;
```

### 实现创建 Mat 的方法

对于 Mat 的构造方法，也就是分配给 Mat 一定大小内存空间来返回
一个指向这块内存的地址

```c
Mat mat_alloc(size_t rows, size_t cols);
```
对于 Mat 结构体构造主要是对于其中 data 进行分配内存空间用于存储数据

```c
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(sizeof(float))
}
```
如果这样定义来写入 `float` 就缺乏灵活性
```
m.data = malloc(sizeof(float))
```
如何这样定义就可以动态根据 Mat 中实际结构来计算出指针分配的内存大小
```c
m.data = malloc(sizeof(*m.data))
```

```c

int main(int argc, char const *argv[])
{   
    double a = 2;
    double* b = &a;
    printf("%d\n",sizeof(*b));
    return 0;
}

```

在 `nn.h` 头文件中判断宏 `NN_MALLOC `是否存在来
如果存在则使用用户在调用 `nn.h` 时自定义的内存分配方法
来代替默认 `malloc` 从而实现多态

```c

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif //NN_MALLOC
```

```c
#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif //NN_MALLOC


#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT
```

```c
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = NN_MALLOC(sizeof(*m.data)*rows*cols);
    NN_ASSERT(m.data != NULL);
    return m;
}
```

上面实现如何创建 Mat 以及在创建 Mat 是如何给 Mat 的 data 数据分配内存大小，接下来来实现打印 Mat 方法

### Mat 打印

```c
#include "nn.h"

int main(int argc, char const *argv[])
{
    Mat m = mat_alloc(2,2);
    mat_print(m);
    return 0;
}

```

定义结构体用来描述模型

```c
typedef struct 
{
    Mat a0;
    Mat w1,b1,a1;
    Mat w2,b2,a2;
    
} Xor;
```

结构体中每一行表示神经网络的一层，
`Mat a0` 表示输入层，这里每层输出都是用 `a` 来表示，`a0` 表示输入
`w1,b1,a1` 表示隐含层的参数，以及输出 `a1` 这里是对经过线性变换的进行一次 `sigmoid` 的非线性变换


#### 前向传播
```c

```

#### 计算成本函数
```c

float cost(Xor m, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);
    size_t n = ti.rows;
    
    float c = 0;
    for (size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);
        
        mat_copy(m.a0,x);
        forward(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(m.a2,0,j) - MAT_AT(y,0,j);
            c += d*d;
        }
       
    }

    return c/n;
}
```
- `ti` 表示数据输入样本也就是 x 样本特征
- `to` 表示数据的 ground truth 也就是样本标准答案
首先要做的是校验输入样本数量和 ground truth 是否相等，然后还要需要预测向量维度和 ground truth 样本维度相同

n 是样本的数量，然后循环每一个样本也就是从 `ti` 每一行都是样本，然后通过 `mat_copy` 函数将样本值赋值给 `m.a0` 计算前向传播，这里 `q` 模型输出向量的维度，我们计算每一个分量的差值求和后，

最后将 c 除以 n 来计算样本 loss 的均值。

```c
float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
```
在 `td` 每一行都是样本，每一行前两列为特征，后一列为 ground truth

```c
typedef struct 
{
    //Mat的 shape (rows,cols)
    size_t rows;
    size_t cols;
    size_t stride;
    float *data;
} Mat;
```
数据依然是连续地放置在内存上，那么我们为什么要 stride ，也就是设置应该如何地读取数据。

```c
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = NN_MALLOC(sizeof(*m.data)*rows*cols);
    NN_ASSERT(m.data != NULL);
    return m;
}
```

```c
Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows=1,
        .cols=m.cols,
        .stride = m.stride,
        .data = &MAT_AT(m,row,0),
    };
}
```
实际上对于一行数据 `stride` 是没有起任何作用的

```c
#define MAT_AT(m,i,j) (m).data[(i)*(m).stride + (j)]
```

```c
size_t stride = 3;
size_t n = sizeof(td)/sizeof(td[0])/3;
Mat ti = {
    .rows = n,
    .cols = 2,
    .stride = stride,
    .data = td
};
```

```c
size_t stride = 3;
size_t n = sizeof(td)/sizeof(td[0])/3;
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .data = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .data = td + 2,
    };

```

```c
[ ti
    0.000000    0.000000
    0.000000    1.000000
    1.000000    0.000000
    1.000000    1.000000
]
[ to
    0.000000
    1.000000
    1.000000
    0.000000
]
```