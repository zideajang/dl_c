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
