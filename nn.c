#define NN_IMPLEMENTATION

#include<stdio.h>

// 可以实现并且定义宏 NN_MALLOC 就可以实现对原有方法的复写
// 从而实现多态
// #define NN_MALLOC my_malloc

#include "nn.h"

int main(int argc, char const *argv[])
{
    Mat m = mat_alloc(2,2);
    mat_print(m);
    return 0;
}
