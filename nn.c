#include<time.h>

#define NN_IMPLEMENTATION

#include<stdio.h>

// 可以实现并且定义宏 NN_MALLOC 就可以实现对原有方法的复写
// 从而实现多态
// #define NN_MALLOC my_malloc

#include "nn.h"

int main(int argc, char const *argv[])
{
    srand(time(0));
    Mat m1 = mat_alloc(2,2);
    mat_fill(m1,1.0f);
    mat_print(m1);
    printf("---------------------");
    Mat m2 = mat_alloc(2,2);
    mat_fill(m2,2.0f);
    mat_print(m2);
    printf("---------------------");
    mat_sum(m1,m2);
    mat_print(m1);
    return 0;
}
