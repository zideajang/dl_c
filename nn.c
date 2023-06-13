#include<time.h>
#include<math.h>
#define NN_IMPLEMENTATION

#include<stdio.h>

#include<stdio.h>

// 可以实现并且定义宏 NN_MALLOC 就可以实现对原有方法的复写
// 从而实现多态
// #define NN_MALLOC my_malloc

#include "nn.h"




size_t arch[] = {2,2,1};
// NN nn = nn_alloc(arch, ARRAY_LEN(arch));



int main(int argc, char const *argv[])
{
    srand(time(0));

    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
   
    return 0;
}
