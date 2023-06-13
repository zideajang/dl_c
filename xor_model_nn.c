#include<time.h>
#include<math.h>
#define NN_IMPLEMENTATION

#include<stdio.h>

#include "nn.h"

// 数据
float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};


int main(int argc, char const *argv[])
{
    srand(time(0));


    // 将数据拆分 x(sample) 和 y(ground truth)
    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/3;
    // sample(input)
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .data = td
    };
    // groud truth
    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .data = td + 2,
    };

    // 定义网络结构
    size_t arch[] = {2,2,1};
    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g = nn_alloc(arch,ARRAY_LEN(arch));

    float eps =1e-1;
    float rate =1e-1;

    // 初始化参数
    nn_rand(nn,0,1);
    // MAT_PRINT(mat_row(ti,1));
    Mat row = mat_row(ti,3);
    // MAT_PRINT(row);
    mat_copy(NN_INPUT(nn),row);

    // nn_forward(nn);
    printf("cost: %f",nn_cost(nn,ti,to));
    // 训练网络更新参数
    for (size_t i = 0; i < 100*1000; i++)
    {
        nn_finite_diff(nn,g,eps,ti,to);
        nn_learn(nn,g,rate);
        printf("%zu: cost = %f\n",i,nn_cost(nn,ti,to));
        /* code */
    }

#if 1
    // 验证模型(推理)
    printf("inference\n");
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            // [(0,0),(0,1)] = [i,j]
            MAT_AT(NN_INPUT(nn),0,0) = i;
            MAT_AT(NN_INPUT(nn),0,1) = j;
            nn_forward(nn);
            float y = MAT_AT(NN_OUTPUT(nn),0,0);
            printf("%zu ^ %zu = %f\n",i,j,y);
        }
        
    }
    
#endif
   

    return 0;
}
