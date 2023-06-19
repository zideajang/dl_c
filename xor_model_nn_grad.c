#include<time.h>
#include<math.h>
#define NN_IMPLEMENTATION

#include<stdio.h>

#include "nn.h"

// 数据
float td_xor[][3] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};

float td_or[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,1,
};

// float* train = td_xor;
// 样本数量
size_t train_count = 4;

float cost(float w1, float w2, float b){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = td_xor[i][0];
        float x2 = td_xor[i][1];
        float y = sigmoidf(w1*x1  + w2 * x2 + b);
        float d = y - td_xor[i][2];
        result += d*d;
    }
    result /=train_count;
    // printf("result = %f\n",result);
    return result;

}
// 分别计算 w1 w2 b1 对于 loss 偏导数
float gcost(float w1,float w2,float b, 
    float *dw1, float *dw2, float *db)
{
    // 初始化梯度
    *dw1 = 0.0f;
    *dw2 = 0.0f;
    *db = 0.0f;

    size_t n = train_count;

    for (size_t i = 0; i < n; i++)
    {
        float x1 =  td_xor[i][0];
        float x2 =  td_xor[i][1];
        float y =   td_xor[i][2];

        float a = sigmoidf(w1*x1 + w2*x2 + b);

        float d = 2*(a - y)*a*(1-a);

        *dw1 += d*x1;
        *dw2 += d*x2;
        *db += d;
    }
    *dw1 /= n;
    *dw2 /= n;
    *db  /= n;
    
}

int main(int argc, char const *argv[])
{
    srand(time(0));
    // 初始化参数
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();

    float c = cost(w1,w2,b);
    printf("cost %f: w1 = %f, w2 = %f, b = %f \n",w1,w2,b,c);
    float rate = 1e-1;
    
    // 训练阶段
    for (size_t i = 0; i < 100; i++){
        float eps = 1e-1;

        float dw1,dw2,db;


        #if 0

        dw1 = (cost(w1 + eps,w2,b) - c)/eps;
        dw2 = (cost(w1,w2 + eps,b) - c)/eps;
        db = (cost(w1,w2 ,b + eps) - c)/eps;

        // printf("dw1 = %f\n",dw1);

        #else

        gcost(w1,w2,b,&dw1,&dw2,&db);
        #endif
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        b -= rate * db;
    }
    printf("cost %f: w1 = %f, w2 = %f, b = %f \n",w1,w2,b,c);


    // 推理阶段
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu | %zu = %f\n",i,j,sigmoidf(w1*1 + w2*j + b));
        }
        
    }
    

   

    return 0;
}
