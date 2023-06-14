#include<stdio.h>
#include<time.h>
#include<math.h>
#define NN_IMPLEMENTATION

#include"nn.h"

// y = 3x
// 准备数据 (x,y)
float train_data[][2] = {
    {0,0},
    {1,3},
    {2,6},
    {3,9},
    {5,15},
};

#define train_count (sizeof(train_data)/sizeof(train_data[0]))
float cost(float w)
{
    
    float result = 0.0f;
    float n = train_count;
    for (size_t i = 0; i < n; i++)
    {   
        float x = train_data[i][0];
        float d  = train_data[i][1] -  w*x;
        result += d*d;
    }

     result /= n;
     return result;
}

float dcost(float w)
{
    float result = 0.0f;
    float n = train_count;
    for (size_t i = 0; i < n; i++)
    {   
        float x = train_data[i][0];
        float y  = train_data[i][1];
        result += 2*x*(w*x-y);
    }

     result /= n;
     return result;


}

int main(int argc, char const *argv[])
{
    srand(time(0));
    float w = rand_float()*10.0f;
    printf(" cost = %f",cost(w));
    float rate = 1e-1;
    
    
    for (size_t i = 0; i < 100; i++)
    {
    #if 0    
        float eps = 1e-1;
        float c = cost(w);
        float dw = (cost(w + eps) - c)/eps;

    #else
        float dw = dcost(w);
    #endif
        w -= rate*dw;
        printf("%zu: cost = %f\n",i,cost(w));
    }

    printf("w = %f\n",w);

    return 0;
}
