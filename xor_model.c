#include<time.h>
#include<math.h>
#define NN_IMPLEMENTATION

#include<stdio.h>

#include "nn.h"

// Xor 输入{x1,x2} 输出为 {y}
// {1,0} = {1} {0,1} = {1} {0,0} = 0 {1,1} = 1
// w11 w12 w21 w22 b1 b2 输入{x1,x2} 输出为 {a1,a2}



typedef struct 
{
    Mat a0;

    Mat w1,b1,a1;
    // 输出层为 y
    Mat w2,b2,a2;
    
} Xor;

// 前向传播(也是推理过程)
float forward(Xor m, float x1, float x2){

    MAT_AT(m.a0,0,0) = x1;
    MAT_AT(m.a0,0,1) = x2;

    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);
    // MAT_PRINT(m.a1);

    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);
    // MAT_PRINT(m.a2);

    return *m.a2.data;
}

int main(int argc, char const *argv[])
{
    srand(time(0));
    Xor m;
    // 输入层{x1,x2}

    m.a0 = mat_alloc(1,2);

    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);
    m.a1 = mat_alloc(1,2);
    // 输出层为 y
    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a2 = mat_alloc(1,1);
    
    mat_rand(m.w1,0,1);
    mat_rand(m.b1,0,1);

    mat_rand(m.w2,0,1);
    mat_rand(m.b2,0,1);

    // float y = forward(m,0,1); 
    // printf("y = %f(predict) ", y);

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            printf("%zu ^ %zu = %f\n",i,j,forward(m,i,j));
        }
        
    }
    

   

    return 0;
}
