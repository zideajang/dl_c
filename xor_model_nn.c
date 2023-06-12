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

Xor xor_alloc(void)
{    
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

    return m;

}

// 前向传播(也是推理过程)
float forward(Xor m){

    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);
    // MAT_PRINT(m.a1);

    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);
    // MAT_PRINT(m.a2);

}
// 计算 cost ，ti 表示输入 to 输出
// 输入是 (x1,x2) ti 矩阵每一个行都是 (x1,x2) 也就是
// 2 个维度的向量，输出为向量 (y) 所以 ti 和 to 需要具有
//相同行数，也就是每一个行表示一个样本(x1,x2) 对应 y
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

void finite_diff(Xor m, Xor g, float eps,Mat ti,Mat to)
{
    float saved;

    float c = cost(m,ti,to);

    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            saved = MAT_AT(m.w1,i,j);
            MAT_AT(m.w1,i,j) += eps;
            MAT_AT(g.w1,i,j) = (cost(m,ti,to) - c )/eps;
            MAT_AT(m.w1,i,j) = saved;
        }
        
    }
    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            saved = MAT_AT(m.b1,i,j);
            MAT_AT(m.b1,i,j) += eps;
            MAT_AT(g.b1,i,j) = (cost(m,ti,to) - c )/eps;
            MAT_AT(m.b1,i,j) = saved;
        }
        
    }
    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            saved = MAT_AT(m.w2,i,j);
            MAT_AT(m.w2,i,j) += eps;
            MAT_AT(g.w2,i,j) = (cost(m,ti,to) - c )/eps;
            MAT_AT(m.w2,i,j) = saved;
        }
        
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            saved = MAT_AT(m.b2,i,j);
            MAT_AT(m.b2,i,j) += eps;
            MAT_AT(g.b2,i,j) = (cost(m,ti,to) - c )/eps;
            MAT_AT(m.b2,i,j) = saved;
        }
        
    }
    
}

void xor_learn(Xor m, Xor g, float rate)
{
    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            MAT_AT(m.w1,i,j) -= rate* MAT_AT(g.w1,i,j);
        }
        
    }
    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            MAT_AT(m.b1,i,j) -= rate* MAT_AT(g.b1,i,j);
        }
        
    }
    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            MAT_AT(m.w2,i,j) -= rate* MAT_AT(g.w2,i,j);
        }
        
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            MAT_AT(m.b2,i,j) -= rate* MAT_AT(g.b2,i,j);
        }
        
    }
}

float td[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};


int main(int argc, char const *argv[])
{
    srand(time(0));
    size_t arch[] = {2,2,1};
    NN xor_nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN_PRINT(xor_nn);

    return 0;
    // 计算样本数量
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

    // MAT_PRINT(ti);

    Xor m = xor_alloc();
    Xor g = xor_alloc();
    // MAT_PRINT(m.w1);
    
    mat_rand(m.w1,0,1);
    mat_rand(m.b1,0,1);

    mat_rand(m.w2,0,1);
    mat_rand(m.b2,0,1);

    float eps =1e-1;
    float rate =1e-1;
    printf("cost = %f\n",cost(m,ti,to));
    for (size_t i = 0; i < 10*1000; i++)
    {
        finite_diff(m,g,eps,ti,to);
        xor_learn(m,g,rate);
        printf("%zu: cost = %f\n",i,cost(m,ti,to));
        /* code */
    }
    


    // float y = forward(m,0,1); 
    // printf("y = %f(predict) ", y);

    

#if 0
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            // [(0,0),(0,1)] = [i,j]
            MAT_AT(m.a0,0,0) = i;
            MAT_AT(m.a0,0,1) = j;
            forward(m);
            float y = *m.a2.data;
            printf("%zu ^ %zu = %f\n",i,j,y);
        }
        
    }
    
#endif
   

    return 0;
}
