#include<stdlib.h>
#include<stdio.h>
#include "../include/nn.h"


NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;

    nn.ws =  NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1,arch[0]);

    for (size_t i = 1; i < arch_count; ++i)
    {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols,arch[i]);
        nn.bs[i-1] = mat_alloc(1,arch[i]);
        nn.as[i] = mat_alloc(1,arch[i]); 
    }
    

    return nn;
}

void nn_rand(NN nn, float low, float high){
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i],low,high);
        mat_rand(nn.bs[i],low,high);
    }
}



void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%d\n",nn.count);
    printf("%s = [\n", name);
    Mat *ws = nn.ws;
    Mat *bs = nn.bs;
    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        // MAT_PRINT(nn.ws[i]);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
        // MAT_PRINT(nn.bs[i]);
        // snprintf(buf, sizeof(buf), "bs%zu", i);
        // row_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}


void nn_forward(NN nn){
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.as[i + 1],nn.as[i],nn.ws[i]);
        mat_sum(nn.as[i + 1],nn.bs[i]);
        mat_sig(nn.as[i+1]);
    }
    
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        // X 和 ground truth
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);

        mat_copy(NN_INPUT(nn),x);
        nn_forward(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(y,0,j);
            c += d*d;
        }
        
    }
    // printf("c = %f\n",c);
    // printf("c = %d\n",n);
    return c/n;
}

void nn_finite_diff(NN nn, NN g, float eps,Mat ti,Mat to)
{
    float saved;
    // 计算 cost
    float c = nn_cost(nn,ti,to);
    // nn.ws:Mat[] 也就是存放每一层的神经的梯度
    // 每一个神经层是由若干神经元组成(y=wx+b)
    // {Mat[i,j]}
    for (size_t i = 0; i < nn.count; i++)
    {   
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                // 暂时将 weight 缓存起来
                saved = MAT_AT(nn.ws[i],j,k);
                // 调整(更新)参数
                MAT_AT(nn.ws[i],j,k) += eps;
                // 计算更新参数的梯度，也就是参数变得对 cost 影响程度
                MAT_AT(g.ws[i],j,k) = (nn_cost(nn,ti,to) - c)/eps;
                // 
                MAT_AT(nn.ws[i],j,k) = saved;
            }
            
        }
        
    }

    for (size_t i = 0; i < nn.count; i++)
    {   
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                // 暂时将 weight 缓存起来
                saved = MAT_AT(nn.bs[i],j,k);
                // 调整(更新)参数
                MAT_AT(nn.bs[i],j,k) += eps;
                // 计算更新参数的梯度，也就是参数变得对 cost 影响程度
                MAT_AT(g.bs[i],j,k) = (nn_cost(nn,ti,to) - c)/eps;
                // 
                MAT_AT(nn.bs[i],j,k) = saved;
            }
            
        }
        
    }
}

float nn_learn(NN nn, NN g,float rate)
{
    for (size_t i = 0; i < nn.count; i++)
    {   
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
               MAT_AT(nn.ws[i],j,k) -= rate* MAT_AT(g.ws[i],j,k);
            }
            
        }
        
    }

    for (size_t i = 0; i < nn.count; i++)
    {   
        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
               MAT_AT(nn.bs[i],j,k) -= rate* MAT_AT(g.bs[i],j,k);
            }
            
        }
        
    }
}



