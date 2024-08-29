#include "../include/perceptron.h"

Perceptron perceptron_alloc(void){
    Perceptron m;
    
    // 输入层 {x1,x2}
    // [[0,0]] 输入层为 row = 1 cols = 2 的二维矩阵
    m.a0 = mat_alloc(1,2);
    
    // hidden layer [[0,0]] 1x2 [[w11,w12],[]] 2x2
    // sigmoid(Wx + b) 
    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);
    m.a1 = mat_alloc(1,2);

    // 暂时对于输出层也做激活函数
    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a2 = mat_alloc(1,1);

    // A ->(action) B  (action) -> C
    // 机器码 汇编 c/c++(23/17 )/rust/zig  java/js/csharp/python lib/frame appliction // requirement
    return m;
}

void finite_diff(Perceptron m,Perceptron g,float eps,Mat ti,Mat to)
{
    float saved;
    float c = perceptron_cost(m,ti,to);
    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            saved = MAT_AT(m.w1,i,j);
            MAT_AT(m.w1,i,j) += eps;
            MAT_AT(g.w1,i,j) = (perceptron_cost(m,ti,to) - c)/eps;
            MAT_AT(m.w1,i,j) = saved;
        }
        
    }

    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            saved = MAT_AT(m.b1,i,j);
            MAT_AT(m.b1,i,j) += eps;
            MAT_AT(g.b1,i,j) = (perceptron_cost(m,ti,to) - c)/eps;
            MAT_AT(m.b1,i,j) = saved;
        }
        
    }

    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            saved = MAT_AT(m.w2,i,j);
            MAT_AT(m.w2,i,j) += eps;
            MAT_AT(g.w2,i,j) = (perceptron_cost(m,ti,to) - c)/eps;
            MAT_AT(m.w2,i,j) = saved;
        }
        
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            saved = MAT_AT(m.b2,i,j);
            MAT_AT(m.b2,i,j) += eps;
            MAT_AT(g.b2,i,j) = (perceptron_cost(m,ti,to) - c)/eps;
            MAT_AT(m.b2,i,j) = saved;
        }
    }
    
    
}
float perceptron_forward(Perceptron m){
    
    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);

    // MAT_PRINT(m.a1);
    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);
}
// Agent c/c++ 提升一下
// 深度学习框架项目
// 线性代数库
// 图像库
float perceptron_cost(Perceptron m, Mat ti,Mat to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);
    size_t n = ti.rows;

    float c = 0;
    for(size_t i = 0; i < n; i++)
    {
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);

        mat_copy(m.a0,x);
        // y_hat
        perceptron_forward(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(m.a2,0,j) - MAT_AT(y,0,j);
            c += d*d;
        }
        

    }
}
float perceptron_train(Perceptron m,Perceptron g,float rate)
{
    for (size_t i = 0; i < m.w1.rows; i++)
    {
        for (size_t j = 0; j < m.w1.cols; j++)
        {
            MAT_AT(m.w1,i,j) -= rate * MAT_AT(g.w1,i,j);
        }
        
    }

    for (size_t i = 0; i < m.b1.rows; i++)
    {
        for (size_t j = 0; j < m.b1.cols; j++)
        {
            MAT_AT(m.b1,i,j) -= rate * MAT_AT(g.b1,i,j);
        }
        
    }

    for (size_t i = 0; i < m.w2.rows; i++)
    {
        for (size_t j = 0; j < m.w2.cols; j++)
        {
            MAT_AT(m.w2,i,j) -= rate * MAT_AT(g.w2,i,j);
        }
        
    }

    for (size_t i = 0; i < m.b2.rows; i++)
    {
        for (size_t j = 0; j < m.b2.cols; j++)
        {
            MAT_AT(m.b2,i,j) -= rate * MAT_AT(g.b2,i,j);
        }
        
    }
    
    
}
