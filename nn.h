#ifndef NN_H_
#define NN_H_

#include<stddef.h>
#include<stdio.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif //NN_MALLOC

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT


// Mat 结构体，tensor 这样结构是 DL 是一切开始
typedef struct 
{
    //Mat的 shape (rows,cols)
    size_t rows;
    size_t cols;
    float *data;
} Mat;

#define MAT_AT(m,i,j) (m).data[(i)*(m).cols + (j)]
#define MAT_PRINT(m) mat_print(m,#m)
// 生成随机数
float rand_float();
float sigmoidf(float);

// 分配内存空间
Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low,float high);
Mat mat_row(Mat m,size_t row);
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat m,float b);
void mat_dot(Mat dst, Mat a, Mat b); // dst = b@c
void mat_sum(Mat dst, Mat b); // dst = dst + b
void mat_sig(Mat m);
void mat_print(Mat m,const char* name);
#endif //NN_H_

// stb howtodo 库写的规范，下面是对头文件实现的部分
// 也就是类似 c body
#ifdef NN_IMPLEMENTATION

// 
float rand_float(void){
    return (float)rand()/(float)RAND_MAX;
}

// 
float sigmoidf(float x)
{
    return 1.0f /(1.0f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = NN_MALLOC(sizeof(*m.data)*rows*cols);
    NN_ASSERT(m.data != NULL);
    return m;
}
void mat_dot(Mat dst, Mat a, Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j)  = 0;
            // printf("(%d,%d) = ",i,j );
            for (size_t k = 0; k < n; ++k)
            {   
                // printf("k = %d\t",k);
                // if(k==n){
                //     printf("%f * %f  ",MAT_AT(a,i,k),MAT_AT(b,k,j));
                // }else{
                //     printf("%f * %f + ",MAT_AT(a,i,k),MAT_AT(b,k,j));
                // }
                MAT_AT(dst,i,j) += MAT_AT(a,i,k)*MAT_AT(b,k,j);
            }
            // printf("%f\n",MAT_AT(dst,i,j));
            
        }
        
    }
    
}
void mat_sum(Mat dst, Mat a){
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) = MAT_AT(dst,i,j)  + MAT_AT(a,i,j);
        }
        
    }
    
}

// 对于 Mat 随机初始化
void mat_rand(Mat m,float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = rand_float()*(high - low) + low;
        }

    }
}

Mat mat_row(Mat m, size_t row)
{
    // MAT_AT 返回 Mat某一个元素
    // 当我们返回的是某一个行第一个元素地址
    // 也就是拿到了那一行数据
    return (Mat){
        .rows=1,
        .cols=m.cols,
        .data = &MAT_AT(m,row,0),
    };
}

void mat_copy(Mat dst, Mat src){
    // 首先需要校验 dst 和 src 是否
    // 具有相同 shape，只有具有相同 shape
    // 才能进行 copy
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) = MAT_AT(src,i,j);
        }
        
    }
    
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {

        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
        
    }
    
}

void mat_print(Mat m,const char* name){
    printf("[ %s\n",name);
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("    %f", (float)MAT_AT(m,i,j));
        }

        printf("\n");
    }

    printf("]\n");
    
}

void mat_fill(Mat m,float b)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = b;
        }
    }
}

#endif //NN_IMPLEMENTATION