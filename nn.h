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
    size_t stride;
    float *data;
} Mat;

#define MAT_AT(m,i,j) (m).data[(i)*(m).stride + (j)]
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
#define MAT_PRINT(m) mat_print(m,#m)
// 生成随机数
float rand_float();
float sigmoidf(float);

// 分配内存空间
Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low,float high);
Mat mat_row(Mat m,size_t row);
// Mat mat_sub(Mat m, size_t )
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat m,float b);
void mat_dot(Mat dst, Mat a, Mat b); // dst = b@c
void mat_sum(Mat dst, Mat b); // dst = dst + b
void mat_sig(Mat m);
void mat_print(Mat m,const char* name);

typedef struct 
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;//The amount of activations is count + 1
} NN;

NN nn_alloc(size_t* arch,size_t arch_count);
void nn_print(NN nn,const char *name);
#define NN_PRINT(nn) nn_print(nn,#nn)
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
    m.stride = cols;
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
    return (Mat){
        .rows=1,
        .cols=m.cols,
        .stride = m.stride,
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

NN nn_alloc(size_t* arch,size_t arch_count)
{   
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;
    // nn.ws 指向存储类型 Mat 数组，也就是每一个元素是 Mat 类型
    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1,arch[0]);
    for (size_t i = 1; i < arch_count; i++)
    {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols,arch[i]);
        nn.bs[i-1] = mat_alloc(1,arch[i]);
        nn.as[i]  = mat_alloc(1,arch[i]);
    }
    
    return nn;
}

void nn_print(NN nn, const char* name)
{
    printf("%s = [\n",name);
    for (size_t i = 0; i < nn.count; i++)
    {
        MAT_PRINT(nn.ws[i]);
        MAT_PRINT(nn.bs[i]);
    }

    printf("]\n");
    
}

#endif //NN_IMPLEMENTATION