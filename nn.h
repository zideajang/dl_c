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

// 分配内存空间
Mat mat_alloc(size_t rows, size_t cols);

void mat_dot(Mat dst, Mat a, Mat b); // dst = b@c
void mat_sum(Mat dst, Mat b); // dst = dst + b
void mat_print(Mat m);
#endif //NN_H_

// stb howtodo 库写的规范，下面是对头文件实现的部分
// 也就是类似 c body
#ifdef NN_IMPLEMENTATION
Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = NN_MALLOC(sizeof(*m.data)*rows*cols);
    NN_ASSERT(m.data != NULL);
    return m;
}
void mat_dot(Mat dst, Mat a, Mat b){
    (void) dst;
    (void) a;
    (void) b;

}
void mat_sum(Mat dst, Mat b){
    (void)dst;
    (void)b;
}
void mat_print(Mat m){
    printf("[ \n");
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f  ", MAT_AT(m,i,j));
        }

        printf("\n");
    }

    printf("]\n");
    
}

#endif //NN_IMPLEMENTATION