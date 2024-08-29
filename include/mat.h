#ifndef MAT_H_
#define MAT_H_

#include<stddef.h>
#include<stdio.h>
#include <stdint.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC


#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif //NN_ASSERT

typedef struct
{
    size_t rows;
    size_t cols;
    float *elements;
} Mat;

typedef struct 
{
    size_t cols;
    float *elements;
} Row;


typedef struct 
{
    size_t capacity;
    size_t size;
    uintptr_t *word;
}Region;

Region region_alloc_alloc(size_t capacity_bytes);
void* region_alloc(Region *r,size_t size_bytes);

#define ROW_AT(row,col) (row).elements[col]
Mat row_as_mat(Row row);
#define row_alloc(r,cols) mat_row
// [1,2,3,2,3,2,3,2,1] row = 3 col = 3 stride = 3
// [ [1,2,3], [2,3,2], [3,2,1]]
// [1,2]
// [1 * 3 + 2] 
#define MAT_AT(m,i,j) (m).elements[(i)*(m).cols + (j)]

// mat 
// [[1,2]]
Mat mat_alloc(size_t rows,size_t cols);
void mat_rand(Mat m,float low,float high); 
void mat_randn(Mat m,float mu, float sigma); 
void mat_print(Mat m,const char* name,size_t padding);
#define MAT_PRINT(m) mat_print(m, #m, 0)
void mat_fill(Mat m, float b);
// void mat_nrand(Mat m);
// 矩阵运算
Mat mat_row(Mat m,size_t row);

void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat b);
void mat_copy(Mat dst,Mat src);
void mat_sig(Mat m);
// void mat_sig(Mat m)

// row
#endif  //MAT_H_