// #include "../include/layer.h"
#include "../include/utils.h"
#include "../include/mat.h"

Region region_alloc_alloc(size_t capacity_bytes)
{
    Region r = {0};
}
void* region_alloc(Region *r,size_t size_bytes)
{

}



Mat mat_alloc(size_t rows,size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.elements = NN_MALLOC(sizeof(*m.elements)*rows*cols);
    NN_ASSERT(m.elements != NULL);
    return m;
}

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

void mat_randn(Mat m,float mu, float sigma)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = randn_float(mu, sigma);
        }
        
    }
    
}

void mat_print(Mat m,const char* name,size_t padding){
    printf("%*s%s = [\n",(int)padding,"",name);
    for (size_t i = 0; i < m.rows; i++)
    {   
        printf("%*s   ",(int)padding,"");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",MAT_AT(m,i,j));
        }
        printf("\n");
    }

    printf("%*s]\n",(int)padding,"");
    
}

Mat mat_row(Mat m,size_t row)
{
    return (Mat){
        .rows=1,
        .cols=m.cols,
        .elements=&MAT_AT(m,row,0),
    };

}

Mat row_as_mat(Row row)
{
    return (Mat) {
        .rows = 1,
        .cols = row.cols,
        .elements = row.elements,
    };
}

void mat_copy(Mat dst, Mat src)
{
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
            MAT_AT(m,i,j) =sigmoidf(MAT_AT(m,i,j));
        }
    }
}



void mat_fill(Mat m, float b)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = b;
        }
    }

}
// gemm
// A(2x3) B(3x3) C(2x3)
void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j)  = 0;
            for (size_t k = 0; k < n; k++)
            {
                // a [[2,3,2]] [[1,2,1]^T]
                MAT_AT(dst,i,j) += MAT_AT(a,i,k)*MAT_AT(b,k,j);
            }
            
        }
        
    }
}

void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows ==  a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) = MAT_AT(dst,i,j) + MAT_AT(a,i,j);
        }
        
    }
    

}