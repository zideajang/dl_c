#include<stdbool.h>

#define NN_IMPLEMENTATION

#include<stdio.h>

#include "nn.h"

bool mat_equal(Mat m1,Mat m2){
    if(m1.rows != m2.rows){
        return false;
    }
    if(m1.cols != m2.cols){
        return false;
    }

    for (size_t i = 0; i < m1.rows; i++)
    {
        for (size_t j = 0; j < m1.cols; j++)
        {
            if( MAT_AT(m1,i,j) != MAT_AT(m2,i,j)){
                return false;
            }
        }
        
    }

    return true;

}



// Mat mat_fill(size_t rows, size_t cols,float a){
//     Mat m = mat_alloc(rows,cols);

// }




                    
// array([[2, 1],
//        [1, 1]])


int main(int argc, char const *argv[])
{

Mat m1 = mat_alloc(2,3);
MAT_AT(m1,0,0) = 1; 
MAT_AT(m1,0,1) = 0; 
MAT_AT(m1,0,2) = 0; 
MAT_AT(m1,1,0) = 0; 
MAT_AT(m1,1,1) = 1; 
MAT_AT(m1,1,2) = 0;

Mat m2 = mat_alloc(3,2);
MAT_AT(m2,0,0) = 2; 
MAT_AT(m2,0,1) = 1; 
MAT_AT(m2,1,0) = 1; 
MAT_AT(m2,1,1) = 1; 
MAT_AT(m2,2,0) = 2; 
MAT_AT(m2,2,1) = 1; 

Mat res = mat_alloc(2,2);

mat_dot(res,m1,m2);
mat_print(res);

#ifdef SUM_TEST
    Mat m1 = mat_alloc(2,3);
    mat_fill(m1,1.0f);
    Mat m2 = mat_alloc(2,3);
    mat_fill(m2,2.0f);

    mat_sum(m1,m2);

    
    Mat expected_value = mat_alloc(2,3);
    mat_fill(expected_value,3.0f);


    if(mat_equal(m1,expected_value)){
        printf("PASS");
    }else{
        printf("FAIL");
    }
#endif

    

    return 0;
}
