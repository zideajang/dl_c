#include<stdio.h>
#include<time.h>

#include "../include/mat.h"
#include "../include/utils.h"
// #include "../include/perceptron.h"
#include "../include/nn.h"


#define TRAIN train
float x[] = {
    0,0,
    0,1,
    1,0,
    1,1,
};

float y[] = {0,1,1,0};

const size_t n = sizeof(x)/sizeof(x[0])/2;


Mat ti ={
    .rows = 4,
    .cols = 2,
    .elements = x,
};

Mat to ={
    .rows = 4,
    .cols = 1,
    .elements = y,
};


int main(int argc, char const *argv[])
{
    srand(time(0));
    // size_t arch[] = {2,10,10,1};
    printf("n = %d\n",n);
    size_t arch[] = {2,2,1};

    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g = nn_alloc(arch,ARRAY_LEN(arch));
    nn_rand(nn,0.0f,1.0f);

    float eps = 1e-1;
    float rate = 1e-1;



    #ifdef TRAIN
        printf("cost = %f\n",nn_cost(nn,ti,to));

        for (size_t i = 0; i < 10*1000; i++)
        {
            nn_finite_diff(nn,g,eps,ti,to);
            nn_learn(nn,g,rate);
            printf("cost = %f\n",nn_cost(nn,ti,to));
        }

        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                MAT_AT(NN_INPUT(nn), 0, 0) = i;
                MAT_AT(NN_INPUT(nn), 0, 1) = j;
                nn_forward(nn);
                printf("%zu ^ %zu = %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0));
            }
        }
        
        // MAT_PRINT(mat_row(ti,1));
        // mat_copy(NN_INPUT(nn),mat_row(ti,1));
        // nn_forward(nn);
        // MAT_PRINT(NN_OUTPUT(nn));
        // NN_PRINT(nn);
    #endif
    return 0;
}

