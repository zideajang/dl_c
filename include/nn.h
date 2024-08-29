#ifndef NN_H_
#define NN_H_

#include<stdlib.h>
#include<assert.h>

#include "mat.h"
typedef struct 
{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
// #define row_print(row, name, padding) mat_print(row_as_mat(row), name, padding)

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn,const char* name);
void nn_rand(NN nn, float low, float high);
#define NN_PRINT(nn) nn_print(nn,#nn)
// size_t arch[] = {2,2,1};

void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps,Mat ti,Mat to);
float nn_learn(NN nn, NN g,float rate);

#endif //NN_H_
