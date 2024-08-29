#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#include "mat.h"


float rand_float(void);
float randn_float(float mu, float sigma);

float cost(float w,float train_data[][2], int rows);
float dcost(float w,float train_data[][2] ,int rows);

void print_train_data(float train_data[][2], int rows);
void mat_print(Mat m,const char* name,size_t padding);