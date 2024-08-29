#ifndef  PERCEPTRON_H_
#define PERCEPTRON_H_

#include<stdio.h>
#include<stddef.h>
#include "mat.h"

// Perceptron
// 输入层 [x_1=0,x_2=0]  y^hat = model([x_1,x_1]) 
typedef struct 
{
    // 输入层
    Mat a0;
    // 隐含层
    Mat w1,b1,a1;
    Mat w2,b2,a2;
}Perceptron;

Perceptron perceptron_alloc(void);
float perceptron_forward(Perceptron m);
float perceptron_cost(Perceptron m, Mat ti,Mat to);
float perceptron_train(Perceptron m,Perceptron grad,float rate);
void finite_diff(Perceptron m, Perceptron g, float eps,Mat ti,Mat to);



#endif //PERCEPTRON_H_