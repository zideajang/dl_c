#ifndef LAYER_H_
#define LAYER_H_
#include<stdio.h>

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif //NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif //NN_RELU_PARAM

typedef enum{
    ACT_SIG, //sigmoid
    ACT_RELU,
    ACT_TANH,  
} Act;

float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);

// 激活函数前向传播
float actf(float x, Act act);
float dactf(float x, Act act);

#endif//LAYER_H_

