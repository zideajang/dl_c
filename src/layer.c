#include "../include/layer.h"

float sigmoidf(float x)
{
    return 1.0f /(1.0f + expf(-x));
}

float reluf(float x)
{
    return x > 0 ? x : x*NN_RELU_PARAM;
}

float tanhf(float x)
{
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx)/(ex + enx);
}


float actf(float x, Act act){
    switch (act)
    {
    case ACT_SIG:  return sigmoidf(x);
    case ACT_RELU: return reluf(x);
    case ACT_TANH: return tanhf(x);
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}
float dactf(float y, Act act)
{
    switch (act)
    {
    case ACT_SIG:  return y*(1 -y);
    case ACT_RELU: return y >= 0? 1:NN_RELU_PARAM;
    case ACT_TANH: return 1 - y*y;
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}
