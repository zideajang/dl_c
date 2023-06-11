#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

typedef struct 
{
    size_t r;
    size_t c;
    float* data;
} Tensor;

int main(int argc, char const *argv[])
{   
    double a = 2;
    double* b = &a;
    printf("%d\n",sizeof(*b));
    return 0;
}


