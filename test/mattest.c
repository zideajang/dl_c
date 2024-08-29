#include<stdio.h>
#include<time.h>

#include "../include/mat.h"
#include "../include/utils.h"
#include "../include/perceptron.h"


// #define TEST_SUM 1
// #define TEST_DOT 1
#define TEST_PERCETRON
int main(int argc, char const *argv[])
{
    printf("test mat\n");

    #ifdef TEST_PERCETRON

    printf("test perceptron...\n");
    srand(time(0));
    // preparation data
    float train_data[] = {
        0,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
    };

    size_t stride = 3;
    size_t n = sizeof(train_data)/sizeof(train_data[0])/stride;

    // idx2word css
    // idx2 {0,'a',1,'b'}
    // pytorch Matrix test
    // pytorch anet 
    // github zideajang anet python 实现 pytorh
    // data input (x1 = 0,x2 = 2)
    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .data = train_data
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .data = train_data + 2
    };

    // mat_print(ti,"ti",0);
    // mat_print(to,"to",0);

    // 初始化网络参数
    Perceptron m = perceptron_alloc();
    
    mat_rand(m.w1,0,1);
    mat_rand(m.b1,0,1);

    mat_rand(m.w2,0,1);
    mat_rand(m.b2,0,1);
    
    // 初始化梯度
    Perceptron g = perceptron_alloc();

    float eps = 1e-1;
    float rate = 1e-1;

    printf("cost =%f\n",perceptron_cost(m,ti,to));

    float* cost_arr = (float*)malloc(20*sizeof(float));
    int j = 0;
    for (size_t i = 0; i < 20*1000; i++)
    {
        finite_diff(m,g,eps,ti,to);
        perceptron_train(m,g,rate);
        if(i%1000== 0){
            // printf("%zu: cost = %f\n",i,perceptron_cost(m,ti,to));
            *(cost_arr + j) = perceptron_cost(m,ti,to);
            j++;

        }
        /* code */
    }
    FILE *fp;
    fp = fopen("cost.csv", "w");
    if (fp == NULL) {
        printf("open fail file\n");
        return 1;
    }
    float cost_value;
    for (size_t i = 0; i < 20; i++)
    {   
        cost_value = *(cost_arr + i);
        printf("%zu: cost = %f\n",i,cost_value);
        // scanf("%f", cost_value);
        fprintf(fp, "%.6f\n", cost_value);
    }
    fclose(fp);


    Mat x = mat_row(ti,0);
    Mat ground = mat_row(to,0);

    mat_copy(m.a0,x);
    float y = perceptron_forward(m);
    printf("y = %f(predict)\n ", y);
    // create percepton model

    // train

    // predict


    #endif //TEST_PERCETRON

    #ifdef GENERATE_MAT
        srand(time(0));
        printf("Test mat_rand\n");
        // 初始化的均值为 0 方差为 1 的正太分布
        Mat m = mat_alloc(3,3);
        mat_randn(m,2.0f,3.0f);    
        // create rand mat
        mat_print(m,"test",0);
        printf("Test mat_dot\n");
    #endif
    
    #ifdef TEST_SUM
    printf("test mat_sum method\n");
    Mat m1 = mat_alloc(3,3);
    Mat m2 = mat_alloc(3,3);
    // Mat m3 = mat_alloc(3,3);
    mat_fill(m1,1.0f);
    mat_fill(m2,2.0f);

    mat_sum(m2,m1);
    mat_print(m2,"m2",0);
    #endif

    #ifdef  TEST_DOT
    printf("test mat dot");
    Mat m1 = mat_alloc(3,2);
    mat_randn(m1,2.0f,3.0f);
    mat_print(m1,"m1",0);
    Mat m2 = mat_alloc(2,3);
    mat_fill(m2,1.0f);
    Mat m3 = mat_alloc(3,3);
    mat_dot(m3,m1,m2);
    mat_print(m3,"m3",0);
    
    #endif //TEST_DOT

    return 0;
}
