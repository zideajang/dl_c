#include <stdio.h>
#include "../include/dataloader.h"
#include "../include/utils.h"

int simple_example(void){

    // 加载数据
    // 流程
    // y = 2x + b
    // {{0,0},{1,2},{2,4},{3,6},{4,8}}
    // 
    float train_data[MAX_ROWS][MAX_COLS] = {0};
    // 加载数据
    int train_count = csv_loader("data.csv", train_data);

    if (train_count == -1) {
        printf("读取文件失败。\n");
        return 1;
    }
    
    print_train_data(train_data,train_count);

    // 设置超参数
    float eps = 1e-3;
    float rate = 1e-2;
    // 初始化参数
    float w = rand_float() * 10.0f;
    float result;
    
    // 开始训练
    result = cost(w,train_data,train_count);
    printf("result = %f before train\n",result);
    for (size_t i = 0; i < 100; i++)
    {
        // 计算梯度
        float dcost = (cost(w + eps,train_data,train_count) - cost(w,train_data,train_count))/eps;
        // 更新参数
        w -= rate*dcost;
        result = cost(w,train_data,train_count);
        printf("%ld: %f\n",i,result);
    }
    
    result = cost(w,train_data,train_count);
    printf("result = %f after train upate parameter\n",result);
    // lim_{eps \rightarrow 0} = \frac{f(w + eps) - f(eps)}{eps}
    // printf("%f\n",cost(w,train_data,train_count));
    // printf("%f\n",cost(w - eps,train_data,train_count));

    printf("------------- predict -----------\n");
    printf("y = x*%f\n",w);

    // forward 

    return 0;
}