#include<stdio.h>
#include<stdlib.h>

// 支持全连接层、卷积层、激活层
#include "../include/layer.h"
#include "../include/nn.h"


// 定义网络结构，
// 2 表示输入层神经元的数量
// 2 表示隐含层神经元的数量
// 1 表示输出层的神经元数量
size_t arch[] = {2, 2, 1};
// 设置最大 epoch 数量
size_t max_epoch = 100*1000;

int main(void){
    printf("create neural network..\n");
    // 加载数据
    Mat t = mat_alloc(NULL, 4, 3);
    // 定义模型
    // 训练模型
    // 推理
    return 0;
}