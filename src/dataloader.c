// dataloader.c
#include <stdio.h>
#include <stdlib.h>
#include "../include/dataloader.h"


// loader data 
// TFRecord / 

// torch Dataset ,getitem(idx) len()
// lua programming
// python 



int csv_loader(const char *filename, float data[][MAX_COLS]) {
    FILE *file;
    int row = 0;

    // 打开 CSV 文件
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("无法打开文件");
        return -1;
    }

    // 逐行读取文件
    while (fscanf(file, "%f,%f", &data[row][0], &data[row][1]) == 2) {
        row++;
        if (row >= MAX_ROWS) {
            printf("超出最大行数限制！\n");
            break;
        }
    }

    // 关闭文件
    fclose(file);

    return row;  // 返回读取的行数
}
