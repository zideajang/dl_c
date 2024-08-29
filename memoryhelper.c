#include <stdio.h>
#include <stdlib.h>

struct MemoryInfo {
    size_t size;
    void *address;
    char location[10]; // "stack" or "heap"
};

struct MemoryInfo getMemoryInfo(void *ptr) {
    struct MemoryInfo info = {0};

    // 判断是否为NULL指针
    if (ptr == NULL) {
        return info;
    }

    // 获取内存大小
    info.size = sizeof(*ptr);

    // 获取内存地址
    info.address = ptr;

    // 判断存储位置
    // 这里是一个简化的判断方法，实际情况可能更复杂
    // 可以通过比较地址范围或使用编译器提供的扩展来判断
    if ((char*)ptr >= (char*)&info && (char*)ptr <= (char*)&main) {
        strcpy(info.location, "stack");
    } else {
        strcpy(info.location, "heap");
    }

    return info;
}