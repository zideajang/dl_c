#include<stdint.h>

#define N 1024
// 加速矩阵相乘在 cpu 侧
float A[N][N];
float B[N][N];
float C[N][N];

uint64_t nanos(){
    // struct timespec start;

    
    
}

int main(int argc, char const *argv[])
{

    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            float acc = 0;
            for (int k = 0; k < N; k++)
            {
                acc += A[y][k] * B[k][x];
            }

            C[y][x] = acc;
            
        }
        
    }
    
    return 0;
}
