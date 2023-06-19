#include<stdio.h>
#include<stdlib.h>

int main(int argc, char const *argv[])
{
    int a[2][3] = {
        {2,3,2},
        {2,1,1}
    };
    printf("%zu\n",sizeof(a[0])/sizeof(int));
    printf("%zu\n",sizeof(a)/sizeof(int));
    // for (size_t i = 0; i < count; i++)
    // {
    //     /* code */
    // }
    


    return 0;
}
