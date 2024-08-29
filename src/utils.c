
#include "../include/utils.h"

float rand_float(void){
    
    return (float)rand()/(float)RAND_MAX;
}


float randn_float (float mu, float sigma)
{
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

  do
    {
      U1 = -1 + ((float) rand () / RAND_MAX) * 2;
      U2 = -1 + ((float) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (float) X1);
}

// 1/(1 + e^{-x})


void print_train_data(float train_data[][2], int rows) {
    printf("[\n");
    for (int i = 0; i < rows; i++) {
        printf("  [%.2f, %.2f]", train_data[i][0], train_data[i][1]);
        if (i < rows - 1) {
            printf(",");  // 在每行的末尾输出逗号，最后一行除外
        }
        printf("\n");
    }
    printf("]\n");
}

// x1, x2, x3
// w1, w2, w3
// y = x1w1 + x2w2 + w3x3

float cost(float w,float train_data[][2], int rows)
{
    
    float result = 0.0f;
    for (size_t i = 0; i < rows; i++)
    {
        float x = train_data[i][0];
        float y = x*w;
        float d = y - train_data[i][1];
        result += d*d;
    }
    result /= rows;
    return result;
}

float dcost(float w,float train_data[][2] ,int rows)
{
    float result =0.0f;
    size_t n = rows;
    for (size_t i = 0; i < n; i++)
    {
        float x = train_data[i][0];
        float y = train_data[i][1];
        result += 2*(x*w - y)*x;
    }
    result /= n;
    return result;
    
}

void mat_print(Mat m,const char* name,size_t padding){
    printf("%*s%s = [\n",(int)padding,"",name);
    for (size_t i = 0; i < m.rows; i++)
    {   
        printf("%*s   ",(int)padding,"");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f ",MAT_AT(m,i,j));
        }
        printf("\n");
    }

    printf("%*s]\n",(int)padding,"");
    
}