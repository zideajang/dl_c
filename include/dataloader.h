// dataloader.h
#ifndef DATALOADER_H_
#define DATALOADER_H_

#define MAX_ROWS 100  
#define MAX_COLS 2    


typedef struct TrainData
{
    int r;
    int c;
    float* data;
}TrainData;

int csv_loader(const char *filename, float data[][MAX_COLS]);

#endif // DATALOADER_H_