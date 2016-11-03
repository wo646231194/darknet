#ifndef IM2COL_H
#define IM2COL_H

typedef enum{
    CHW2HWC,
    HWC2CHW
}RESHPE_TYPE;

void reshape_cpu(float* data_im,
        int channels, int height, int width, 
        float* data_col, RESHPE_TYPE type);

#ifdef GPU

void reshape_ongpu(float *im,
         int channels, int height, int width,
         float *data_col, RESHPE_TYPE type);

#endif
#endif
