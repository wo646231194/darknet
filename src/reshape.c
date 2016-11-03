#include "reshape.h"
#include <stdio.h>

void reshape_cpu(float* data_im,
     int channels,  int height,  int width,
     float* data_col, RESHPE_TYPE type)
{
    int c,h,w;
    for (c = 0; c < channels; ++c){
        for (h = 0; h < height; ++h){
            for (w = 0; w < width; ++w){
                switch (type)
                {
                    case CHW2HWC: data_col[c + h*width*channels + w*channels] = data_im[w + c*height*width + h*width]; break;
                    case HWC2CHW: data_col[w + c*height*width + h*width] = data_im[c + h*width*channels + w*channels]; break;
                    default: break;
                }
            }
        }
    }
}
