#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "reshape.h"
#include "cuda.h"
}

__global__ void reshape_gpu_kernel(const int n, const float* data_im,
    const int channels, const int height, const int width,
    float* data_col, RESHPE_TYPE type)
{
    int index = threadIdx.x;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    switch (type)
    {
        case CHW2HWC: data_col[c + h*width*channels + w*channels] = data_im[w + c*height*width + h*width]; break;
        case HWC2CHW: data_col[w + c*height*width + h*width] = data_im[c + h*width*channels + w*channels]; break;
        default: break;
    }
}

void reshape_ongpu(float *im,
         int channels, int height, int width,
         float *data_col, RESHPE_TYPE type){
    int num_kernels = channels * height * width;
    reshape_gpu_kernel <<<1, num_kernels>>>(num_kernels, im, channels, height, width, data_col, type);
}
