#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "roipool_layer.h"
#include "cuda.h"
#include "utils.h"
#include "blas.h"
}

__global__ void forward_roipool_layer_kernel(int n, int in_h, int in_w, int in_c, int out_h, int out_w, float *roi, float *input, float *output, int *indexes, int num)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % out_w;
    id /= out_w;
    int i = id % out_h;
    id /= out_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;
    int s = b/num;

    int pool_index = j + out_w*(i + out_h*(k + in_c*b));

    int p_index = (s + b*num) * 5;
    int roi_start_w = roi[p_index+1] * in_w;
    int roi_start_h = roi[p_index+2] * in_h;
    // int roi_end_w = roi[p_index+3] * in_w;
    int roi_end_h = roi[p_index+4] * in_h;
    int roi_end_w = roi_end_h;

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
	int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    float bin_size_h = 1.0 * roi_height / out_h;//min is 1
    float bin_size_w = 1.0 * roi_width / out_w;//min is 1

    int hstart = floor(i*bin_size_h);
    int wstart = floor(j*bin_size_w);
    int hend = ceil((i+1)*bin_size_h);
    int wend = ceil((j+1)*bin_size_w);

    hstart = min(max(hstart + roi_start_h, 0), in_h);
    wstart = min(max(wstart + roi_start_w, 0), in_w);
    hend = min(max(hend + roi_end_h, 0), in_h);
    wend = min(max(wend + roi_end_w, 0), in_w);

    int is_empty = (hend <= hstart) || (wend <= wstart);

    float maxval = is_empty ? 0 : -INFINITY;
    input = input + s * in_h * in_w * in_c + k * in_h * in_w;
    for(int ih = hstart; ih < hend; ++ih){
        for(int iw = wstart; iw < wend; ++iw){
            int in = ih * in_w + iw;
            if(input[in] > maxval){
                maxval = input[in]; 
            }
        }
    }
    output[pool_index] = maxval;
}

// __global__ void backward_roipool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes)
// {
    
// }

extern "C" void forward_roipool_layer_gpu(roipool_layer l, network_state state)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    size_t n = h*w*c*l.batch;

    layer conv = state.net.layers[l.index];
    if(conv.type != CONVOLUTIONAL) error("ROI index is error\n");
    if(conv.c!= l.c || conv.h!=l.h || conv.w!=l.w) error("ROI width height channel is error\n");

    forward_roipool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.h, l.w, l.c, l.out_h, l.out_w, state.input, conv.output_gpu, l.output_gpu, l.indexes_gpu, l.n);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_roipool_layer_gpu(roipool_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
    // size_t n = layer.h*layer.w*layer.c*layer.batch;

    // backward_roipool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.h, layer.w, layer.c, layer.stride, layer.size, layer.pad, layer.delta_gpu, state.delta, layer.indexes_gpu);
    // check_error(cudaPeekAtLastError());
}

