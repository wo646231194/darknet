#include "roipool_layer.h"
#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include "network.h"
#include <stdio.h>

roipool_layer make_roipool_layer(int batch, int inputs, int h, int w, int c, int out_h, int out_w, int n, int index)
{
    fprintf(stderr, "ROIpool Layer: %d x %d x %d image -> %d x %d x %d x %d image\n", h,w,c,n,out_h,out_w,c);
    roipool_layer l = {0};
    l.type = ROIPOOL;
    l.batch = batch * n;//per image n roi, parallel to batch
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.out_w = out_w;
    l.out_h = out_h;
    l.out_c = c;
    l.index = index;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = inputs;
    int output_size = l.out_h * l.out_w * l.out_c * l.batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    // l.forward = forward_roipool_layer;
    l.backward = backward_roipool_layer;
    #ifdef GPU
    l.forward_gpu = forward_roipool_layer_gpu;
    l.backward_gpu = backward_roipool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void forward_roipool_layer(const roipool_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    int b,i,j,k,s;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    layer conv = state.net.layers[l.index];
    if(conv.type != CONVOLUTIONAL) error("ROI index is error\n");
    if(conv.out_c!= l.c || conv.out_h!=l.h || conv.out_w!=l.w) error("ROI width height channel is error\n");
    
    float *roi = state.input;
    float *incpu = get_network_output_layer_gpu(state.net, l.index);

    int batch = l.batch/l.n;
    for(b = 0; b < batch; ++b){
        int cell_index = y[b] * width + x[b];
        int im_index = b/l.n;//image index
        for(s = 0; s < l.n; ++s){
            int p_index = (s + b*l.n) * 5 + cell_index * l.n * 5;
            int roi_start_w = roi[p_index+1] * l.w;//left
            int roi_start_h = roi[p_index+2] * l.h;//top
            int roi_end_w = roi[p_index+3] * l.w;//right
            int roi_end_h = roi[p_index+4] * l.h;//bottom
            if(roi[p_index+1]==0 && roi[p_index+2]==0 && roi[p_index+3]==0 && roi[p_index+4]==0) error("roi is zero");

            int roi_height = constrain_int(roi_end_h - roi_start_h + 1, 1, l.h);
			int roi_width = constrain_int(roi_end_w - roi_start_w + 1, 1, l.w);

            float bin_size_h = 1.0 * roi_height / l.out_h;
            float bin_size_w = 1.0 * roi_width / l.out_w;

            for(k = 0; k < c; ++k){
                for(i = 0; i < h; ++i){
                    for(j = 0; j < w; ++j){
                        int hstart = floor(i*bin_size_h);
                        int wstart = floor(j*bin_size_w);
                        int hend = ceil((i+1)*bin_size_h);
                        int wend = ceil((j+1)*bin_size_w);

                        hstart = constrain_int(hstart + roi_start_h,0,l.h);
                        wstart = constrain_int(hstart + roi_start_w,0,l.w);
                        hend = constrain_int(hstart + roi_end_h,0,l.h);
                        wend = constrain_int(hstart + roi_end_w,0,l.w);

                        int is_empty = (hend <= hstart) || (wend <= wstart);
                        int pool_index = (b*l.n + s) * l.outputs + k*h*w + i*w + j;
                        if (is_empty){
                            l.output[pool_index] = 0;
                        }

                        float maxval = is_empty ? 0 : -INFINITY;
                        for(int ih = hstart; ih < hend; ++ih){
                            for(int iw = wstart; iw < wend; ++iw){
                                int in = im_index * l.inputs + k * l.h * l.w + ih * l.w + iw;
                                if(incpu[in] > maxval){
                                    maxval = incpu[in];
                                }
                            }
                        }
                        l.output[pool_index] = maxval;
                    }
                }
            }
        }
    }
    cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
}

void backward_roipool_layer(const roipool_layer l, network_state state)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        state.delta[index] += l.delta[i];
    }
}
