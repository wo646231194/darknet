#include "centerpool_layer.h"
#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include "network.h"
#include <stdio.h>

centerpool_layer make_centerpool_layer(int batch, int inputs, int h, int w, int c, int out_h, int out_w, int n, int index)
{
    fprintf(stderr, "CenterPool Layer: %d x %d x %d image -> %d x %d x %d image\n", h,w,c,out_h,out_w,c);
    centerpool_layer l = {0};
    l.type = CENTERPOOL;
    l.batch = batch;
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
    // l.forward = forward_centerpool_layer;
    l.backward = backward_centerpool_layer;
    #ifdef GPU
    // l.forward_gpu = forward_centerpool_layer_gpu;
    // l.backward_gpu = backward_centerpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void forward_centerpool_layer(const centerpool_layer l, network_state state, int n, int height, int width, int* x, int* y)
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
    
    for(b = 0; b < l.batch; ++b){
        int cell_index = y[b] * width + x[b];
        int wstart = x[b] - (w-1)/2;
        int wend = x[b] + (w-1)/2 + 1;
        int hstart = y[b] - (w-1)/2;
        int hend = y[b] + (w-1)/2 + 1;
        if((wend - wstart)!=w) error("width is error\n");
        if((hend - hstart)!=h) error("height is error\n");

        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int ht = hstart + i;
                    int wt = hend + j;
                    int is_empty = (ht < 0) || (wt < 0) || (ht >= l.h) || (wt >= l.w);
                    int pool_index = b * l.outputs + k*h*w + i*w + j;
                    int conv_index = b * l.inputs + k*l.h*l.w + ht*l.w + wt;
                    if (is_empty){
                        l.output[pool_index] = 0;
                    }else{
                        l.output[pool_index] = incpu[conv_index];
                    }
                }
            }
        }
    }
    cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
}

void backward_centerpool_layer(const centerpool_layer l, network_state state)
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
