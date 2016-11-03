#include "map_layer.h"
#include "cuda.h"
#include <stdio.h>

map_layer make_map_layer(int batch, int h, int w, int c, int size, int stride, int padding, int classes)
{
    fprintf(stderr, "Map Layer: %d x %d x %d image -> %d x %d x %d image\n", h,w,c,size,size,c);
    map_layer l = {0};
    l.classes = classes;
    l.type = MAP;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = size;
    l.out_h = size;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    int input_size = l.h * l.w * l.c * batch;
    l.indexes = calloc(output_size, sizeof(int));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(input_size, sizeof(float));
    l.forward = forward_map_layer;
    l.backward = backward_map_layer;
    #ifdef GPU
    l.forward_gpu = forward_map_layer_gpu;
    l.backward_gpu = backward_map_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, input_size);
    #endif
    return l;
}

void forward_map_layer(const map_layer l, network_state state)
{
    
}

void backward_map_layer(const map_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

#ifdef GPU

void forward_map_layer_gpu(map_layer l, network_state state)
{
    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    fill_ongpu(l.inputs*l.batch, 0, l.delta_gpu, 1);
    memset(l.delta, 0, l.inputs*l.batch*sizeof(float));
    
    //---------------pull truth-----------------
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*50*(5+l.classes);
        truth_cpu = calloc(num_truth, sizeof(float));//------------------------------truth_cpu
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    float *before = calloc(l.batch*l.c*l.h*l.w, sizeof(float));//-------------before
    cuda_pull_array(state.input, before, l.batch*l.c*l.h*l.w);
    float *after = calloc(l.batch*l.c*l.h*l.w, sizeof(float));//------------------------after
    //--------------CHW to HWC--------------------
    for(int i = 0; i < l.batch; ++i){
        reshape_cpu(before + i*l.c*l.h*l.w, l.c, l.h, l.w, after + i*l.c*l.h*l.w, CHW2HWC);
        //reshape_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, state.workspace + i*l.c*l.h*l.w, CHW2HWC);
    }
    free(before); before = NULL;//--------------------------------------------before end
    //--------------max truth box in batch-------------
    int m=0;
    for(int i = 0; i < l.batch; ++i){
        int bt = i*50*(5+l.classes);
        int num=0;
        while (truth_cpu[bt]==1.0){
            num++;
            bt += (5+l.classes);
        }
        m = m>num?m:num;
    }
    //---------------forward max truth-------------------
    for(int n=0; n < m; ++n){
        int *offset = 0;//save where to map
        offset = calloc(l.batch, sizeof(int));//-----------------------offset
        int *x = calloc(l.batch, sizeof(int));//-----------------------x
        int *y = calloc(l.batch, sizeof(int));//-----------------------y
        for(int i = 0; i < l.batch; ++i){
            int bt = i*50*(5+l.classes) + n*(5+l.classes);
            box b = float_to_box(truth_cpu + bt + 1 + l.classes);
            int col = b.x * l.w;
            int row = b.y * l.h;
            if(b.h == 0){
                col = rand()%l.w;
                row = rand()%l.h;
            }
            int index = i*l.c*l.h*l.w + row*l.c*l.h + col*l.c;
            offset[i] = index; x[i] = col; y[i] = row;

            for (int j = 0; j < l.outputs; ++j){
                l.output[j + i*l.outputs] = after[index + j];
            }
            //copy_ongpu(l.outputs, state.input + index, 1, l.output_gpu + i * l.outputs, 1);
        }
        cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
        int now = state.index;
        forward_network_map_gpu(state.net, state, now+1, n, l.w, l.h, x, y);
        backward_network_map_gpu(state.net, state, now+1, n, l.w, l.h);
        float *delta = calloc(l.batch*l.outputs, sizeof(float));//---------------------delta
        cuda_pull_array(l.delta_gpu, delta, l.batch*l.outputs);
        for(int i = 0; i < l.batch; ++i){
            int index = offset[i];
            for (int j = 0; j < l.outputs; ++j){
                l.delta[index + j] += delta[j + i*l.outputs];
            }
        }
        free(offset); offset = NULL;//---------------------------offset end
        free(x); x = NULL;//-------------------------------------x end
        free(y); y = NULL;//-------------------------------------y end
        free(delta); delta = NULL;//-----------------------------------------------------delta end
    }
    free(after); after = NULL;//-------------------------------------------after end
    free(truth_cpu); truth_cpu = NULL;//-----------------------------------------------------truth_cpu end
    //--------------HWC to CHW--------------------
    float *delta = calloc(l.batch*l.inputs, sizeof(float));//-----------------delta
    for(int i = 0; i < l.batch; ++i){
        reshape_cpu(l.delta + i*l.c*l.h*l.w, l.c, l.h, l.w, delta + i*l.c*l.h*l.w, HWC2CHW);
    }
    cuda_push_array(l.delta_gpu, delta, l.batch*l.inputs);
    free(delta); delta = NULL;//----------------------------------------------delta end
}

void backward_map_layer_gpu(map_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}

#endif
