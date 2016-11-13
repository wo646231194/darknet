#include "copy_layer.h"
#include "cuda.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>

copy_layer make_copy_layer(int batch, int h, int w, int c, int index)
{
    copy_layer l = {0};
    l.type = COPY;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_h = h;
    l.out_w = w;
    l.out_c = c;
    l.outputs = h*w*c;
    l.index = index;
    l.forward = forward_copy_layer;
    l.backward = backward_copy_layer;

    #ifdef GPU
    l.forward_gpu = forward_copy_layer_gpu;
    l.backward_gpu = backward_copy_layer_gpu;
    #endif

    fprintf(stderr, "Copy Layer From %d : %d x %d x %d image\n", index, h,w,c);
    return l;
}

void forward_copy_layer(copy_layer l, network_state state)
{
    layer copy = state.net.layers[l.index];
    if(l.h!=copy.out_h || l.w!=copy.out_w || l.c!=copy.out_c) error("CopyLayer h w c is error");
}
void backward_copy_layer(const copy_layer l, network_state state)
{
    layer copy = state.net.layers[l.index];
    axpy_cpu(copy.batch*copy.outputs, 1, copy.delta, 1, state.delta, 1);
}

#ifdef GPU
void forward_copy_layer_gpu(copy_layer l, network_state state)
{
    layer copy = state.net.layers[l.index];
    if(l.h!=copy.out_h || l.w!=copy.out_w || l.c!=copy.out_c) error("CopyLayer h w c is error\n");
}
void backward_copy_layer_gpu(copy_layer l, network_state state)
{
    layer copy = state.net.layers[l.index];
    axpy_ongpu(copy.batch*copy.outputs, 1, copy.delta_gpu, 1, state.delta, 1);
}
#endif