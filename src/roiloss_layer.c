#include "roiloss_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

roiloss_layer make_roiloss_layer(int batch, int inputs, int classes, int coords, int num, int h, int w)
{
    roiloss_layer l = {0};
    l.type = ROILOSS;

    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.w = w;
    l.h = h;
    fprintf(stderr, "ROILoss Layer : %d batch\n", batch);
    l.n = num;
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs*l.batch;
    l.truths = 50*(5+l.classes);//if 50 person one image
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    // l.forward = forward_roiloss_layer;
    l.backward = backward_roiloss_layer;
#ifdef GPU
    // l.forward_gpu = forward_roiloss_layer_gpu;
    l.backward_gpu = backward_roiloss_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    srand(0);

    return l;
}

// void forward_roiloss_layer(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y)
// {
    
// }

void backward_roiloss_layer(const roiloss_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

// void get_roiloss_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
// {
    
// }

#ifdef GPU

void forward_roiloss_layer_gpu(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));//-----in_cpu
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    if(!state.train){
        float *mapout = get_maploss_layer_output(state.net);
        int i,j;
        for (i = 0; i < 1; ++i){
            int cell_index = y[i] * l.w + x[i];
            for (j = 0; j < l.n; ++j){
                int p_index = i*l.n + j;
                box out  = float_to_box(mapout + cell_index*l.n*5 + p_index*5 + 1);
                mapout[cell_index*l.n*5 + p_index*5 + 0] = in_cpu[p_index];
                mapout[cell_index*l.n*5 + p_index*5 + 1] = (out.x+out.w)/2;
                mapout[cell_index*l.n*5 + p_index*5 + 2] = (out.y+out.h)/2;
                mapout[cell_index*l.n*5 + p_index*5 + 3] = out.w-out.x;
                mapout[cell_index*l.n*5 + p_index*5 + 4] = out.h-out.y;
            }
        }
        return;
    }

    float neg_loss = 0, pos_loss=0;
    int count = 0, i;
    int *truth = get_maploss_layer_indexes(state.net);
    for(i=0;i<l.batch;i++){
        l.delta[i] = truth[i] - in_cpu[i];
        if(truth[i]==1){
            count++;
            pos_loss -= log(in_cpu[i]);
        } else {
            neg_loss -= log(1 - in_cpu[i]);
        }
    }
    if(count)
    *(l.cost) += pos_loss/count + neg_loss/(l.batch - count);

    if(count) printf("Pos loss: %f, Neg loss: %f, count %d\n", pos_loss/count, neg_loss/(l.batch - count), count);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_roiloss_layer_gpu(roiloss_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif
