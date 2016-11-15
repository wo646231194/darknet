#include "roiloss_layer.h"
#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>

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
    fprintf(stderr, "ROILoss Layer : %d x %d batch\n", inputs, batch);
    l.n = num;
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = 50*(5+l.classes);//if 50 person one image
    int output_size = l.outputs * l.batch;
    l.output = calloc(output_size, sizeof(float));
    l.delta = calloc(output_size, sizeof(float));

    // l.forward = forward_roiloss_layer;
    l.backward = backward_roiloss_layer;
#ifdef GPU
    // l.forward_gpu = forward_roiloss_layer_gpu;
    l.backward_gpu = backward_roiloss_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, output_size);
    l.delta_gpu = cuda_make_array(l.delta, output_size);
#endif

    srand(0);

    return l;
}

void forward_roiloss_layer(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    return;
}

void backward_roiloss_layer(const roiloss_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void print_value(const roiloss_layer l, FILE **fps)
{
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    int r;
    for(r=0;r<l.batch*l.outputs;r++){
        fprintf(fps[0],"%.2f ",l.output[r]);
    }
    fprintf(fps[0],"\r\n");
}

#ifdef GPU

void forward_roiloss_layer_gpu(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    cuda_pull_array(state.input, l.output, l.batch*l.outputs);
    if(!state.train){
        float *mapout = get_maploss_layer_output(state.net);
        int i,j,ix,iy;
        for (i = 0; i < l.batch; ++i){
            for(iy=0; iy<l.h; iy++){
                for(ix=0; ix<l.w; ix++){
                    int p_index = i*l.h*l.w + iy*l.w + ix;
                    for (j = 0; j < l.n; ++j){
                        mapout[p_index*l.n*5 + j*5] *= l.output[p_index];
                    }
                    // printf("%.2f ",l.output[p_index]);
                }
                // printf("\n");
            }
            // for (j = 0; j < l.n; ++j){
            //     int p_index = i*l.n + j;
            //     box out  = float_to_box(mapout + cell_index*l.n*5 + p_index*5 + 1);
            //     mapout[cell_index*l.n*5 + p_index*5 + 0] += in_cpu[p_index];
            // }
        }
        cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
        return;
    }

    float neg_loss = 0, pos_loss=0;
    int count = 0, i;
    int *truth = get_map_layer_indexes(state.net);
    *(l.cost) = 0;
    memset(l.delta, 0, l.outputs * sizeof(float));
    for(i=0;i<l.outputs*l.batch;i++){
        l.delta[i] = truth[i] - l.output[i];
        if(truth[i]==1){
            count++;
            pos_loss -= log(l.output[i]);
        } else {
            neg_loss -= log(1 - l.output[i]);
        }
    }
    int neg_num = (l.outputs*l.batch - count);
    if(count)
    *(l.cost) += pos_loss/count + neg_loss/neg_num;

    if(count) printf("  Pos loss: %f, Neg loss: %f, count %d\n", pos_loss/count, neg_loss/neg_num, count);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_roiloss_layer_gpu(roiloss_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.outputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif
