#include "maploss_layer.h"
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

maploss_layer make_maploss_layer(int batch, int inputs, int classes, int coords, int size, float step, int h)
{
    maploss_layer l = {0};
    l.type = MAPLOSS;

    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.size = size;
    l.dot = step;
    l.w = h;
    l.h = h;
    int n = 0;
    float base = 1.0 * size / h;
    while (base < 1.0){
        n++;
        base *= step;
    }
    fprintf(stderr, "MapLoss Layer : %d scale\n", n);
    assert(n*(classes+coords) == inputs);
    l.n = n;
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs*l.w*l.h;
    l.truths = 50*(5+l.classes);//if 50 person one image
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.inputs, sizeof(float));

    l.forward = forward_maploss_layer;
    l.backward = backward_maploss_layer;
#ifdef GPU
    l.forward_gpu = forward_maploss_layer_gpu;
    l.backward_gpu = backward_maploss_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.inputs);
#endif

    srand(0);

    return l;
}

void forward_maploss_layer(const maploss_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    int i,j;
    float neg_loss = 0;
    float loc_loss = 0;
    float pos_loss = 0;
    float iou = 0;
    
    int count = 0;
    *(l.cost) = 0;
    int size = l.inputs * l.batch;
    memset(l.delta, 0, size * sizeof(float));
    //----------------neg---------------
    ACTIVATION active = get_activation("logistic");
    for (i = 0; i < l.batch; ++i){
        for (j = 0; j < l.n; ++j){
            int in = (i*l.n + j)*(l.coords + l.classes);
            float h_theta_y[1] = { state.input[in] };
            activate_array(h_theta_y, 1, active);
            float delta[1] = { (0.0 - h_theta_y[0]) };
            gradient_array(h_theta_y, 1, active, delta);
            l.delta[in] = delta[0]*l.noobject_scale;
            neg_loss -= log(1 - h_theta_y[0]);
        }
    }
    //----------------pos--------------------
    for (i = 0; i < l.batch; ++i){
        int bt = i * 50 * (5 + l.classes) + n*(5 + l.classes);
        box truth = float_to_box(state.truth + bt + 1 + l.classes);
        if (truth.h <= 0) continue;
        count++;

        //-------------find active box----------------
        float base = 1.0 * l.size / l.h;
        float ms = 1; int m = 0;
        for (int s = 0; s < l.n; ++s){
            float sub = fabs(base - truth.h);
            if (sub < ms){
                ms = sub;
                m = s;
            }
            base *= l.dot;
        }
        float ax = 1.0 * l.size / l.h * pow(l.dot, m);
        
        //---------------fix box---------------
        box b_a;
        b_a.x = truth.x * l.w - x[i] - 0.5;
        b_a.y = truth.y * l.h - y[i] - 0.5;
        b_a.h = log(truth.h / ax) * 10;

        int p_index = m * (l.coords + l.classes);
        box out = float_to_box(state.input + p_index + 1);
        float coord[4] = { out.x , out.y , 0, out.h };
        float tcoord[4] = { b_a.x, b_a.y, 0, b_a.h };
        float dcoord[4] = { 0 };
        float ecoord[4] = { 0 };

        //----------------loc loss--------------------
        smooth_l1_cpu(4, coord, tcoord, dcoord, ecoord);
        l.delta[p_index + 1] = dcoord[0]*l.coord_scale;
        l.delta[p_index + 2] = dcoord[1]*l.coord_scale;
        l.delta[p_index + 3] = dcoord[2]*l.coord_scale;
        l.delta[p_index + 4] = dcoord[3]*l.coord_scale;
        loc_loss = sum_array(ecoord, 4);

        //-----------------pos loss--------------------
        float h_theta_y[1] = { state.input[p_index] };
        activate_array(h_theta_y, 1, active);
        float delta[1] = { (1.0 - h_theta_y[0]) };
        gradient_array(h_theta_y, 1, active, delta);
        l.delta[p_index] = delta[0]*l.object_scale;
        pos_loss -= log(h_theta_y[0]);
        
        out.x = (out.x + x[i] + 0.5) / l.w;
        out.y = (out.y + y[i] + 0.5) / l.h;
        out.h = ax * exp(out.h / 10);
        out.w = out.h * 0.41 / 64 * 48;
        iou += box_iou(truth, out);
    }
    *(l.cost) = pos_loss/count + neg_loss*5/(l.inputs*l.batch) + loc_loss/count;

    printf("MapLoss Avg IOU: %f, Pos loss: %f, Neg loss: %f, Loc loss: %f, count: %d\n", iou / count, pos_loss / count, neg_loss * 5 / (l.inputs*l.batch), loc_loss / count, count);
}

void forward_maploss_layer_test(const maploss_layer l, float* in_cpu, int n, int height, int width, int* x, int* y)
{
    ACTIVATION active = get_activation("logistic");
    for (int i = 0; i < l.batch; ++i){
        int cell_index = y[i] * l.w + x[i];
        for (int j = 0; j < l.n; ++j){
            int p_index = i * l.inputs + j * (l.coords + l.classes);
            float h_theta_y[1] = { state.input[p_index] };
            activate_array(h_theta_y, 1, active);

            float ax = 1.0 * l.size / l.h * pow(l.dot, j);

            box out = float_to_box(state.input + p_index + 1);

            out.x = (out.x + x[i] + 0.5) / l.w;
            out.y = (out.y + y[i] + 0.5) / l.h;
            out.h = ax * exp(out.h / 10);
            out.w = out.h * 0.41 / 64 * 48;

            l.output[cell_index*l.inputs + p_index + 0] = h_theta_y[0];
            l.output[cell_index*l.inputs + p_index + 1] = out.x;
            l.output[cell_index*l.inputs + p_index + 2] = out.y;
            l.output[cell_index*l.inputs + p_index + 3] = out.w;
            l.output[cell_index*l.inputs + p_index + 4] = out.h;
        }
    }
}

void backward_maploss_layer(const maploss_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_maploss_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int box_num = l.outputs / (l.coords + l.classes);
    for (int i = 0; i<box_num; ++i){
        int p_index = i*(l.coords + l.classes);
        probs[i][0] = (l.output[p_index] > thresh) ? l.output[p_index] : 0;
        boxes[i].x = l.output[p_index + 1];
        boxes[i].y = l.output[p_index + 2];
        boxes[i].w = l.output[p_index + 3];
        boxes[i].h = l.output[p_index + 4];
    }
}

#ifdef GPU

void forward_maploss_layer_gpu(const maploss_layer l, network_state state, int n, int height, int width, int* x, int* y)
{
    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));//-----in_cpu
    cuda_pull_array(state.input, in_cpu, l.batch*l.inputs);
    if(!state.train){
        forward_maploss_layer_test(l, in_cpu, n, height, width, x, y);
        free(in_cpu);in_cpu=NULL;//--------------------------------in_cpu end
        return;
    }

    float *truth_cpu = 0;
    if (state.truth){
        int num_truth = l.batch * 50 * (5 + l.classes);
        truth_cpu = calloc(num_truth, sizeof(float));//--------------------truth_cpu
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_maploss_layer(l, cpu_state, n, height, width, x, y);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);//---------------------------truth_cpu end
}

void backward_maploss_layer_gpu(maploss_layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
    //copy_ongpu(l.batch*l.inputs, l.delta_gpu, 1, state.delta, 1);
}
#endif
