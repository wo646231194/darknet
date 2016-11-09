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

maploss_layer make_maploss_layer(int batch, int inputs, int classes, int coords, int size, int num, int h, int w, int step)
{
    maploss_layer l = {0};
    l.type = MAPLOSS;

    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.size = size;
    l.w = w;
    l.h = h;
    l.steps = step;
    fprintf(stderr, "MapLoss Layer : %d scale\n", num);
    assert(num*(classes+coords) == inputs);
    l.n = num;
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs*l.w*l.h;
    l.truths = 50*(5+l.classes);//if 50 person one image
    if(l.size>0){
        l.output = calloc(batch*l.outputs, sizeof(float));
    }else{
        l.output = calloc(batch*l.inputs, sizeof(float));
    }
    l.delta = calloc(batch*l.inputs, sizeof(float));
    l.indexes = calloc(batch*l.n, sizeof(int));

    // l.forward = forward_maploss_layer;
    l.backward = backward_maploss_layer;
#ifdef GPU
    // l.forward_gpu = forward_maploss_layer_gpu;
    l.backward_gpu = backward_maploss_layer_gpu;
    if(l.size>0){
        l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    }else{
        l.output_gpu = cuda_make_array(l.output, batch*l.inputs);
    }
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
    int size = l.inputs * l.batch;
    memset(l.delta, 0, size * sizeof(float));
    memset(l.indexes, 0, l.batch * l.n * sizeof(int));
    ACTIVATION active = get_activation("logistic");
    if(l.size==0){
        float siou = 0.0;
        for (i = 0; i < l.batch; ++i){
            for (j = 0; j < l.n; ++j){
                int in = (i*l.n + j)*(l.coords + l.classes);
                float h_theta_y[1] = { state.input[in] };
                activate_array(h_theta_y, 1, active);

                box out = float_to_box(state.input + in + 1);
                out.x = (out.x + x[i]) / l.w;
                out.y = (out.y + y[i]) / l.h;
                out.h = out.h*out.h;
                out.w = out.h * 0.41 / 64 * 48;

                l.output[in + 0] = h_theta_y[0];
                l.output[in + 1] = out.x-out.w/2;//left
                l.output[in + 2] = out.y-out.h/2;//top
                l.output[in + 3] = out.w+out.w/2;//right
                l.output[in + 4] = out.h+out.h/2;//bottom

                int truth_index = i * 50 * (5 + l.classes),t=0;
                while(state.truth[truth_index]==1){
                    box truth;
                    truth = float_to_box(state.truth + truth_index + 1 + l.classes);
                    
                    truth.x /= width;
                    truth.y /= height;
                    out.x /= width;
                    out.y /= height;
                    float tiou = box_iou(truth, out);
                    if(tiou>0.7){
                        siou += tiou;
                        count++;
                        l.indexes[i*l.n + j] = 1;//----pos----
                    }
                    t++;
                    truth_index += t*(5 + l.classes);
                }
            }
        }
        if(count)
        printf("ROILoss Avg IOU: %f, ", siou / count);
        return;
    }
    //----------------neg---------------
    for (i = 0; i < l.batch; ++i){
        for (j = 0; j < l.n; ++j){
            int in = (i*l.n + j)*(l.coords + l.classes);
            float h_theta_y[1] = { state.input[in] };
            activate_array(h_theta_y, 1, active);
            float delta[1] = { (0.0 - h_theta_y[0]) };
            gradient_array(h_theta_y, 1, active, delta);
            l.delta[in] = delta[0]*l.noobject_scale;
            if(h_theta_y[0]>0.9999999999) neg_loss = 10;
            neg_loss -= log(1 - h_theta_y[0]);
        }
    }
    //----------------pos--------------------
    for (i = 0; i < l.batch; ++i){
        int truth_index = i * 50 * (5 + l.classes);
        int t=0,is_obj=0;
        box truth;
        while(state.truth[truth_index]==1){
            truth = float_to_box(state.truth + truth_index + 1 + l.classes);
            if(floor(truth.x*width) == x[i] && floor(truth.y*height) == y[i]) {
                is_obj=1;
                break;
            }
            t++;
            truth_index += t*(5 + l.classes);
        }
        if (is_obj==0) continue;

        truth.x = truth.x*width - x[i];
        truth.y = truth.y*height - y[i];

        int best_index = -1;
        float best_iou = 0;
        float best_rmse = 20;

        //-------------find best match box----------------
        for(j = 0; j < l.n; ++j){
            int p_index = i*l.inputs + j*(l.coords + l.classes);
            box out = float_to_box(state.input + p_index + 1);

            out.h = out.h*out.h;
            out.w = out.h * 0.41 / 64 * 48;

            truth.x /= width;
            truth.y /= height;
            out.x /= width;
            out.y /= height;

            float tiou  = box_iou(out, truth);
            float rmse = box_rmse(out, truth);
            if(best_iou > 0 || tiou > 0){
                if(tiou > best_iou){
                    best_iou = tiou;
                    best_index = j;
                }
            }else{
                if(rmse < best_rmse){
                    best_rmse = rmse;
                    best_index = j;
                }
            }
            if(l.steps == 1 && tiou > 0.5){
                count++;
                // int p_index = i*l.inputs + j*(l.coords + l.classes);
                //---------------fix box---------------
                // box out = float_to_box(state.input + p_index + 1);

                out.h = out.h*out.h;
                out.w = out.h * 0.41 / 64 * 48;

                float coord[4] = { out.x , out.y , 0, state.input[p_index+4] };
                float tcoord[4] = { truth.x, truth.y, 0, sqrt(truth.h) };
                float dcoord[4] = { 0 };
                float ecoord[4] = { 0 };

                //----------------loc loss--------------------
                l2_cpu(4, coord, tcoord, dcoord, ecoord);
                l.delta[p_index + 1] = dcoord[0]*l.coord_scale;
                l.delta[p_index + 2] = dcoord[1]*l.coord_scale;
                l.delta[p_index + 3] = dcoord[2]*l.coord_scale;
                l.delta[p_index + 4] = dcoord[3]*l.coord_scale;
                loc_loss += sum_array(ecoord, 4);

                //-----------------pos loss--------------------
                float h_theta_y[1] = { state.input[p_index] };
                activate_array(h_theta_y, 1, active);
                float delta[1] = { (1.0 - h_theta_y[0]) };
                gradient_array(h_theta_y, 1, active, delta);
                l.delta[p_index] = delta[0]*l.object_scale;
                pos_loss -= log(h_theta_y[0]);
                
                iou += tiou;
            }
        }
        if(best_iou > 0.5){

            count++;
            int p_index = i*l.inputs + best_index*(l.coords + l.classes);
            //---------------fix box---------------
            box out = float_to_box(state.input + p_index + 1);

            out.h = out.h*out.h;
            out.w = out.h * 0.41 / 64 * 48;

            float coord[4] = { out.x , out.y , 0, state.input[p_index+4] };
            float tcoord[4] = { truth.x, truth.y, 0, sqrt(truth.h) };
            float dcoord[4] = { 0 };
            float ecoord[4] = { 0 };

            //----------------loc loss--------------------
            l2_cpu(4, coord, tcoord, dcoord, ecoord);
            l.delta[p_index + 1] = dcoord[0]*l.coord_scale;
            l.delta[p_index + 2] = dcoord[1]*l.coord_scale;
            l.delta[p_index + 3] = dcoord[2]*l.coord_scale;
            l.delta[p_index + 4] = dcoord[3]*l.coord_scale;
            loc_loss += sum_array(ecoord, 4);

            //-----------------pos loss--------------------
            float h_theta_y[1] = { state.input[p_index] };
            activate_array(h_theta_y, 1, active);
            float delta[1] = { (1.0 - h_theta_y[0]) };
            gradient_array(h_theta_y, 1, active, delta);
            l.delta[p_index] = delta[0]*l.object_scale;
            pos_loss -= log(h_theta_y[0]);

            truth.x /= width;
            truth.y /= height;
            out.x /= width;
            out.y /= height;
            
            iou += box_iou(truth, out);
        }
    }
    *(l.cost) += pos_loss + neg_loss/l.n + loc_loss;

    if(count)
    printf("MapLoss Avg IOU: %f, Pos loss: %f, Neg loss: %f, Loc loss: %f, count: %d\n", iou / count, pos_loss / count, neg_loss * 5 / (l.inputs*l.batch), loc_loss / count, count);
}

void forward_maploss_layer_test(const maploss_layer l, float* in_cpu, int n, int height, int width, int* x, int* y)
{
    int i,j;
    ACTIVATION active = get_activation("logistic");
    for (i = 0; i < l.batch; ++i){
        int cell_index = y[i] * l.w + x[i];
        for (j = 0; j < l.n; ++j){
            int p_index = i * l.inputs + j * (l.coords + l.classes);
            float h_theta_y[1] = { in_cpu[p_index] };
            activate_array(h_theta_y, 1, active);

            box out = float_to_box(in_cpu + p_index + 1);

            out.x = (out.x + x[i]) / l.w;
            out.y = (out.y + y[i]) / l.h;
            out.h = out.h*out.h;
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
    int i;
    int box_num = l.outputs / (l.coords + l.classes);
    for (i = 0; i<box_num; ++i){
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
    cuda_push_array(l.output_gpu, l.output, l.batch*l.inputs);
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
