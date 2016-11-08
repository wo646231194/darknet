#ifndef MAPLOSS_LAYER_H
#define MAPLOSS_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer maploss_layer;

maploss_layer make_maploss_layer(int batch, int inputs, int classes, int coords, int size, int num, int h, int w);
void forward_maploss_layer(const maploss_layer l, network_state state, int n, int height, int width, int* x, int* y);
void forward_maploss_layer_test(const maploss_layer l, float* in_cpu, int n, int height, int width, int* x, int* y);
void backward_maploss_layer(const maploss_layer l, network_state state);
void get_maploss_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);

#ifdef GPU
void forward_maploss_layer_gpu(const maploss_layer l, network_state state, int n, int height, int width, int* x, int* y);
void backward_maploss_layer_gpu(maploss_layer l, network_state state);
#endif

#endif
