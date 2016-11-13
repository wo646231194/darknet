#ifndef ROILOSS_LAYER_H
#define ROILOSS_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer roiloss_layer;

roiloss_layer make_roiloss_layer(int batch, int inputs, int classes, int coords, int num, int h, int w);
void forward_roiloss_layer(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y);
void backward_roiloss_layer(const roiloss_layer l, network_state state);

#ifdef GPU
void forward_roiloss_layer_gpu(const roiloss_layer l, network_state state, int n, int height, int width, int* x, int* y);
void backward_roiloss_layer_gpu(roiloss_layer l, network_state state);
#endif

#endif
