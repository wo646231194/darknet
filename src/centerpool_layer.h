#ifndef CENTERPOOL_LAYER_H
#define CENTERPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer centerpool_layer;

centerpool_layer make_centerpool_layer(int batch, int inputs, int h, int w, int c, int out_h, int out_w, int n, int index);
void forward_centerpool_layer(const centerpool_layer l, network_state state, int n, int height, int width, int* x, int* y);
void backward_centerpool_layer(const centerpool_layer l, network_state state);

#ifdef GPU
// void forward_centerpool_layer_gpu(centerpool_layer l, network_state state);
// void backward_centerpool_layer_gpu(centerpool_layer l, network_state state);
#endif

#endif

