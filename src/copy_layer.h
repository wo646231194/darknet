#ifndef COPY_LAYER_H
#define COPY_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer copy_layer;

copy_layer make_copy_layer(int batch, int h, int w, int c, int index);
void forward_copy_layer(const copy_layer l, network_state state);
void backward_copy_layer(const copy_layer l, network_state state);

#ifdef GPU
void forward_copy_layer_gpu(copy_layer l, network_state state);
void backward_copy_layer_gpu(copy_layer l, network_state state);
#endif

#endif

