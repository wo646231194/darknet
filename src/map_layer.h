#ifndef MAP_LAYER_H
#define MAP_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer map_layer;

map_layer make_map_layer(int batch, int h, int w, int c, int size, int stride, int padding, int class);
void forward_map_layer(const map_layer l, network_state state);
void backward_map_layer(const map_layer l, network_state state);

#ifdef GPU
void forward_map_layer_gpu(map_layer l, network_state state);
void backward_map_layer_gpu(map_layer l, network_state state);
#endif

#endif

