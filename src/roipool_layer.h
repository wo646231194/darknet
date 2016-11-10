#ifndef ROIPOOL_LAYER_H
#define ROIPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer roipool_layer;

roipool_layer make_roipool_layer(int batch, int inputs, int h, int w, int c, int out_h, int out_w, int n, int index);
void forward_roipool_layer(const roipool_layer l, network_state state, int n, int height, int width, int* x, int* y);
void backward_roipool_layer(const roipool_layer l, network_state state);

#ifdef GPU
void forward_roipool_layer_gpu(roipool_layer l, network_state state);
void backward_roipool_layer_gpu(roipool_layer l, network_state state);
#endif

#endif

