#include "utility.cuh"
#include "scene.cuh"

#include <iostream>
#include <cublas.h>
#include <iostream>

void testRender(int nx, int ny, float3* fb);
void cuda_update(float3* output);
void init_cuda(int screen_width, int screen_height);
void free_cuda();