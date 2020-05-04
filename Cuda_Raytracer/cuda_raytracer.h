#include "utility.cuh"
#include "scene.cuh"

#include <iostream>
#include <cublas.h>
#include <iostream>

void cuda_update(float3* output);
void generate_sphere();
void add_light_power(float amount);
void init_cuda(int screen_width, int screen_height);
void free_cuda();