#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

#define PI 3.14159265359
#define kEpsilon 0.01

//column majored matrix
class mat4x4 {
public:
	float e[16];
	__host__ __device__ float& operator() (int i,int j) {
		return e[i + j*4];
	}
};

enum class OBJECT_TYPE {
	SPHERE, TRIANGLE
};
enum class TR_TYPE {
	WARD, REINHARD
};

__host__ __device__ float rad_2_deg(float rad);
__host__ __device__ float deg_2_rad(float degree);
__host__ __device__ float3 normalize(float3 vec3);
__host__ __device__ float4 normalize(float4 vec4);

__host__ __device__ mat4x4 transpose(mat4x4 ma);

__host__ __device__ mat4x4 operator*(mat4x4 ma1, mat4x4 ma2);
__host__ __device__ float3 operator*(mat4x4 ma, float3 vec3);
__host__ __device__ float4 operator*(mat4x4 ma, float4 vec4);

__host__ __device__ float3 operator+(float3 v1, float3 v2);
__host__ __device__ float3 operator-(float3 v1, float3 v2);
__host__ __device__ float3 operator*(float3 v, float f);
__host__ __device__ float3 operator/(float3 v, float f);
__host__ __device__ float3 operator*(float f, float3 v);
__host__ __device__ float3 operator-(float3 f);

__host__ __device__ float2 operator+(float2 v1, float2 v2);
__host__ __device__ float2 operator-(float2 v1, float2 v2);
__host__ __device__ float2 operator*(float2 v1, float f);
__host__ __device__ float2 operator/(float2 v1, float f);

__host__ __device__ mat4x4 mtx_gen_translate(float x, float y, float z);

template<typename T>
__host__ __device__ void swap(T &t1, T &t2);


__host__ __device__ float dot(float3 v1, float3 v2);
__host__ __device__ float3 cross(float3 v1, float3 v2);
__host__ __device__ float length(float3 v);
__host__ __device__ float angle(float3 v1, float3 v2);

__host__ __device__ float3 phongShading(float3 lightPos, float3 normal, float3 fragPos, float3 viewPos, float3 lightColor, float3 ambientColor, float3 diffuseColor, float shininess);

__device__ float3 reflect(float3 input_vec, float3 normal);
__device__ float3 refract(float3 input_vec, float3 normal, float ni, float nt);
__host__ __device__ float getTime(std::chrono::time_point<std::chrono::steady_clock> time_begin);
__host__ __device__ float3 rotateAbout(std::chrono::time_point<std::chrono::steady_clock> time_begin, float3 rotCenter, float3& origin, float A, float f, float shift, float x_mult, float y_mult, float z_mult);

__host__ __device__ float getLuminance(float3 color);
