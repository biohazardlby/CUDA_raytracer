#include "utility.cuh"


__host__ __device__ float rad_2_deg(float rad) {
	return rad * 180 / PI;
}

__host__ __device__ float deg_2_rad(float degree) {
	return degree * PI / 180;
}

__host__ __device__ float3 normalize(float3 v)
{
	float length = sqrtf((float)(v.x * v.x + v.y * v.y + v.z * v.z));
	return { v.x / length, v.y / length, v.z / length };
}
__host__ __device__ float4 normalize(float4 v) {
	float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x / length, v.y / length, v.z / length,v.w };
}


__host__ __device__ mat4x4 transpose(mat4x4 ma) {
	mat4x4 output;
	for (int m = 0; m < 4; m++) {
		for (int n = 0; n < 4; n++) {
			output(n, m) = ma(m, n);
		}
	}
	return output;
}

__host__ __device__ mat4x4 operator*(mat4x4 ma1, mat4x4 ma2) {
	mat4x4 output;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float sum = 0;
			for (int k = 0; k < 4; k++) {
				sum += ma1(i, k) * ma2(k, j);
			}
			output(i, j) = sum;
		}
	}
	return output;
}

__host__ __device__ float3 operator*(mat4x4 ma, float3 vec3) {
	float res[4];
	float vec[4] = { vec3.x,vec3.y,vec3.z, 1 };

	for (int i = 0; i < 4; i++) {
		float sum = 0;
		for (int j = 0; j < 4; j++) {
			sum += ma(i, j) * vec[j];
		}
		res[i] = sum;
	}
	return { res[0],res[1],res[2] };
}

__host__ __device__ float4 operator*(mat4x4 ma, float4 vec4)
{
	float res[4];
	float vec[4] = { vec4.x,vec4.y,vec4.z,vec4.w };

	for (int i = 0; i < 4; i++) {
		float sum = 0;
		for (int j = 0; j < 4; j++) {
			sum += ma(i, j) * vec[j];
		}
		res[i] = sum;
	}
	return { res[0],res[1],res[2],res[3] };
}

__host__ __device__ float3 operator+(float3 v1, float3 v2)
{
	return { v1.x + v2.x,v1.y + v2.y,v1.z + v2.z };
}

__host__ __device__ float3 operator-(float3 v1, float3 v2)
{
	return { v1.x - v2.x,v1.y - v2.y,v1.z - v2.z };
}

__host__ __device__ float3 operator*(float3 v, float f)
{
	return { v.x * f, v.y * f,v.z * f };
}

__host__ __device__ float3 operator*(float f, float3 v)
{
	return v * f;
}
__host__ __device__ float3 operator/(float3 v, float f) {
	return { v.x / f, v.y / f, v.z / f };
}

__host__ __device__ float3 operator-(float3 f) {
	return { -f.x,-f.y,-f.z };
}

__host__ __device__ float2 operator+(float2 v1, float2 v2)
{
	return { v1.x + v2.x, v1.y + v2.y };
}

__host__ __device__ float2 operator-(float2 v1, float2 v2)
{
	return { v1.x - v2.x, v1.y - v2.y };
}

__host__ __device__ float2 operator*(float2 v1, float f)
{
	return { v1.x*f, v1.y*f };
}

__host__ __device__ float2 operator/(float2 v1, float f)
{
	return { v1.x/f, v1.y/f };
}

__host__ __device__ mat4x4 mtx_gen_translate(float x, float y, float z)
{
	return {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		x,y,z,1
	};
}

__host__ __device__ float dot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ float3 cross(float3 v1, float3 v2)
{
	return {
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x
	};
}
__host__ __device__ float length(float3 v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__host__ __device__ float angle(float3 v1, float3 v2) {
	float dt = dot(v1, v2);
	float len1 =  length(v1);
	float len2 =  length(v2);
	return acos(dt / len1*len2);
}


__host__ __device__ float3 phongShading(
	float3 lightPos, float3 normal, float3 fragPos, float3 viewPos,
	float3 lightColor, float3 ambientColor, float3 diffuseColor, float shininess) 
{
	float3 lightDir = normalize(lightPos - fragPos);
	float3 viewDir = normalize(viewPos - fragPos);
	float3 halfwayDir = normalize(lightDir + viewDir);
	float spec = powf(fmaxf(dot(normal, halfwayDir), 0.0), shininess);
	float3 specular = lightColor * spec;

	float lambertian = fmaxf(dot(lightDir, normal), 0.0);
	float3 diffuse = lambertian * diffuseColor;

	return (specular + diffuse + ambientColor);
}

__device__ float3 reflect(float3 input_vec, float3 normal) {
	return input_vec - 2 * dot(input_vec, normal) * normal;
}

__device__ float3 refract(float3 I, float3 N, float ni, float nt) {


	float ior = ni / nt;
	float cosi =  dot(I, N);
	float etai = 1, etat = ior;
	float3 n = N;

	if (cosi < 0) {
		cosi = -cosi;
	}
	else {
		float temp = etai;
		etai = etat;
		etat = temp;

		n = -N;
	}

	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);

	return k < 0 ? reflect(I,N) : eta * I + (eta * cosi - sqrtf(k)) * n;



	int sq = 1 - ni * ni * (1 - pow((dot(I, N)), 2)) / (nt * nt);
	if (sq < 0) {
		return reflect(I, N);
	}
	return ni * (I - N * dot(I, N)) / nt + N * sqrtf(sq);

}

__host__ __device__ float getTime(std::chrono::time_point<std::chrono::steady_clock> time_begin) {
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = end - time_begin;
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
	return ms;
}

__host__ __device__ float3 rotateAbout(std::chrono::time_point<std::chrono::steady_clock> time_begin, float3 rotCenter, float3& tar_origin, float A, float f, float shift, float x_mult, float y_mult, float z_mult) {

	float t = getTime(time_begin) / 5000;
	float3 res_origin = { 0,0,0 };
	res_origin.x = rotCenter.x + x_mult * A * sin(2 * PI * f * t + shift);
	res_origin.y = rotCenter.y + y_mult * A * sin(2 * PI * f * t + shift);
	res_origin.z = rotCenter.z + z_mult * A * cos(2 * PI * f * t + shift);
	return res_origin;
}

__host__ __device__ float getLuminance(float3 color)
{
	return 0.27 * color.x + 0.67 * color.y + 0.06 * color.z;
}
