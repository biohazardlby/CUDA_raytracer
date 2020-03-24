#include "cuda_raytracer.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__ float3* cuda_fb_ptr;
__device__ Scene* cuda_scene_ptr;
__device__ size_t cuda_fb_size;
__managed__ float scene_screen_ratio;
__managed__ int nx, ny;
__device__ float3 BG_COLOR = { 0.8,0.8,0.8 };


__device__ std::chrono::time_point<std::chrono::steady_clock> time_begin;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


__global__ void test(float3* fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index] = { float(i) / max_x,float(j) / max_y,.2 };
}
void testRender(int nx, int ny, float3* output) {

	int tx = 8;
	int ty = 8;

	int num_pixels = nx * ny;
	cuda_fb_size = 3 * num_pixels * sizeof(float3);

	// allocate FB
	float3* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, cuda_fb_size));

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	test << <blocks, threads >> > (fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// Output FB 
	cudaMemcpy(output, fb, cuda_fb_size, cudaMemcpyDeviceToHost);

	checkCudaErrors(cudaFree(fb));
}

template<typename T>
__device__ int cuda_raytrace_object(T* obj_array, int obj_arr_size, int skip_idx, Ray ray, float3& frag_point, float3& frag_normal, float& minDist) {
	int hit = -1;
	for (int obj_idx = 0; obj_idx < obj_arr_size; obj_idx++) {
		if (obj_idx == skip_idx) continue;
		T cur_obj = obj_array[obj_idx];
		float3 hit_point = { 0,0,0 };
		float3 hit_normal = { 0,0,0 };
		if (cur_obj.rayTrace(ray, hit_point, hit_normal)) {
			float hit_distance = length(ray.origin - frag_point);
			if (hit_distance <= minDist) {
				hit = obj_idx;
				frag_point = hit_point;
				frag_normal = hit_normal;
				minDist = hit_distance;
			}
		}
	}
	return hit;
}
template<typename T, typename R>
__device__ float3 cuda_calculate_color(
	Light* light_arr, int light_arr_size, T* blocker_arr, int blocker_arr_size, int skip_idx, float3 frag_point, float3 frag_normal, float3 view_pos, R rdr_object) 
{
	float3 finalColor = { 0,0,0 };
	for (int light_idx = 0; light_idx < light_arr_size; light_idx++) {
		Light cur_light = light_arr[light_idx];
		float3 blockRayDir = cur_light.position - frag_point;
		blockRayDir = normalize(blockRayDir);
		Ray blockRay = Ray(frag_point, blockRayDir);

		float3 blocked_frag_pos = { 0,0,0 };
		float3 blocked_frag_normal = { 0,0,0 };
		float blocked_min_dist = FLT_MAX;

		if (cuda_raytrace_object<T>(blocker_arr, blocker_arr_size, skip_idx, blockRay, blocked_frag_pos, blocked_frag_normal, blocked_min_dist) != -1)
		{
			finalColor = finalColor + cur_light.ambientColor;
		}
		else {
			float3 rdr_object_diffuse = rdr_object.get_color(frag_point);
			float rdr_object_shininess = rdr_object.shininess;
			finalColor = finalColor + phongShading(
				cur_light.position, frag_normal, frag_point,
				view_pos, cur_light.color, cur_light.ambientColor, rdr_object_diffuse, rdr_object_shininess);
		}
	}
	return finalColor;
}

__global__ void cuda_trace(Scene* scene, int max_x, int max_y, float scene_screen_ratio, float3* fb) {
	int sx = threadIdx.x + blockIdx.x * blockDim.x;
	int sy = threadIdx.y + blockIdx.y * blockDim.y;
	if ((sx >= max_x) || (sy >= max_y)) return;
	int pixel_index = sy * max_x * 3 + sx * 3;

	Camera cam = scene->current_cam;

	float3 rayOrigin = { 0,0,0 };
	float3 rayEnd = {
		-cam.filmPlane_width / 2 + sx * scene_screen_ratio,
		-cam.filmPlane_height / 2 + sy * scene_screen_ratio,
		-cam.rayTrace_plane_dist
	};

	float3 rayDir = rayEnd - rayOrigin;
	rayDir = normalize(rayDir);
	Ray ray = Ray(rayOrigin, rayDir);

	float3 sphere_frag_point = { 0,0,0 };
	float3 sphere_frag_normal = { 0,0,0 };

	float3 triangle_frag_point = { 0,0,0 };
	float3 triangle_frag_normal = { 0,0,0 };

	float sphere_min_dist = FLT_MAX;
	float triangle_min_dist = FLT_MAX;

	int hit_sphere_idx = cuda_raytrace_object<Sphere>(scene->spheres, scene->sphere_size, -1, ray, sphere_frag_point, sphere_frag_normal, sphere_min_dist);
	int hit_triangle_idx = cuda_raytrace_object<Triangle>(scene->triangles, scene->triangle_size, -1 , ray, triangle_frag_point, triangle_frag_normal, triangle_min_dist);

	if (hit_sphere_idx != -1 && sphere_min_dist <= triangle_min_dist) {
		Sphere rdr_sphere = scene->spheres[hit_sphere_idx];
		fb[pixel_index] = 
			cuda_calculate_color(scene->lights, scene->light_size, scene->spheres, scene->sphere_size, hit_sphere_idx, sphere_frag_point, sphere_frag_normal, rayOrigin, rdr_sphere);
	}
	else if (hit_triangle_idx != -1 && triangle_min_dist < sphere_min_dist){
		Triangle rdr_triangle = scene->triangles[hit_triangle_idx];
		fb[pixel_index] = 
			cuda_calculate_color(scene->lights, scene->light_size, scene->spheres, scene->sphere_size, -1, triangle_frag_point, triangle_frag_normal, rayOrigin, rdr_triangle);
	}
	else {
		fb[pixel_index] = BG_COLOR;
	}
}

__device__ void cuda_create_objects(Scene* scene) {

	Sphere* sphere1 = new Sphere();
	sphere1->origin = { -0.5, 1 ,-4 };
	sphere1->radius = 0.85;
	sphere1->color = { 0.0,0.6,.6 };
	sphere1->shininess = 16;

	Sphere* sphere2 = new Sphere();
	sphere2->origin = { 0.5, 0.6, -6 };
	sphere2->radius = .65;
	sphere2->color = { .4,.4,.4 };
	sphere2->shininess = 50;

	scene->addSphere(*sphere1);
	scene->addSphere(*sphere2);

	float quad_y_offset = -1.5;
	float quad_length = 6;
	float quad_x_offset = 0;
	float quad_z_offset = -5;
	float3 quad_v1 = { -quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v2 = { quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v3 = { quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };
	float3 quad_v4 = { -quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };
	Triangle* triangle1 = new Triangle(
		quad_v1,
		quad_v3,
		quad_v2,
		{ 0,1 },
		{ 1,0 },
		{ 1,1 }
	);
	triangle1->color = { 0, 1, 0 };
	Triangle* triangle2 = new Triangle(
		quad_v1,
		quad_v4,
		quad_v3,
		{ 0,1 },
		{ 0,0 },
		{ 1,0 }
	);
	triangle2->color = {1, 0, 0 };

	scene->addTriangle(*triangle1);
	scene->addTriangle(*triangle2);

	Light light1 = Light({ -8,4,6 }, { .4,.1,.05 }, { .05,0,0 });
	Light light2 = Light({ 3,12, 1 }, { .1,.15,.4 }, { 0,0,0.07 });
	scene->addLight(light2);
	scene->addLight(light1);
}

__global__ void cuda_set_scene(Scene* scene) {

	scene->current_cam.setCamera(
		{ -1.103, 1.312, 4 },		//position
		{ -1.256, 1.026, -3.945 },	//LookAt
		{ -1.103, 2.325, 4 },		//up
		nx,				            //width
		ny,				            //height
		1000,						//rayTrace plane distance to camera
		60						    //angle of view
	);

	cuda_create_objects(scene);
	scene_screen_ratio = scene->current_cam.filmPlane_width / nx;
}

void init_cuda(int screen_width, int screen_height) {

	nx = screen_width;
	ny = screen_height;

	cuda_fb_size = 3 * nx * ny * sizeof(float3);
	checkCudaErrors(cudaMalloc(&cuda_scene_ptr, sizeof(Scene)));
	checkCudaErrors(cudaMalloc(&cuda_fb_ptr, cuda_fb_size));

	cuda_set_scene << <1, 1 >> > (cuda_scene_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	time_begin = std::chrono::high_resolution_clock::now();
}

void free_cuda() {
	checkCudaErrors(cudaFree(cuda_scene_ptr));
	checkCudaErrors(cudaFree(cuda_fb_ptr));
}

__global__ void cuda_test_pos(Scene* scene, float t) {
	float A = 1;
	float f = 1;
	float shift = .5;
	scene->spheres[0].origin.y = 0.5 + 0.5 * sin(2 * PI * f * t / 5000 + shift);
}

void cuda_update(float3* output) {
	int tx = 8;
	int ty = 8;
	cuda_test_pos << <1, 1 >> > (cuda_scene_ptr, getTime(time_begin));
	checkCudaErrors(cudaDeviceSynchronize());
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	cuda_trace << <blocks, threads >> > (cuda_scene_ptr, nx, ny, scene_screen_ratio, cuda_fb_ptr);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(output, cuda_fb_ptr, cuda_fb_size, cudaMemcpyDeviceToHost);
}