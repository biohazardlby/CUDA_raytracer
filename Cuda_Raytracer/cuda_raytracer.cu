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
		std::cerr << "Err name: " << cudaGetErrorName(result) << "\n" << cudaGetErrorString(result);
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

		
__device__ int raytrace_object(Ray ray, Object* objects, int object_size,int skip_index, float3 &frag_point, float3 &frag_normal, float &min_dist) {
	int hit_idx = -1;
	for (int i = 0; i < object_size; i++) {
		if (i == skip_index) continue;
		Object cur_object = objects[i];
		float3 hit_point = { 0,0,0 };
		float3 hit_normal = { 0,0,0 };
		if (cur_object.rayTrace(ray, hit_point, hit_normal)) {
			float dist = length(hit_point - ray.origin);
			if (dist <= min_dist) {
				min_dist = dist;
				frag_point = hit_point;
				frag_normal = hit_normal;
				hit_idx = i;
			}
		}
	}
	return hit_idx;
}
__device__ float3 calculate_light_object(Object* objects, int object_size, int hit_object_idx, float3 frag_point, float3 frag_normal, float3 viewPos, Light* lights, int light_size) {
	float3 final_color = { 0,0,0 };
	Object hit_object = objects[hit_object_idx];
	for (int light_idx = 0; light_idx < light_size; light_idx++) {
		Light cur_light = lights[light_idx];
		Ray light_ray = Ray(frag_point, cur_light.position);
		float3 block_point = { 0,0,0 };
		float3 block_normal = { 0,0,0 };
		bool blocked = false;
		for (int blocker_idx = 0; blocker_idx < object_size; blocker_idx++) {
			if (blocker_idx == hit_object_idx) continue;
			Object block_object = objects[blocker_idx];
			if (block_object.rayTrace(light_ray, block_point, block_normal)) {
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			final_color = final_color + phongShading(cur_light.position, frag_normal, frag_point, viewPos, cur_light.color, cur_light.ambientColor, hit_object.obj_get_color(frag_point), hit_object.shininess);
		}
	}
	return final_color;
}
__device__ void recursion(int i) {
	if (i == 0) return;
	printf("%d\n", i);
	recursion(i - 1);
}

__device__ float3 illuminate_object(float3 viewPos, Ray cam_ray, Object* objects, int object_size, Light* lights, int light_size, int kr, int kt) {
	float3 frag_point = { 0,0,0 };
	float3 frag_normal = { 0,0,0 };
	float3 final_color = { 0,0,0 };

	Ray reflection_ray = cam_ray;
	Ray refraction_ray = cam_ray;
	//start with objects color

	float reflectiveness_stack = 1;
	float transparency_stack = 1;
	float min_dist = FLT_MAX;
	int rdr_obj_idx = raytrace_object(cam_ray, objects, object_size, -1, frag_point, frag_normal, min_dist);


	if (rdr_obj_idx != -1) {
		transparency_stack *= 1 - objects[rdr_obj_idx].transparency;
		final_color = final_color + calculate_light_object(objects, object_size, rdr_obj_idx, frag_point, frag_normal, viewPos, lights, light_size) * transparency_stack;
		reflectiveness_stack *= objects[rdr_obj_idx].reflectiveness;

		reflection_ray.origin = frag_point;
		reflection_ray.direction = reflect(cam_ray.direction, frag_normal);

		refraction_ray.origin = frag_point;
		refraction_ray.direction = refract(cam_ray.direction, frag_normal, 1, objects[rdr_obj_idx].refractive_index);
	}
	else {
		return BG_COLOR;
	}

	float3 refraction_frag_point = frag_point;
	float3 refraction_frag_normal = frag_normal;


	//after some experience, cuda has very limited stack and recursive function seems to always exceed stack limits for some reason. In this case, I'm using while loop
	//to calculate the reflection and refraction. 
	//This, however, will result in lack of reflection ray in refraction ray, and refraction ray in reflection ray, until I find what is causing the recursion to fail

	int reflect_obj_idx = rdr_obj_idx;
	while (kr > 0) {
		min_dist = FLT_MAX;
	    reflect_obj_idx = raytrace_object(reflection_ray, objects, object_size, reflect_obj_idx, frag_point, frag_normal, min_dist);
		if (reflect_obj_idx != -1) {
			transparency_stack *= 1 - objects[reflect_obj_idx].transparency;
			float3 added_color = calculate_light_object(objects, object_size, reflect_obj_idx, frag_point, frag_normal, viewPos, lights, light_size);
			final_color = final_color + added_color * reflectiveness_stack * transparency_stack;
			reflection_ray.origin = frag_point;
			reflection_ray.direction = reflect(reflection_ray.direction, frag_normal);
			reflectiveness_stack *= objects[reflect_obj_idx].reflectiveness;
			kr--;
		}
		else {
			final_color = final_color + BG_COLOR * reflectiveness_stack;
			break;
		}
	}
	bool ray_inside = true;
	transparency_stack *= objects[rdr_obj_idx].transparency;
	int refraction_obj_idx = rdr_obj_idx;
	while (kt > 0) {
		min_dist = FLT_MAX;
		refraction_obj_idx = raytrace_object(refraction_ray, objects, object_size, -1, refraction_frag_point, refraction_frag_normal, min_dist);
		if (refraction_obj_idx == -1) {
			final_color = final_color + BG_COLOR * transparency_stack;
			break;
		}
		else {

			float3 added_color = calculate_light_object(objects, object_size, refraction_obj_idx, refraction_frag_point, refraction_frag_normal, viewPos, lights, light_size);

			final_color = final_color + added_color * transparency_stack;
			transparency_stack *= objects[refraction_obj_idx].transparency;

			float ni = 1, nt = objects[refraction_obj_idx].refractive_index;

			refraction_ray.origin = refraction_frag_point;
			refraction_ray.direction = refract(refraction_ray.direction, refraction_frag_normal, ni, nt);
			ray_inside = !ray_inside;

			kt--;
		}
	}
	return final_color;
}
__global__ void cuda_trace_object(Scene* scene, int max_x, int max_y, float scene_screen_ratio, float3* fb) {
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
	Ray camera_ray = Ray(rayOrigin, rayDir);
	fb[pixel_index] = illuminate_object(rayOrigin, camera_ray, scene->objects, scene->object_size, scene->lights, scene->light_size,2,4);
}
__device__ void cuda_create_objects(Scene* scene) {


	Object* s1 = new Object();
	s1->set_sphere({ -0.5, 1 ,-4 }, 0.85);
	s1->color = { .6,.6,0 };
	s1->shininess = 16;
	s1->reflectiveness = 0;
	s1->transparency = 0.5;
	s1->refractive_index = 1.1;

	Object* s2 = new Object();
	s2->set_sphere({ 0.5, 0.6, -6 }, .65);
	s2->color = { 0,.4,.4 };
	s2->shininess = 50;
	s2->reflectiveness = .4;

	scene->addObject(*s1);
	scene->addObject(*s2);

	float quad_y_offset = -1.5;
	float quad_length = 6;
	float quad_x_offset = 0;
	float quad_z_offset = -5;

	float3 quad_v1 = { -quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v2 = { quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v3 = { quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };
	float3 quad_v4 = { -quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };


	Object* t1 = new Object();
	t1->set_triangle(
		quad_v1,
		quad_v3,
		quad_v2,
		{ 0,1 },
		{ 1,0 },
		{ 1,1 }
	);
	Object* t2 = new Object();
	t2->set_triangle(
		quad_v1,
		quad_v4,
		quad_v3,
		{ 0,1 },
		{ 0,0 },
		{ 1,0 }
	);
	scene->addObject(*t1);
	scene->addObject(*t2);

	Light light1 = Light({ -8,4,6 }, { .4,.1,.05 }, { .05,0,0 });
	Light light2 = Light({ 3,12, 1 }, { .1,.15,.4 }, { 0,0,0.07 });
	scene->addLight(light2);
	scene->addLight(light1);

	//legacy objects
	/*
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

*/

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

__global__ void kernel(int i) {
	recursion(i);
}

void init_cuda(int screen_width, int screen_height) {
	//set stack size
	size_t max_stack_size;


	checkCudaErrors(cudaThreadGetLimit(&max_stack_size, cudaLimitStackSize));
	checkCudaErrors(cudaDeviceSynchronize());
	printf("max stack size = %d\n", max_stack_size);

	checkCudaErrors(cudaThreadSetLimit(cudaLimitStackSize, max_stack_size));
	checkCudaErrors(cudaDeviceSynchronize());

	//kernel << <1, 1 >> > (10);
	//checkCudaErrors(cudaDeviceSynchronize());

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
	scene->objects[0].origin.y = 0.5 + 0.5 * sin(2 * PI * f * t / 5000 + shift);
}

void cuda_update(float3* output) {
	int tx = 8;
	int ty = 8;
	cuda_test_pos << <1, 1 >> > (cuda_scene_ptr, getTime(time_begin));
	checkCudaErrors(cudaDeviceSynchronize());
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	cuda_trace_object << <blocks, threads >> > (cuda_scene_ptr, nx, ny, scene_screen_ratio, cuda_fb_ptr);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(output, cuda_fb_ptr, cuda_fb_size, cudaMemcpyDeviceToHost);
}