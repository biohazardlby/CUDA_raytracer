#include "cuda_raytracer.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__managed__ Scene* cuda_scene_ptr;
__device__ float3* cuda_fb_ptr;
Scene* scene_ptr = (Scene*)malloc(sizeof(Scene));

__device__ size_t cuda_fb_size;
__managed__ float scene_screen_ratio;
__managed__ int nx, ny;
__device__ float3 BG_COLOR = { 0.8,0.8,0.8 };
__managed__ float3 GRAVITY = { 0,-9.8,0 };
__managed__ float FLOOR_LEVEL = -1.5;

float last_frame_time = 0;
int tx = 16;
int ty = 16;

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


__device__ int raytrace_object(Ray ray, Object* objects, int object_size,int skip_index, float3 &frag_point, float3 &frag_normal, float &min_dist) {
	int hit_idx = -1;
	for (int i = 0; i < object_size; i++) {
		if (i == skip_index) continue;
		Object cur_object = objects[i];
		float3 hit_point = { 0,0,0 };
		float3 hit_normal = { 0,0,0 };
		if (cur_object.Raytrace(ray, hit_point, hit_normal)) {
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
		Ray light_ray = Ray(frag_point, cur_light.get_position());
		float3 block_point = { 0,0,0 };
		float3 block_normal = { 0,0,0 };
		bool blocked = false;
		for (int blocker_idx = 0; blocker_idx < object_size; blocker_idx++) {
			if (blocker_idx == hit_object_idx) continue;
			Object block_object = objects[blocker_idx];
			if (block_object.Raytrace(light_ray, block_point, block_normal)) {
				blocked = true;
				break;
			}
		}
		if (!blocked) {
			final_color = final_color + phongShading(cur_light.get_position(), frag_normal, frag_point, viewPos, cur_light.get_color(), cur_light.get_ambient_color(), hit_object.GetColor(frag_point), hit_object.shininess);
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
	int pixel_index = sy * max_x  + sx ;

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
	s1->SetSphere({ -0.5, 1 ,-4 }, 0.85);
	s1->color = { .6,.6,0 };
	s1->shininess = 16;
	s1->reflectiveness = 0;
	s1->transparency = 0.5;
	s1->refractive_index = 1.1;

	Object* s2 = new Object();
	s2->SetSphere({ 0.5, 0.6, -6 }, .65);
	s2->color = { 0,.4,.4 };
	s2->shininess = 50;
	s2->reflectiveness = .4;

	scene->addObject(*s1);
	scene->addObject(*s2);

	float quad_y_offset = FLOOR_LEVEL;
	float quad_length = 6;
	float quad_x_offset = 0;
	float quad_z_offset = -5;

	float3 quad_v1 = { -quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v2 = { quad_length/2 + quad_x_offset,quad_y_offset,-quad_length/2 + quad_z_offset };
	float3 quad_v3 = { quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };
	float3 quad_v4 = { -quad_length/2 + quad_x_offset,quad_y_offset,quad_length/2 + quad_z_offset };


	Object* t1 = new Object();
	t1->SetTriangle(
		quad_v1,
		quad_v3,
		quad_v2,
		{ 0,1 },
		{ 1,0 },
		{ 1,1 }
	);
	Object* t2 = new Object();
	t2->SetTriangle(
		quad_v1,
		quad_v4,
		quad_v3,
		{ 0,1 },
		{ 0,0 },
		{ 1,0 }
	);
	scene->addObject(*t1);
	scene->addObject(*t2);

	Light light1 = Light({ -8,4,6 }, { .4,.1,.05 },0.5, { .05,0,0 });
	Light light2 = Light({ 3,12, 1 }, { .1,.15,.4 },0.5, { 0,0,0.07 });
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

	cuda_fb_size = nx * ny * sizeof(float3);
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


bool check_collision(Object* objects, int idx_1, int idx_2) {
	Object& o1 = objects[idx_1];
	Object& o2 = objects[idx_2];
	float floor_bounce = 0.9;
	float R = 0.8;
	if (o1.primitive_type == OBJECT_TYPE::SPHERE && o2.primitive_type == OBJECT_TYPE::SPHERE) {

		float3 vecd = o1.origin - o2.origin;
		float distance = length(vecd);
		float sum_radius = o1.radius + o2.radius;
		if (distance <= sum_radius) {

			float3 U1x, U1y, U2x, U2y, V1x, V1y, V2x, V2y;
			float m1, m2, x1, x2;
			float3 v1temp, v1, v2, v1x, v2x, v1y, v2y, x(o1.origin - o2.origin);

			x = normalize(x);
			v1 = o1.speed;
			x1 = dot(x,v1);
			v1x = x * x1;
			v1y = v1 - v1x;
			m1 = o1.radius * 5;

			x = x * -1;
			v2 = o2.speed;
			x2 = dot(x,v2);
			v2x = x * x2;
			v2y = v2 - v2x;
			m2 = o2.radius * 5;

			o1.speed = float3(v1x * (m1 - m2) / (m1 + m2) + v2x * (2 * m2) / (m1 + m2) + v1y);
			o2.speed = float3(v1x * (2 * m1) / (m1 + m2) + v2x * (m2 - m1) / (m1 + m2) + v2y);

			o1.origin = o1.origin - (x * (sum_radius - distance + kEpsilon));
			o2.origin = o2.origin + (x * (sum_radius - distance + kEpsilon));
			return true;
		}
		else {
			return false;
		}
	}
	else if (o1.primitive_type == OBJECT_TYPE::SPHERE) {
		if ((o1.origin.y - o1.radius) < FLOOR_LEVEL) {
			o1.speed.y = -o1.speed.y * floor_bounce;
			o1.origin.y = o1.radius + FLOOR_LEVEL + kEpsilon;
			return true;
		}
	}
	else if (o2.primitive_type == OBJECT_TYPE::SPHERE) {

		if (o2.origin.y - o2.radius < FLOOR_LEVEL){
			o2.speed.y = -o2.speed.y * floor_bounce;
			o2.origin.y = o2.radius + FLOOR_LEVEL + kEpsilon;
			return true;
		}
	}
	return false;
}

void solve_movement(Object* objects, int object_size, int t_idx, float deltaTime) {

	Object& cur_object = objects[t_idx];
	//only sphere movement for now
	if (cur_object.primitive_type == OBJECT_TYPE::TRIANGLE) return;
	float3 last_position = cur_object.origin;

	cur_object.origin = cur_object.origin + cur_object.speed * deltaTime / 1000;

	//check if the destiny collide
	for (int i = 0; i < object_size; i++) {
		if (i == t_idx) continue;
		check_collision(objects, i, t_idx);
	}
}

void update_position(Scene& scene, float deltaTime) {

	for (int i = 0; i < scene.object_size; i++) {
		Object& cur_obj = scene.objects[i];
		cur_obj.speed = cur_obj.speed + cur_obj.accel * deltaTime;
	}
	for (int i = 0; i < scene.object_size; i++) {
		solve_movement(scene.objects, scene.object_size, i, deltaTime);
	}
}
void apply_accel(Scene& scene) {
	for (int i = 0; i < scene.object_size; i++) {
		Object& cur_obj = scene.objects[i];
		if (cur_obj.primitive_type == OBJECT_TYPE::SPHERE) {
			cur_obj.accel =  GRAVITY / 1000;
		}
	}
}
void calculate_physics(Scene &scene, float time, float deltaTime) {
	apply_accel(scene);
	update_position(scene, deltaTime);
}

void cuda_update(float3* output) {
	float time = getTime(time_begin);
	float deltaTime = time - last_frame_time;

	int obj_size = MAX_OBJECT_SIZE;

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	cuda_trace_object << <blocks, threads >> > (cuda_scene_ptr, nx, ny, scene_screen_ratio, cuda_fb_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(output, cuda_fb_ptr, cuda_fb_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(scene_ptr, cuda_scene_ptr, sizeof(Scene), cudaMemcpyDeviceToHost);
	calculate_physics(*scene_ptr, time, deltaTime);
	cudaMemcpy(cuda_scene_ptr, scene_ptr, sizeof(Scene), cudaMemcpyHostToDevice);
	last_frame_time = time;
}
__global__ void cuda_generate_sphere(Object* sphere) {

	cuda_scene_ptr->addObject(*sphere);
}

void generate_sphere()
{
	Object* sphere = new Object;
	float x = 2 - random_float() * 4;
	float y = 4 + random_float() * 3;
	float z = -5 - random_float() * 4;

	float radius = 0.1 + random_float() * 0.3 + 0.25;
	sphere->SetSphere({ x,y,z }, radius);

	sphere->color = { random_float(), random_float(), random_float() };
	sphere->shininess = random_float();
	sphere->transparency = random_float();
	sphere->reflectiveness = random_float();
	sphere->refractive_index = random_float();
	sphere->speed = { -x * 0.5f , 0, (z+5)*0.5f };

	Object* gpu_sphere;
	checkCudaErrors(cudaMalloc(&gpu_sphere, sizeof(Object)));
	checkCudaErrors(cudaMemcpy(gpu_sphere, sphere, sizeof(Object), cudaMemcpyHostToDevice));
	cuda_generate_sphere << <1, 1 >> > (gpu_sphere);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void cuda_add_light_power(float amount)
{
	for (int i = 0; i < cuda_scene_ptr->light_size; i++) {
		cuda_scene_ptr->lights[i].set_power(cuda_scene_ptr->lights[i].get_power() + amount);
	}
}

void add_light_power(float amount) {
	cuda_add_light_power << <1, 1 >> > (amount);
	checkCudaErrors(cudaDeviceSynchronize());
}
