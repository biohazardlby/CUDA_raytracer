#include "utility.cuh"

class Ray {
public:
	float3 origin = { 0,0,0 };
	float3 direction = { 0,0,0 };
	float length = 1;
	__host__ __device__ Ray(float3 origin, float3 direciton);
	__host__ __device__ ~Ray();
};
class Light {
public:
	float3 position = { 0,0,0 };
	float3 color = { 0,0,0 };
	float3 ambientColor = { 0,0,0 };
	__host__ __device__ Light();
	__host__ __device__ Light(float3 position, float3 color, float3 ambientColor);
};


class Object {
public:
	OBJECT_TYPE primitive_type = OBJECT_TYPE::TRIANGLE;
	float3 origin = { 0,0,0 };

	//material properties
	float3 color = { 1,0,0 };
	float reflection = 0.2;
	float shininess = 10;

	//sphere properties
	float radius = 0;

	//triangle properties
	float2 a0 = { 0,0 }, a1 = { 0,0 }, a2 = { 0,0 };
	float3 v0 = { 0,0,0 }, v1 = { 0,0,0 }, v2 = { 0,0,0 };
	float checker_scale = 3;
	float3 checker_color_1 = { 0.5,0.5,0.2 };
	float3 checker_color_2 = { 0.2,0.5,0.5 };

	__device__ Object();
	__device__ ~Object();
	__device__ bool rayTrace(Ray ray, float3& hit_point, float3& normal);
	__device__ void set_sphere(float3 origin, float radius);
	__device__ void set_triangle(float3 vtx0, float3 vtx1, float3 vtx2, float2 anchor0, float2 anchor1, float2 anchor2);
	__device__ float3 obj_get_color(float3 frag_point);
private:
	__device__ float3 get_checker_color(float x, float y);
	__device__ void barycentric(float3 frag_point, float& u, float& v, float& w);
};

class Sphere : public Object {
public:
	__device__ bool rayTrace(Ray ray, float3& hit_point, float3& normal);
	__device__ float3 get_color(float3 frag_point);
};

class Triangle : public Object {
public:
	float checker_scale = 3;
	float3 checker_color_1 = {0.5,0.5,0.2};
	float3 checker_color_2 = {0.2,0.5,0.5};

	__device__ Triangle();
	__device__ Triangle(float3 vtx0, float3 vtx1, float3 vtx2);
	__device__ Triangle(float3 vtx0, float3 vtx1, float3 vtx2, float2 anchor1, float2 anchor2, float2 anchor3);
	__device__ bool rayTrace(Ray ray, float3& hit_point, float3& normal);
	__device__ float3 get_color(float3 frag_point);
private:
	__device__ float3 get_checker_color(float x, float y);
	__device__ void barycentric(float3 frag_point, float& u, float& v, float& w);
};

class Camera {
public:
	float3 position = { 0,0,0 };
	float3 lookAt = { 0,0,-1 };
	float3 up = { 0,1,0 };

	float screen_height = 800;
	float screen_width = 600;
	float rayTrace_plane_dist = 1000;
	float viewAngle = 60;

	float filmPlane_height = -1;
	float filmPlane_width = -1;

	__host__ __device__ void setCamera(float3 position, float3 lookat, float3 up, float width, float height, float raytrace_plane_dist, float view_angle);
	__host__ __device__ Camera(float3 position, float3 lookAt, float3 up, float width, float height, float rayTrace_plane_dist, float angleView);
	__host__ __device__ ~Camera();
};
class Scene {
public:
	Object objects[10];
	int object_size = 0;
	Sphere spheres[10];
	int sphere_size = 0;
	Triangle triangles[10];
	int triangle_size = 0;
	Light lights[10];
	int light_size = 0;
	Camera current_cam = Camera(
		{ 0,0,0 },		//position
		{ 0,0,-1 },	    //LookAt
		{ 0,1,0 },		//up
		800,			//width
		600,			//height
		1000,					//rayTrace plane distance to camera
		60						//angle of view
	);

	__host__ __device__ Scene();
	__host__ __device__ ~Scene();

	__host__ __device__ void addObject(Object& object);
	__host__ __device__ void addSphere(Sphere& sphere);
	__host__ __device__ void addTriangle(Triangle& triangle);
	__host__ __device__ void addLight(Light& light);
};