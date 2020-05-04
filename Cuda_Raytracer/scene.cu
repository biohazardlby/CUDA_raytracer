#include "scene.cuh"

__host__ __device__ Ray::Ray(float3 origin, float3 dir) {
	this->origin = origin;
	this->direction = normalize(dir);
}

__host__ __device__ Ray::~Ray()
{
}

__host__ __device__ Light::Light()
{
}

__host__ __device__ Light::Light(float3 p, float3 c, float3 ac) {
	position = p;
	color = c;
	ambientColor = ac;
}
__host__ __device__ Light::Light(float3 p, float3 c,float pow, float3 ac) {
	position = p;
	color = c;
	ambientColor = ac;
}

__host__ __device__ float3 Light::get_color()
{
	return color * power;
}

__host__ __device__ float3 Light::get_ambient_color()
{
	return  ambientColor;
}

__host__ __device__ float3 Light::get_position()
{
	return position;
}

__host__ __device__ float Light::get_power() {
	return power;
}

__host__ __device__ void Light::set_power(float p)
{
	power = p;
}

__host__ __device__ Object::Object()
{
}
__host__ __device__ Object::~Object()
{
}

__host__ __device__ bool Object::Raytrace(Ray ray, float3 &hit_point, float3 &normal) {
	switch (primitive_type) {
	case OBJECT_TYPE::TRIANGLE:
	{
		//MOLLER_TRUMBORE 
		float3 v0v1 = v1 - v0;
		float3 v0v2 = v2 - v0;
		float3 pvec = cross(ray.direction, v0v2);
		float det = dot(v0v1, pvec);
		float t, u, v;
		// CULLING 
			// if the determinant is negative the triangle is backfacing
			// if the determinant is close to 0, the ray misses the triangle
		if (det < 0.00) return false;

		// ray and triangle are parallel if det is close to 0
		if (fabs(det) < kEpsilon) return false;

		float invDet = 1 / det;

		float3 tvec = ray.origin - v0;
		u = dot(tvec, pvec) * invDet;
		if (u < 0 || u > 1) return false;

		float3 qvec = cross(tvec, v0v1);
		v = dot(ray.direction, qvec) * invDet;
		if (v < 0 || u + v > 1) return false;

		t = dot(v0v2, qvec) * invDet;

		hit_point = ray.origin + t * ray.direction;
		normal = normalize(cross(v0v1, v0v2));
		return true;
		break;
	}
	case OBJECT_TYPE::SPHERE:
	{
		float3 oc = ray.origin - this->origin;
		float a = dot(ray.direction, ray.direction);
		float b = 2.0 * dot(oc, ray.direction);
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4 * a * c;

		if (discriminant < 0) return false;

		float numerator = -b - sqrt(discriminant);
		if (numerator > kEpsilon) {
			hit_point = ray.origin + ray.direction * numerator / (2 * a);
			normal = normalize(hit_point - this->origin);
			return true;
		}

		numerator = -b + sqrt(discriminant);
		if (numerator > kEpsilon) {
			hit_point = ray.origin + ray.direction * numerator / (2 * a);
			normal = normalize(hit_point - this->origin);
			return true;
		}
		return false;
		break;
	}
	}
	return false;
}
__host__ __device__ void Object::SetSphere(float3 origin, float radius)
{
	this->radius = radius;
	this->origin = origin;
	primitive_type = OBJECT_TYPE::SPHERE;
}
__host__ __device__ void Object::SetTriangle(float3 vtx0, float3 vtx1, float3 vtx2, float2 anchor0, float2 anchor1, float2 anchor2)
{
	v0 = vtx0;
	v1 = vtx1;
	v2 = vtx2;
	a0 = anchor0;
	a1 = anchor1;
	a2 = anchor2;
	primitive_type = OBJECT_TYPE::TRIANGLE;
}
__host__ __device__ void Object::Barycentric(float3 frag_point, float& u, float& v, float& w) {
	float3 v0v1 = v1 - v0, v1v2 = v2 - v0, v0vp = frag_point - v0;
	float d00 = dot(v0v1, v0v1);
	float d01 = dot(v0v1, v1v2);
	float d11 = dot(v1v2, v1v2);
	float d20 = dot(v0vp, v0v1);
	float d21 = dot(v0vp, v1v2);
	float denom = d00 * d11 - d01 * d01;
	v = (d11 * d20 - d01 * d21) / denom;
	w = (d00 * d21 - d01 * d20) / denom;
	u = 1.0f - v - w;
}

__host__ __device__ float3 Object::GetCheckerColor(float x, float y)
{
	float z = 1;
	x = abs(x);
	y = abs(y);
	x = modf(x * checker_scale, &z);
	y = modf(y * checker_scale, &z);
	if (x < 0.5) {
		if (y < 0.5) {
			return checker_color_1;
		}
		else {
			return checker_color_2;
		}
	}
	else {
		if (y < 0.5) {
			return checker_color_2;
		}
		else {
			return checker_color_1;
		}
	}
}



__host__ __device__ float3 Object::GetColor(float3 frag_point)
{
	switch (primitive_type)
	{
	case OBJECT_TYPE::TRIANGLE:
		float u, v, w;
		Barycentric(frag_point, u, v, w);
		float2 p = a0 * u + a1 * v + a2 * w;
		return GetCheckerColor(p.x, p.y);;
	case OBJECT_TYPE::SPHERE:
		return color;
	}
}

__host__ __device__ void Camera::setCamera(float3 position, float3 lookat, float3 up, float width, float height, float raytrace_plane_dist, float viewAngle)
{
	this->position = position;
	this->lookAt = lookAt;
	this->up = up;
	this->screen_height = height;
	this->screen_width = width;
	this->rayTrace_plane_dist = raytrace_plane_dist;
	this->viewAngle = viewAngle;
	if (width >= height) {
		filmPlane_height = 2 * tan(deg_2_rad(viewAngle / 2)) * rayTrace_plane_dist;
		filmPlane_width = filmPlane_height * (screen_width / screen_height);
	}
	else {
		filmPlane_width = 2 * tan(deg_2_rad(viewAngle / 2)) * rayTrace_plane_dist;
		filmPlane_height = filmPlane_width * (screen_height / screen_width);
	}	
}

__host__ __device__ Camera::Camera(float3 position, float3 lookAt, float3 up, float width, float height, float rayTrace_plane_dist, float viewAngle)
{
	this->position = position;
	this->lookAt = lookAt;
	this->up = up;
	this->screen_height = height;
	this->screen_width = width;
	this->rayTrace_plane_dist = rayTrace_plane_dist;
	this->viewAngle = viewAngle;
	if (width >= height) {
		filmPlane_height = 2 * tan(deg_2_rad(viewAngle / 2)) * rayTrace_plane_dist;
		filmPlane_width = filmPlane_height * (screen_width / screen_height);
	}
	else {
		filmPlane_width = 2 * tan(deg_2_rad(viewAngle / 2)) * rayTrace_plane_dist;
		filmPlane_height = filmPlane_width * (screen_height / screen_width);
	}
}
__host__ __device__ Camera::~Camera() {

}
__host__ __device__ Scene::Scene()
{
}
__host__ __device__ Scene::~Scene() {

}
__host__ __device__ void Scene::addObject(Object& object) {
	if (object_size < MAX_OBJECT_SIZE) {
		objects[object_size] = object;
		object_size++;
	}
}
__host__ __device__ void Scene::addLight(Light& light)
{
	lights[light_size] = light;
	light_size++;
}
