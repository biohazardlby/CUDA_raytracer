
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include <iostream>

#include "point_drawer.h"
#include "cuda_raytracer.h"


GLFWwindow* window;
int SCREEN_WIDTH = 1600;
int SCREEN_HEIGHT = 800;
Scene* scene_ptr;


int num_pixels = SCREEN_WIDTH * SCREEN_HEIGHT;
size_t fb_size = 3 * num_pixels * sizeof(float3);
float3* fb = (float3*)malloc(fb_size);


void render_loop()
{
	glClearColor(.1f,.1f,.1f,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPointSize(1);

	cuda_update(fb);
	//testRender(SCREEN_WIDTH, SCREEN_HEIGHT, fb);

	for (int sy = 0; sy < SCREEN_HEIGHT; sy++) {
		for (int sx = 0; sx < SCREEN_WIDTH; sx++) {
			int pixel_index = sy * 3 * SCREEN_WIDTH + sx * 3;
			float r = fb[pixel_index].x;
			float g = fb[pixel_index].y;
			float b = fb[pixel_index].z;
			Draw_Point(sx, sy, { r,g,b });
		}
	}
}

void create_window() {
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Ray Tracer Yes", NULL, NULL);
	if (!window)
	{
		fprintf(stderr, "Failed to open GLFW window\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
}
void setup_viewport() {
	// set up view
	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, SCREEN_WIDTH, 0.0, SCREEN_HEIGHT, 0.0, 1.0); // this creates a canvas you can do 2D drawing on

}
//void construct_scene() {
//	scene_ptr = new Scene();
//	scene_ptr->current_cam.setCamera(
//		{ -1.103, 1.312, 4 },		//position
//		{ -1.256, 1.026, -3.945 },	//LookAt
//		{ -1.103, 2.325, 4 },		//up
//		SCREEN_WIDTH,				//width
//		SCREEN_HEIGHT,				//height
//		1000,						//rayTrace plane distance to camera
//		60						//angle of view
//	);
//
//	Sphere* sphere1 = new Sphere();
//	sphere1->origin = { -0.5, 1 ,-5 };
//	sphere1->radius = 0.85;
//	sphere1->color = { 0.0,0.0,.6 };
//	sphere1->shininess = 16;
//
//	Sphere* sphere2 = new Sphere();
//	sphere2->origin = { 0.5, 0.6, -6 };
//	sphere2->radius = .65;
//	sphere2->color = { .8,.8,.8 };
//	sphere2->shininess = 50;
//
//	scene_ptr->addSphere(*sphere1);
//	scene_ptr->addSphere(*sphere2);
//
//	Light light1 = Light({ 100,100,-2 }, { .7,.4,.05 }, { .05,0,0 });
//	scene_ptr->addLight(light1);
//	Light light2 = Light({ 20,120, 1 }, { .1,.15,.4 }, { 0,0,.05 });
//	scene_ptr->addLight(light2);
//
//}

int main(int argc, char* argv[])
{

	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		exit(EXIT_FAILURE);
	}

	create_window();
	setup_viewport();

	init_cuda( SCREEN_WIDTH, SCREEN_HEIGHT);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		// Draw gears
		render_loop();

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Terminate GLFW
	glfwTerminate();
	free_cuda();
	// Exit program
	exit(EXIT_SUCCESS);
}

