
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include <iostream>

#include "point_drawer.h"
#include "cuda_raytracer.h"


GLFWwindow* window;
int SCREEN_WIDTH = 1600;
int SCREEN_HEIGHT = 800;
float LdMax = 1;
TR_TYPE TR_method = TR_TYPE::WARD;


int num_pixels = SCREEN_WIDTH * SCREEN_HEIGHT;
size_t fb_size =  num_pixels * sizeof(float3);
float3* fb = (float3*)malloc(fb_size);
float* luminance = (float*)malloc(num_pixels * sizeof(float));

void tone_reproduction(float3* pixels, TR_TYPE type, float LdMax) {
	float La = 0;
	for (int i = 0; i < num_pixels; i++) {
		luminance[i] = getLuminance(pixels[i]);
		La +=  log(0.001 + luminance[i]);
	}
	La = exp(La / num_pixels);

	switch (type)
	{
	case TR_TYPE::WARD:
	{
		float numerator = 1.219 + powf(LdMax / 2, 0.4);
		float denominator = 1.219 + powf(La, 0.4);
		float sf = powf(numerator / denominator, 2.5);
		for (int i = 0; i < num_pixels; i++) {
			pixels[i] = pixels[i] * sf;
		}
	}
	break;
	case TR_TYPE::REINHARD:
	{
		float a = 0.18;
		for (int i = 0; i < num_pixels; i++) {
			float Rs = a * pixels[i].x / La;
			float Gs = a * pixels[i].y / La;
			float Bs = a * pixels[i].z / La;


			float Rr = Rs / (1 + Rs);
			float Gr = Gs / (1 + Gs);
			float Br = Bs / (1 + Bs);


			pixels[i].x = Rr * LdMax;
			pixels[i].y = Gr * LdMax;
			pixels[i].z = Br * LdMax;


			//float Ls = a * luminance[i] / La;
			//float Lr = Ls / (1 + Ls);
			//pixels[i] = pixels[i] * Lr;

		}
		break;
	}
	}
}

void key_callBack(GLFWwindow* window, int key, int scancode, int action, int mod) {
	if (key == GLFW_KEY_T && action == GLFW_PRESS) {
		TR_method = static_cast<TR_TYPE>((static_cast<int>(TR_method) + 1) % (static_cast<int>(TR_TYPE::DUMMY)));
		printf("%d", static_cast<int>(TR_method));
	}
	if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS) {
		add_light_power(10);
	}
	if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS) {
		add_light_power(-10);
	}
}

void render_loop()
{
	glClearColor(.1f,.1f,.1f,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPointSize(1);

	cuda_update(fb);
	tone_reproduction(fb, TR_method, LdMax);

	for (int sy = 0; sy < SCREEN_HEIGHT; sy++) {
		for (int sx = 0; sx < SCREEN_WIDTH; sx++) {
			int pixel_index = sy *  SCREEN_WIDTH + sx ;
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
	glOrtho(0.0, SCREEN_WIDTH, 0.0, SCREEN_HEIGHT, 0.0, 1.0); 

}

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
	glfwSetKeyCallback(window, key_callBack);

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

