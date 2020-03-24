	#pragma once
#include <GL/glew.h>
#include "utility.cuh"

void Draw_Point(GLint x, GLint y, float3 color)
{
	glColor3f(color.x, color.y, color.z);

	glBegin(GL_POINTS);
	glVertex2i(x, y);
	glEnd();
}
