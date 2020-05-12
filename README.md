# CUDA_raytracer
ray tracer using CUDA for class CSCI-711.

Final video demo: https://youtu.be/Btuc_D0diA8

This is a CUDA Raytracing project implement sphere collision detection. The project uses cuda to compute each pixel's color and render using OPENGL/GLFW. 

There is a built exe file to execute the program.

Keyboard shortcut:

- s:		tgenerate new sphere
- +/- on numpad:	change light intensity
- t:		change tone reproduction method


Build with visual sutdio:
	Dependencies:
	
		-glew-2.1.0
		-glfw-3.3.2
		-cuda-10.2
	

Known Issue: 

    -No camera manipulation yet
	
    -No recursion used, results in lack of refraction calculation in reflection ray and reflection calculation in refraction ray. It seems that with limit stack size for GPU(hardware restriction), ray trace recursion can easily use up the stack and gives runtime error 719. I might change the code structure to see if I will be able to use the recursion in the future. 
