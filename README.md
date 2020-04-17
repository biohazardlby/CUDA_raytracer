# CUDA_raytracer
ray tracer using CUDA for class CSCI-711

Current progress video: https://www.youtube.com/watch?v=bIqZGtwRsuo&feature=youtu.be
Known Issue: 
    -No camera manipulation yet
    -No recursion used, results in lack of refraction calculation in reflection ray and reflection calculation in refraction ray. It seems that with limit stack size for GPU(hardware restriction), ray trace recursion can easily use up the stack and gives runtime error 719. I might change the code structure to see if I will be able to use the recursion in the future. 
