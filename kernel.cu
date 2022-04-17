#include <iostream>
#include"Vec3.cuh"
#include"Ray.cuh"
#include <device_launch_parameters.h>

#define HEIGHT 768
#define WIDTH 1024
typedef unsigned int uint;
using point3 = Vec3;   // 3D point
using color = Vec3;    // RGB color

struct Sphere {
	Vec3 position;
	double radius;
	__device__ double hit(Ray r) {
		Vec3 M = r.pos - position;
		double a = dot(r.dir, r.dir);
		double h = dot(r.dir, M);
		double c = dot(M, M)-radius*radius;
		double delta = h * h -  a * c;
		if (delta < 0) {
			return -1.0;
		}
		else {
			double x1 = (-h - sqrt(delta)) / a;
			double x2 = (-h + sqrt(delta)) / a;
			if (x1 > 0) {
				return x1;
			}
			else if(x2 > 0)
			{
				return x2;
			}
			else
			{
				return -1.0;
			}
		}
	}
};

__constant__ Sphere spheres[]{
	{{0.0,0.0,-1.0},0.5}
};

__device__ Vec3 Ray_background(Ray a) {
	double t = spheres[0].hit(a);
	if (t > 0.0) {
		Vec3 N = normalize(a.pos + a.dir*t - make_vec3(0, 0, -1));
		return 0.5 * make_vec3(N.x + 1.0, N.y + 1.0, N.z + 1.0);
	}
	Vec3 unit_dir = normalize(a.dir);
	t = 0.5 * (unit_dir.y + 1.0);
	return (1.0 - t) * make_vec3(1.0, 1.0, 1.0) + t * make_vec3(0.5, 0.7, 1.0);
}

inline void write_pixar(FILE* f,Vec3 a) {
	int x = int(a.x);
	int y = int(a.y);
	int z = int(a.z);
	fprintf(f,"%d %d %d ", int(a.x), int(a.y), int(a.z));
}


inline __device__ double clamp(double x,double y,double z) {
	if (x < y) {
		return y;
	}
	else if(x>z)
	{
		return z;
	}
	else
	{
		return x;
	}
}

inline __device__ Vec3 clamp(Vec3 a, double y, double z) {
	return make_vec3(clamp(a.x, y, z), clamp(a.y, y, z), clamp(a.z, y, z));
}

__global__ void Ray_tracing_kernal(Vec3 *d_output) {
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = (HEIGHT - y - 1) * WIDTH + x;

	const double aspect_ratio = double(WIDTH) / HEIGHT;
	double viewport_height = 2.0;
	double viewport_width = aspect_ratio * viewport_height;
	double focal_length = 1.0;

	auto origin = make_vec3(0, 0, 0);
	auto horizontal = make_vec3(viewport_width, 0, 0);
	auto vertical = make_vec3(0, viewport_height, 0);
	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_vec3(0, 0, focal_length);

	auto u = double(x) / (WIDTH - 1);
	auto v = double(y) / (HEIGHT - 1);
	Ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	color pixel = Ray_background(r);
	d_output[i] = clamp(pixel, 0.0, 1.0);
}

int main() {

	FILE* f = fopen("test.ppm","w");
	fprintf(f, "P3\n%d %d\n%d\n",WIDTH,HEIGHT,255);

	Vec3* host_output = new Vec3[WIDTH*HEIGHT];
	Vec3* device_output;
	cudaMalloc(&device_output, WIDTH * HEIGHT * sizeof(Vec3));

	dim3 block(8,8,1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);
	Ray_tracing_kernal<<<grid ,block>>> (device_output);
	printf("开始复制output");
	cudaMemcpy(host_output, device_output, WIDTH * HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost);
	cudaFree(device_output);
	
	for (int i = 0; i < HEIGHT*WIDTH; i++) {
		write_pixar(f,host_output[i]*255);
	}
	fclose(f);
	return 0;
}
