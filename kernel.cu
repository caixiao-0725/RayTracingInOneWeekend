#include <iostream>
#include"Vec3.cuh"
#define HEIGHT 256
#define WIDTH 256
using point3 = Vec3;   // 3D point
using color = Vec3;    // RGB color

struct Sphere {
	Vec3 position, color, emission;
};

__constant__  Sphere spheres[] = {
	 { 1e5 + 1,40.8,81.6 }, { 0.0, 0.0, 0.0 }, { 0.75, 0.25, 0.25 }
};

inline void write_pixar(FILE* f,Vec3 a) {
	fprintf(f,"%d %d %d ", a.x, a.y, a.z);
}
int main() {
	FILE* f = fopen("test.ppm","w");
	fprintf(f, "P3\n%d %d\n%d\n",HEIGHT,WIDTH,255);
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			color new_color = make_vec3(i,j,100);
			write_pixar(f,new_color);
		}
	}
	fclose(f);
	return 0;
}