#ifndef RAY_H
#define RAY_H

#include "Vec3.cuh"
class Ray {
public:
	Vec3 pos,dir;
	__device__ Ray(Vec3 pos_, Vec3 dir_) :pos(pos_), dir(dir_) {};

};



#endif