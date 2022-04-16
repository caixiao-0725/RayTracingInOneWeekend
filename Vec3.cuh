#ifndef VEC3_H
#define VEC3_H
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#define  __hd__ static __host__ __device__


//һ�������£�����ѭ��������inline�����ֺܳ��ĺ������壬����inline����Ϊ��ᵼ���ڴ濪���ܴ�
struct Vec3 {
	double x,y,z;
};

inline __hd__ Vec3 make_vec3(double x_, double y_, double z_) {
	Vec3 temp_vec3;
	temp_vec3.x = x_;
	temp_vec3.y = y_;
	temp_vec3.z = z_;
	return temp_vec3;
};


//��д  operator +
inline __hd__ Vec3 operator+(Vec3 a,Vec3 b) {
	return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __hd__ void operator+=(Vec3& a, Vec3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __hd__ void operator+=(Vec3& a, double b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __hd__ Vec3 operator+(Vec3 a, double b)
{
	return make_vec3(a.x + b, a.y + b, a.z + b);
}

//��д  operator -

inline __hd__ Vec3 operator-(Vec3 a, Vec3 b) {
	return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __hd__ void operator-=(Vec3& a, Vec3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

inline __hd__ void operator-=(Vec3& a, double b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __hd__ Vec3 operator-(Vec3 a, double b)
{
	return make_vec3(a.x - b, a.y - b, a.z - b);
}

//��д  *

inline __hd__ Vec3 operator*(Vec3 a, Vec3 b) {
	return make_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __hd__ void operator*=(Vec3& a, Vec3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}

inline __hd__ void operator*=(Vec3& a, double b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __hd__ Vec3 operator*(Vec3 a, double b)
{
	return make_vec3(a.x * b, a.y * b, a.z * b);
}

inline __hd__ Vec3 operator*(double b, Vec3 a)
{
	return make_vec3(a.x * b, a.y * b, a.z * b);
}

//���

inline __hd__ double dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//���

inline __hd__ Vec3 cross(Vec3 a, Vec3 b) {
	return make_vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

//��һ��

inline __hd__ Vec3 normalize(Vec3 a) {
	double mod = sqrt(dot(a, a));
	return a * (1.0 / mod);
}

//cout���

inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
	return out << v.x << ' ' << v.y << ' ' << v.z;
}


#endif
