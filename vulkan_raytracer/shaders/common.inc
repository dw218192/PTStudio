#ifndef COMMON_INC
#define COMMON_INC

// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GLSL_EXT_ray_tracing.txt

// a payload can be in one of the following states:
// done:    done = true
// valid but not done: invalid = false && done = false
// invalid: invalid = true or any other state not mentioned above

struct Payload {
    // output from closest hit shader
    vec3 Li;  // sampled radiance
    vec3 pos; // intersection pos, world space
    vec3 wi;  // incoming dir, world space
    bool done, invalid; // state of the payload

    // input to chit shader, because push_constant is not usable in chit
    int iteration;
    int level;
    bool direct_lighting_only;
};

// utility functions and constants
#define PI               3.14159265358979323
#define TWO_PI           6.28318530717958648
#define FOUR_PI          12.5663706143591729
#define INV_PI           0.31830988618379067
#define INV_TWO_PI       0.15915494309
#define INV_FOUR_PI      0.07957747154594767
#define PI_OVER_TWO      1.57079632679489662
#define ONE_THIRD        0.33333333333333333
#define E                2.71828182845904524
#define INFINITY         1000000.0
#define OneMinusEpsilon  0.99999994
#define RayEpsilon       0.000005
#define Epsilon          0.000001

// Hash Functions for GPU Rendering, Jarzynski et al.
// http://www.jcgt.org/published/0009/03/02/
vec3 random_pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v ^= v >> 16u;
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return normalize(vec3(v) * (1.0/float(0xffffffffu)));
}

mat3 normalToWorldSpace(in vec3 normal) {
   vec3 someVec = vec3(1.0, 0.0, 0.0);
   float dd = dot(someVec, normal);
   vec3 tangent = vec3(0.0, 1.0, 0.0);
   if(1.0 - abs(dd) > 1e-6) {
     tangent = normalize(cross(someVec, normal));
   }
   vec3 bitangent = cross(normal, tangent);
   return mat3(tangent, bitangent, normal);
}

mat3 worldToNormalSpace(in vec3 normal) {
    return transpose(normalToWorldSpace(normal));
}

vec3 Faceforward(vec3 n, vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}
bool SameHemisphere(vec3 w, vec3 wp) {
    return w.z * wp.z > 0;
}
vec3 squareToHemisphereCosine(vec2 xi) {
    float phi = TWO_PI * xi.x;
    float cosTheta = sqrt(xi.y);
    float sinTheta = sqrt(1.0 - xi.y);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
float squareToHemisphereCosinePDF(vec3 sp) {
    return sp.z * INV_PI;
}

vec2 baryLerp(vec2 a, vec2 b, vec2 c, vec3 bary) {
    return vec2(
        bary.x * a.x + bary.y * b.x + bary.z * c.x,
        bary.x * a.y + bary.y * b.y + bary.z * c.y
    );
}

vec3 baryLerp(vec3 a, vec3 b, vec3 c, vec3 bary) {
    return vec3(
        bary.x * a.x + bary.y * b.x + bary.z * c.x,
        bary.x * a.y + bary.y * b.y + bary.z * c.y,
        bary.x * a.z + bary.y * b.z + bary.z * c.z
    );
}

#endif