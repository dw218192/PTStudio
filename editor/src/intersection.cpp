#include "intersection.h"
#include <cmath>

#define SMALL_FLOAT 1e-6f
#define LARGE_FLOAT 1e6f
#define EPSILON 1e-6f

// we try to avoid any functions in std namespace
// because we need this code to be runnable on both host and device
auto Intersection::ray_box(BoundingBox const& box, Ray const& r) -> Result {
    float tmin = SMALL_FLOAT;
    float tmax = LARGE_FLOAT;
    glm::vec3 tmins = (box.min_pos - r.origin) / r.direction;
    glm::vec3 tmaxs = (box.max_pos - r.origin) / r.direction;

    for (int i = 0; i < 3; ++i) {
        if (fabsf(r.direction[i]) < EPSILON) {
            if (r.origin[i] < box.min_pos[i] || r.origin[i] > box.max_pos[i]) {
                return false;
            }
        } else {
            auto t0 = tmins[i];
            auto t1 = tmaxs[i];
            if (t0 > t1) {
                auto tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            if (tmin > t1 || tmax < t0) {
                return false;
            }
            tmin = fmax(t0, tmin);
            tmax = fmin(t1, tmax);
            if (tmin > tmax) {
                return false;
            }
        }
    }

    if (tmin < 0 && tmax < 0) {
        return false;
    }
    
    return Result { 
        true,
        tmin < 0 ? tmax : tmin 
    };
}