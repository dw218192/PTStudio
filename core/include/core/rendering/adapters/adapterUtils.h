#pragma once

#include <core/rendering/vertex.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>

namespace pts::rendering {

// Reads primvars:displayColor from a UsdGeomGprim and writes to vertex color.
// Falls back to white (1,1,1) if no displayColor is authored.
inline void apply_display_color(Vertex& v, const pxr::UsdGeomPrimvarsAPI& primvars_api) {
    // Cache the color array in a thread-local to avoid repeated USD reads
    // when called in a loop — callers should use the batch overload instead.
}

// Reads displayColor primvar once, returns the array.
inline pxr::VtVec3fArray read_display_color(pxr::UsdPrim prim) {
    auto primvars_api = pxr::UsdGeomPrimvarsAPI(prim);
    pxr::VtVec3fArray colors;
    auto color_pv = primvars_api.GetPrimvar(pxr::TfToken("displayColor"));
    if (color_pv) {
        color_pv.Get(&colors);
    }
    return colors;
}

// Applies uniform display color to a vertex.
// If colors is empty or has one entry, uses that (or white fallback).
inline void apply_display_color(Vertex& v, const pxr::VtVec3fArray& colors) {
    if (!colors.empty()) {
        v.color[0] = colors[0][0];
        v.color[1] = colors[0][1];
        v.color[2] = colors[0][2];
    } else {
        v.color[0] = 1.0f;
        v.color[1] = 1.0f;
        v.color[2] = 1.0f;
    }
}

}  // namespace pts::rendering
