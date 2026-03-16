#pragma once

#include <core/rendering/propertyDescriptor.h>
#include <pxr/usd/usd/prim.h>

namespace pts::editor {

/// Draw a single property widget. Returns true if the value was modified.
bool draw_property(rendering::PropertyDescriptor& prop);

/// Write a modified property value back to the USD stage.
void write_property(const pxr::UsdPrim& prim, const rendering::PropertyDescriptor& prop);

/// Draw the full property panel for a prim. Returns true if any value was modified.
bool draw_prim_properties(const pxr::UsdPrim& prim);

}  // namespace pts::editor
