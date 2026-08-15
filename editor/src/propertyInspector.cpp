#include "propertyInspector.h"

#include <core/diagnostics.h>
#include <core/rendering/adapters/registry.h>
#include <imgui.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/usd/attribute.h>

#include <any>
#include <string>

namespace pts::editor {

bool draw_property(rendering::PropertyDescriptor& prop) {
    bool read_only = rendering::has_tag(prop.tags, rendering::PropertyTag::ReadOnly);
    if (read_only) ImGui::BeginDisabled();

    bool changed = false;
    auto const& ti = prop.value.type();

    if (ti == typeid(float)) {
        auto val = std::any_cast<float>(prop.value);
        if (ImGui::DragFloat(prop.label.c_str(), &val, prop.drag_speed, prop.min_val,
                             prop.max_val)) {
            prop.value = val;
            changed = true;
        }
    } else if (ti == typeid(double)) {
        auto val = static_cast<float>(std::any_cast<double>(prop.value));
        if (ImGui::DragFloat(prop.label.c_str(), &val, prop.drag_speed, prop.min_val,
                             prop.max_val)) {
            prop.value = static_cast<double>(val);
            changed = true;
        }
    } else if (ti == typeid(int)) {
        auto val = std::any_cast<int>(prop.value);
        if (ImGui::DragInt(prop.label.c_str(), &val)) {
            prop.value = val;
            changed = true;
        }
    } else if (ti == typeid(bool)) {
        auto val = std::any_cast<bool>(prop.value);
        if (ImGui::Checkbox(prop.label.c_str(), &val)) {
            prop.value = val;
            changed = true;
        }
    } else if (ti == typeid(pxr::GfVec3f)) {
        auto val = std::any_cast<pxr::GfVec3f>(prop.value);
        bool is_color = rendering::has_tag(prop.tags, rendering::PropertyTag::Color);
        if (is_color) {
            if (ImGui::ColorEdit3(prop.label.c_str(), val.data())) {
                prop.value = val;
                changed = true;
            }
        } else {
            if (ImGui::DragFloat3(prop.label.c_str(), val.data(), prop.drag_speed)) {
                prop.value = val;
                changed = true;
            }
        }
    } else if (ti == typeid(std::string)) {
        auto const& val = std::any_cast<std::string const&>(prop.value);
        ImGui::LabelText(prop.label.c_str(), "%s", val.c_str());
    }

    if (read_only) ImGui::EndDisabled();

    if (changed && prop.validate) {
        if (!prop.validate(prop.value)) {
            changed = false;
        }
    }

    return changed;
}

void write_property(const pxr::UsdPrim& prim, const rendering::PropertyDescriptor& prop) {
    auto attr = prim.GetAttribute(pxr::TfToken(prop.name));
    CHECK_MSG(attr, "adapter returned property name with no matching USD attribute");

    auto const& ti = prop.value.type();
    if (ti == typeid(float)) {
        attr.Set(std::any_cast<float>(prop.value));
    } else if (ti == typeid(double)) {
        attr.Set(std::any_cast<double>(prop.value));
    } else if (ti == typeid(int)) {
        attr.Set(std::any_cast<int>(prop.value));
    } else if (ti == typeid(bool)) {
        attr.Set(std::any_cast<bool>(prop.value));
    } else if (ti == typeid(pxr::GfVec3f)) {
        attr.Set(std::any_cast<pxr::GfVec3f>(prop.value));
    }
}

bool draw_prim_properties(const pxr::UsdPrim& prim) {
    // Find adapter for this prim
    for (auto* adapter : rendering::k_scene_adapters()) {
        if (!adapter->can_adapt(prim)) continue;

        ImGui::TextUnformatted(prim.GetPath().GetText());
        ImGui::TextDisabled("%s", prim.GetTypeName().GetText());
        ImGui::Spacing();

        auto props = adapter->get_properties(prim);
        if (props.empty()) return false;

        bool any_changed = false;
        for (auto& prop : props) {
            ImGui::PushID(prop.name.c_str());
            if (draw_property(prop)) {
                write_property(prim, prop);
                any_changed = true;
            }
            ImGui::PopID();
        }
        return any_changed;
    }

    // No adapter found -- show basic info
    ImGui::TextUnformatted(prim.GetPath().GetText());
    ImGui::TextDisabled("%s", prim.GetTypeName().GetText());
    return false;
}

}  // namespace pts::editor
