#ifndef __EMSCRIPTEN__

#include <core/rendering/shaderc/slangMetadata.h>
#include <slang.h>

#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pts::rendering {
namespace {
using Layout = slang::TypeLayoutReflection;
using Kind = slang::TypeReflection::Kind;
using Scalar = slang::TypeReflection::ScalarType;

bool identifier(std::string_view name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_'))
        return false;
    for (unsigned char ch : name) {
        if (!std::isalnum(ch) && ch != '_') return false;
    }
    return true;
}

struct CppType {
    std::string name;
    size_t size = 0;

    bool operator==(const CppType& other) const {
        return std::tie(name, size) == std::tie(other.name, other.size);
    }
};

struct CppField {
    std::string name;
    CppType type;
    size_t offset;

    bool operator==(const CppField& other) const {
        return std::tie(name, type, offset) == std::tie(other.name, other.type, other.offset);
    }
};

// One description drives declarations, padding, ABI checks and conflict detection.
struct CppStruct {
    CppType type;
    size_t alignment;
    std::vector<CppField> fields;

    bool operator==(const CppStruct& other) const {
        return std::tie(type, alignment, fields) ==
               std::tie(other.type, other.alignment, other.fields);
    }

    void write(std::ostream& out) const {
        out << "struct alignas(" << alignment << ") " << type.name << " {\n";
        size_t cursor = 0;
        unsigned padding = 0;
        auto pad_to = [&](size_t offset) {
            if (offset > cursor)
                out << "    std::byte _pts_padding_" << padding++ << "[" << offset - cursor
                    << "]{};\n";
        };
        for (const auto& field : fields) {
            pad_to(field.offset);
            out << "    " << field.type.name << " " << field.name << "{};\n";
            cursor = field.offset + field.type.size;
        }
        pad_to(type.size);
        out << "};\nstatic_assert(sizeof(" << type.name << ") == " << type.size << ");\n"
            << "static_assert(alignof(" << type.name << ") == " << alignment << ");\n"
            << "static_assert(std::is_standard_layout_v<" << type.name
            << "> && std::is_trivially_copyable_v<" << type.name << ">);\n";
        for (const auto& field : fields) {
            out << "static_assert(offsetof(" << type.name << ", " << field.name
                << ") == " << field.offset << ");\n"
                << "static_assert(sizeof(" << type.name << "::" << field.name
                << ") == " << field.type.size << ");\n";
        }
        out << '\n';
    }
};

class CppStructBuilder {
   public:
    CppStructBuilder(std::string name, size_t size, size_t alignment)
        : m_struct{{std::move(name), size}, alignment, {}} {
    }

    CppStructBuilder& field(std::string name, CppType type, size_t offset) {
        m_struct.fields.push_back({std::move(name), std::move(type), offset});
        return *this;
    }

    std::optional<CppStruct> build(std::string& error) {
        const auto& name = m_struct.type.name;
        if (!identifier(name)) {
            error = "upload struct requires a simple identifier: " + name;
            return {};
        }
        const auto alignment = m_struct.alignment;
        const auto size = m_struct.type.size;
        if (!alignment || (alignment & (alignment - 1)) || !size) {
            error = "unsupported struct alignment/stride: " + name;
            return {};
        }
        size_t cursor = 0;
        for (const auto& field : m_struct.fields) {
            if (!identifier(field.name) || field.name.find("_pts_padding_") == 0) {
                error = "unsupported upload field name: " + field.name;
                return {};
            }
            if (field.offset < cursor || field.offset > size ||
                field.type.size > size - field.offset) {
                error = "overlapping or oversized upload field: " + name + "." + field.name;
                return {};
            }
            cursor = field.offset + field.type.size;
        }
        return std::move(m_struct);
    }

   private:
    CppStruct m_struct;
};

struct ReflectedTypes {
    std::map<std::string, std::vector<Layout*>> layouts;
    std::set<Layout*> visited[2];

    void visit(Layout* layout, bool in_buffer = false) {
        if (!layout || !visited[in_buffer].insert(layout).second) return;
        const auto kind = layout->getKind();
        if (kind == Kind::Struct) {
            if (in_buffer && layout->getName()) layouts[layout->getName()].push_back(layout);
            for (unsigned i = 0; i < layout->getFieldCount(); ++i)
                visit(layout->getFieldByIndex(i)->getTypeLayout(), in_buffer);
        } else if (kind == Kind::Array) {
            visit(layout->getElementTypeLayout(), in_buffer);
        } else if (kind == Kind::ParameterBlock) {
            visit(layout->getElementTypeLayout());
        } else if (kind == Kind::ConstantBuffer || kind == Kind::ShaderStorageBuffer ||
                   (kind == Kind::Resource &&
                    (layout->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK) ==
                        SLANG_STRUCTURED_BUFFER)) {
            visit(layout->getElementTypeLayout(), true);
        }
    }
};

struct Emitter {
    std::map<std::string, CppStruct> definitions;
    std::vector<std::string> definition_order;
    std::string error;

    CppType fail(std::string message) {
        if (error.empty()) error = std::move(message);
        return {};
    }

    CppType type(Layout* layout) {
        if (!layout) return fail("missing element layout");
        const auto kind = layout->getKind();
        const auto size = layout->getSize();
        if (kind == Kind::Scalar || kind == Kind::Vector || kind == Kind::Matrix) {
            std::string scalar;
            switch (layout->getScalarType()) {
                case Scalar::Float32:
                    scalar = "float";
                    break;
                case Scalar::Int32:
                    scalar = "std::int32_t";
                    break;
                case Scalar::UInt32:
                    scalar = "std::uint32_t";
                    break;
                default:
                    return fail("only float32, int32 and uint32 upload scalars are supported");
            }
            if (kind == Kind::Scalar) {
                if (size != 4) return fail("unexpected scalar size");
                return {scalar, size};
            }
            const auto rows = layout->getRowCount();
            const auto columns = layout->getColumnCount();
            if (kind == Kind::Vector) {
                const auto count = layout->getElementCount();
                if (count < 2 || count > 4 || size != count * 4)
                    return fail("unsupported vector layout");
                return {
                    "glm::vec<" + std::to_string(count) + ", " + scalar + ", glm::packed_highp>",
                    size};
            }
            // GLM stores packed columns. Reject layouts requiring a conversion
            // instead of assuming that the CPU-target matrix ABI matches WGSL.
            if (layout->getMatrixLayoutMode() != SLANG_MATRIX_LAYOUT_COLUMN_MAJOR || rows < 2 ||
                rows > 4 || columns < 2 || columns > 4 || scalar != "float" ||
                size != rows * columns * 4)
                return fail(
                    "matrix requires a packed column-major float layout; use padded columns "
                    "explicitly");
            return {"glm::mat<" + std::to_string(columns) + ", " + std::to_string(rows) +
                        ", float, glm::packed_highp>",
                    size};
        }
        if (kind == Kind::Array) {
            auto element = type(layout->getElementTypeLayout());
            const auto count = layout->getElementCount();
            if (element.name.empty()) return {};
            if (count == 0 || count == SLANG_UNBOUNDED_SIZE ||
                layout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM) != element.size ||
                size != count * element.size)
                return fail("array requires fixed elements with matching C++ stride");
            return {"std::array<" + element.name + ", " + std::to_string(count) + ">", size};
        }
        if (kind != Kind::Struct) return fail("resource or unsupported type inside upload data");
        const std::string name = layout->getName() ? layout->getName() : "";
        const auto alignment = layout->getAlignment();
        const auto stride = layout->getStride();
        if (alignment < 0 || stride == SLANG_UNBOUNDED_SIZE)
            return fail("unsupported struct alignment/stride: " + name);
        CppStructBuilder builder(name, stride, alignment);
        for (unsigned i = 0; i < layout->getFieldCount(); ++i) {
            auto* field = layout->getFieldByIndex(i);
            auto field_type = type(field->getTypeLayout());
            if (field_type.name.empty()) return {};
            builder.field(field->getName() ? field->getName() : "", std::move(field_type),
                          field->getOffset());
        }
        auto definition = builder.build(error);
        if (!definition) return {};
        const auto existing = definitions.find(name);
        if (existing != definitions.end()) {
            if (!(existing->second == *definition))
                return fail("conflicting buffer layouts for " + name +
                            "; use distinct Slang type names for incompatible layouts");
            return existing->second.type;
        }
        auto [entry, inserted] = definitions.emplace(name, std::move(*definition));
        definition_order.push_back(name);
        return entry->second.type;
    }
};
}  // namespace

bool run_slang_types_header(slang::ShaderReflection* reflection,
                            const std::vector<std::string>& names, std::string_view ns,
                            std::string& header, std::string& diagnostics) {
    header.clear();
    if (!reflection || ns.empty() || names.empty()) {
        diagnostics += "types require reflection, names and a namespace\n";
        return false;
    }
    // Validate namespace segments before emitting them as C++ source.
    for (size_t start = 0; start <= ns.size();) {
        auto end = ns.find("::", start);
        if (end == std::string_view::npos) end = ns.size();
        if (!identifier(ns.substr(start, end - start))) {
            diagnostics += "invalid types namespace\n";
            return false;
        }
        if (end == ns.size()) break;
        start = end + 2;
    }
    ReflectedTypes reflected;
    for (unsigned i = 0; i < reflection->getParameterCount(); ++i)
        reflected.visit(reflection->getParameterByIndex(i)->getTypeLayout());

    Emitter emitter;
    std::vector<std::string> pending = names;
    std::set<std::string> checked;
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto name = pending[i];
        if (!checked.insert(name).second) continue;
        const auto found = reflected.layouts.find(name);
        if (found == reflected.layouts.end()) {
            diagnostics += "type not found in reflected WGSL buffers: " + name + "\n";
            return false;
        }
        for (auto* layout : found->second) {
            if (emitter.type(layout).name.empty()) {
                diagnostics += "type " + name + ": " + emitter.error + "\n";
                return false;
            }
        }
        // Dependencies must also agree across every reflected use, including
        // uses in buffers whose outer type was not requested.
        for (const auto& definition : emitter.definitions)
            if (!checked.count(definition.first)) pending.push_back(definition.first);
    }
    std::ostringstream output;
    output
        << "#pragma once\n// Generated from WGSL buffer reflection by pts_shaderc. DO NOT EDIT.\n"
           "#include <array>\n#include <cstddef>\n#include <cstdint>\n#include <type_traits>\n"
           "#include <glm/glm.hpp>\n\nnamespace "
        << ns << " {\n";
    for (const auto& name : emitter.definition_order) emitter.definitions.at(name).write(output);
    output << "}  // namespace " << ns << "\n";
    header = output.str();
    return true;
}
}  // namespace pts::rendering
#endif
