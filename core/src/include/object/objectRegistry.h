#pragma once
#include "singleton.h"
#include "utils.h"
#include "memory/handle.h"
#include "object/object.h"
#include "objectID.h"
#include <unordered_map>
#include <tl/expected.hpp>

namespace PTS {
struct Archive;

/**
 * @brief Registry for object types and their respective constructors.
 * This allows for serialization and deserialization of polymorphic types.
*/
struct ObjectRegistry : LazySingleton<ObjectRegistry> {
    NO_COPY_MOVE(ObjectRegistry);

    using CreatorFn = tl::expected<Handle<Object>, std::string>(*)(Archive&);
    
    auto register_type(std::string_view name, CreatorFn fn) noexcept -> void {
        if (m_creators.count(name)) {
            // this is not meant to be caught, it indicates a programming error and will be thrown during static initialization
            throw std::runtime_error("Type already registered");
        }
        m_creators[name] = fn;
    }

    [[nodiscard]] auto get_creator(std::string_view name) const noexcept -> tl::expected<CreatorFn, std::string> {
        auto it = m_creators.find(name);
        if (it == m_creators.end()) {
            return TL_ERROR("Type {} not found in registry\n This is likely due to a missing registration macro", name);
        }
        return it->second;
    }

private:
    std::unordered_map<std::string_view, CreatorFn> m_creators;
};


/**
 * @brief Helper class to automatically register object types and their respective constructors.
*/
struct Registrar {
    NO_COPY_MOVE(Registrar);

    Registrar(std::string_view name, ObjectRegistry::CreatorFn fn) noexcept {
        ObjectRegistry::get().register_type(name, fn);
    }
};

#define DECL_CREATOR(T) \
    static auto create(Archive& archive) -> tl::expected<Handle<Object>, std::string>;\
    static inline Registrar _register_##T { #T, T::create };\
    static auto create(Archive& archive) -> tl::expected<Handle<Object>, std::string>


} // namespace PTS