#pragma once

#include <string_view>
#include <string>
#include <utils.h>
#include <tl/expected.hpp>

#include "scene.h"
#include "camera.h"
#include "jsonArchive.h"


namespace PTS {
    enum class ArchiveFormat {
        JSON,
    };

    /**
     * \brief Interface for any serializer that can be used to save/load a scene.
    */
    struct Archive {
        DEFAULT_COPY_MOVE(Archive);
        ~Archive() = default;

        Archive(std::string_view path, ArchiveFormat format) 
            : m_format(format), m_path(path)
        {
            switch (m_format) {
            case ArchiveFormat::JSON:
                m_json_archive = std::make_unique<JsonArchive>();
                break;
            default:
                throw std::runtime_error("Unknown archive format");
            }
        }
        
        auto get_path() const -> std::string_view {
            return m_path;
        }

        auto get_format() const -> ArchiveFormat {
            return m_format;
        }
        
        auto get_ext() const -> std::string_view {
            switch (m_format) {
            case ArchiveFormat::JSON:
                return m_json_archive->get_ext();
            default:
                throw std::runtime_error("Unknown archive format");
            }
        }

        template<typename T>
        auto get(std::string_view key) -> tl::expected<T, std::string> {
            switch (m_format) {
            case ArchiveFormat::JSON:
                return m_json_archive->get<T>(key);
            default:
                return tl::unexpected("Unknown archive format");
            }
        }

        template<typename T>
        auto set(std::string_view key, T const& value) -> tl::expected<void, std::string> {
            switch (m_format) {
            case ArchiveFormat::JSON:
                return m_json_archive->set(key, value);
            default:
                return tl::unexpected("Unknown archive format");
            }
        }

    private:
        ArchiveFormat m_format;
        std::string m_path;
        std::unique_ptr<JsonArchive> m_json_archive;
    };
}
