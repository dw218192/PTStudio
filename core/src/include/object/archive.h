#pragma once

#include <string_view>
#include <string>
#include <utils.h>
#include <tl/expected.hpp>
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

        Archive(ArchiveFormat format) : m_format(format) {}

        auto open(std::string_view path) -> tl::expected<void, std::string> {
            m_path = path;

            switch (m_format) {
            case ArchiveFormat::JSON:
                m_json_archive = std::make_unique<JsonArchive>();
                TL_CHECK_AND_PASS(m_json_archive->open(path));
                break;
            default:
                throw std::runtime_error("Unknown archive format");
            }


            return {};
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
        std::string m_path;
        ArchiveFormat m_format;
        std::unique_ptr<JsonArchive> m_json_archive;
    };
}
