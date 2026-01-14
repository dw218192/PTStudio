#pragma once
#include <cmrc/cmrc.hpp>
#include <tl/expected.hpp>

#include "utils.h"

namespace PTS {
[[nodiscard]] auto inline try_get_embedded_res(cmrc::embedded_filesystem const& embedded_fs,
                                               std::string const& path)
    -> tl::expected<std::string, std::string> {
    if (!embedded_fs.exists(path)) {
        return TL_ERROR("built-in resource {} does not exist", path);
    }
    auto const raw = embedded_fs.open(path);
    return std::string{raw.begin(), raw.end()};
}
}  // namespace PTS
