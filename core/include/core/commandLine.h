#pragma once

#include <core/defines.h>
#include <core/inPlacePimpl.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace pts {

struct CommandLineImpl;

/// CLI argument parser (wraps cxxopts).
class CommandLine final
    : private InPlacePimpl<CommandLine, CommandLineImpl, 1024, alignof(std::max_align_t)> {
   public:
    NO_COPY_MOVE(CommandLine);

    CommandLine();
    ~CommandLine();

    /// Add a boolean flag (defaults to false).
    void add_flag(std::string_view name, std::string_view description);

    /// Add a string option with an optional default value.
    /// implicit_value: value used when the flag is present but no argument follows it.
    void add_string(std::string_view name, std::string_view description,
                    std::optional<std::string> default_value = std::nullopt,
                    std::optional<std::string> implicit_value = std::nullopt);

    /// Add an integer option with an optional default value.
    void add_int(std::string_view name, std::string_view description,
                 std::optional<int> default_value = std::nullopt);

    /// Parse argc/argv. Returns false if --help was requested (caller should exit).
    /// Unrecognized arguments are logged to stderr but do not cause failure.
    [[nodiscard]] auto parse(int argc, char* argv[]) -> bool;

    /// Query whether a flag was set.
    [[nodiscard]] auto get_flag(std::string_view name) const -> bool;

    /// Query a string option. Returns default_value if not present.
    [[nodiscard]] auto get_string(std::string_view name, std::string_view default_value = {}) const
        -> std::string;

    /// Query an integer option. Returns default_value if not present.
    [[nodiscard]] auto get_int(std::string_view name, int default_value = 0) const -> int;

    /// Query whether an option was explicitly provided on the command line.
    [[nodiscard]] auto has(std::string_view name) const -> bool;
};

}  // namespace pts
