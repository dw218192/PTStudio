#include <core/commandLine.h>

#include <cxxopts.hpp>
#include <iostream>
#include <set>
#include <stdexcept>

namespace pts {

struct CommandLineImpl {
    cxxopts::Options options{"app", ""};
    std::optional<cxxopts::ParseResult> result;
    std::set<std::string> registered;   // all registered option names
    std::set<std::string> has_default;  // options registered with a default value

    CommandLineImpl() {
        options.allow_unrecognised_options();
    }
};

static_assert(sizeof(CommandLineImpl) <= 1024,
              "CommandLineImpl grew past in-place PIMPL buffer -- bump Size in commandLine.h");
static_assert(alignof(CommandLineImpl) <= alignof(std::max_align_t),
              "CommandLineImpl alignment exceeds in-place PIMPL alignment -- bump Align");

CommandLine::CommandLine() {
    construct();
    impl().options.add_options()("h,help", "produce help message");
}

CommandLine::~CommandLine() {
    destroy();
}

void CommandLine::add_flag(std::string_view name, std::string_view description) {
    std::string n(name);
    std::string d(description);
    impl().registered.insert(n);
    impl().has_default.insert(n);
    impl().options.add_options()(
        n, d, cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
}

void CommandLine::add_string(std::string_view name, std::string_view description,
                             std::optional<std::string> default_value,
                             std::optional<std::string> implicit_value) {
    std::string n(name);
    std::string d(description);
    impl().registered.insert(n);
    auto val = cxxopts::value<std::string>();
    if (default_value) {
        impl().has_default.insert(n);
        val->default_value(*default_value);
    }
    if (implicit_value) {
        val->implicit_value(*implicit_value);
    }
    impl().options.add_options()(n, d, std::move(val));
}

void CommandLine::add_int(std::string_view name, std::string_view description,
                          std::optional<int> default_value) {
    std::string n(name);
    std::string d(description);
    impl().registered.insert(n);
    if (default_value) {
        impl().has_default.insert(n);
        impl().options.add_options()(
            n, d, cxxopts::value<int>()->default_value(std::to_string(*default_value)));
    } else {
        impl().options.add_options()(n, d, cxxopts::value<int>());
    }
}

auto CommandLine::parse(int argc, char* argv[]) -> bool {
    try {
        auto result = impl().options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << impl().options.help() << std::endl;
            impl().result.emplace(std::move(result));
            return false;
        }

        impl().result.emplace(std::move(result));
        return true;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error parsing command line arguments: ") + e.what());
    }
}

auto CommandLine::get_flag(std::string_view name) const -> bool {
    std::string key(name);
    if (!impl().result || !impl().registered.count(key)) return false;
    // Flags always have a registered default ("false"), so as<bool>() never throws.
    return (*impl().result)[key].as<bool>();
}

auto CommandLine::get_string(std::string_view name, std::string_view default_value) const
    -> std::string {
    std::string key(name);
    if (!impl().result || !impl().registered.count(key)) return std::string(default_value);
    // Explicitly provided OR has a registered default -> as<T>() returns the value.
    // Not provided AND no registered default -> return call-site default.
    if (impl().result->count(key) == 0 && !impl().has_default.count(key)) {
        return std::string(default_value);
    }
    return (*impl().result)[key].as<std::string>();
}

auto CommandLine::get_int(std::string_view name, int default_value) const -> int {
    std::string key(name);
    if (!impl().result || !impl().registered.count(key)) return default_value;
    if (impl().result->count(key) == 0 && !impl().has_default.count(key)) {
        return default_value;
    }
    return (*impl().result)[key].as<int>();
}

auto CommandLine::has(std::string_view name) const -> bool {
    std::string key(name);
    if (!impl().result || !impl().registered.count(key)) return false;
    return impl().result->count(key) > 0;
}

}  // namespace pts
