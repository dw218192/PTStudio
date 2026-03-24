#include <core/commandLine.h>

#include <cxxopts.hpp>
#include <iostream>
#include <set>
#include <stdexcept>

namespace pts {

struct CommandLine::Impl {
    cxxopts::Options options{"app", ""};
    std::optional<cxxopts::ParseResult> result;
    std::set<std::string> registered;   // all registered option names
    std::set<std::string> has_default;  // options registered with a default value

    Impl() {
        options.allow_unrecognised_options();
    }
};

CommandLine::CommandLine() : m_impl(std::make_unique<Impl>()) {
    m_impl->options.add_options()("h,help", "produce help message");
}

CommandLine::~CommandLine() = default;

void CommandLine::add_flag(std::string_view name, std::string_view description) {
    std::string n(name);
    std::string d(description);
    m_impl->registered.insert(n);
    m_impl->has_default.insert(n);
    m_impl->options.add_options()(
        n, d, cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
}

void CommandLine::add_string(std::string_view name, std::string_view description,
                             std::optional<std::string> default_value,
                             std::optional<std::string> implicit_value) {
    std::string n(name);
    std::string d(description);
    m_impl->registered.insert(n);
    auto val = cxxopts::value<std::string>();
    if (default_value) {
        m_impl->has_default.insert(n);
        val->default_value(*default_value);
    }
    if (implicit_value) {
        val->implicit_value(*implicit_value);
    }
    m_impl->options.add_options()(n, d, std::move(val));
}

void CommandLine::add_int(std::string_view name, std::string_view description,
                          std::optional<int> default_value) {
    std::string n(name);
    std::string d(description);
    m_impl->registered.insert(n);
    if (default_value) {
        m_impl->has_default.insert(n);
        m_impl->options.add_options()(
            n, d, cxxopts::value<int>()->default_value(std::to_string(*default_value)));
    } else {
        m_impl->options.add_options()(n, d, cxxopts::value<int>());
    }
}

auto CommandLine::parse(int argc, char* argv[]) -> bool {
    try {
        auto result = m_impl->options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << m_impl->options.help() << std::endl;
            m_impl->result.emplace(std::move(result));
            return false;
        }

        m_impl->result.emplace(std::move(result));
        return true;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error parsing command line arguments: ") + e.what());
    }
}

auto CommandLine::get_flag(std::string_view name) const -> bool {
    std::string key(name);
    if (!m_impl->result || !m_impl->registered.count(key)) return false;
    // Flags always have a registered default ("false"), so as<bool>() never throws.
    return (*m_impl->result)[key].as<bool>();
}

auto CommandLine::get_string(std::string_view name, std::string_view default_value) const
    -> std::string {
    std::string key(name);
    if (!m_impl->result || !m_impl->registered.count(key)) return std::string(default_value);
    // Explicitly provided OR has a registered default → as<T>() returns the value.
    // Not provided AND no registered default → return call-site default.
    if (m_impl->result->count(key) == 0 && !m_impl->has_default.count(key)) {
        return std::string(default_value);
    }
    return (*m_impl->result)[key].as<std::string>();
}

auto CommandLine::get_int(std::string_view name, int default_value) const -> int {
    std::string key(name);
    if (!m_impl->result || !m_impl->registered.count(key)) return default_value;
    if (m_impl->result->count(key) == 0 && !m_impl->has_default.count(key)) {
        return default_value;
    }
    return (*m_impl->result)[key].as<int>();
}

auto CommandLine::has(std::string_view name) const -> bool {
    std::string key(name);
    if (!m_impl->result || !m_impl->registered.count(key)) return false;
    return m_impl->result->count(key) > 0;
}

}  // namespace pts
