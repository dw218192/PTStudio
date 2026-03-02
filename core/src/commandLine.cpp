#include <core/commandLine.h>

#include <cxxopts.hpp>
#include <iostream>
#include <stdexcept>

namespace pts {

struct CommandLine::Impl {
    cxxopts::Options options{"app", ""};
    std::optional<cxxopts::ParseResult> result;

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
    m_impl->options.add_options()(
        n, d, cxxopts::value<bool>()->default_value("false")->implicit_value("true"));
}

void CommandLine::add_string(std::string_view name, std::string_view description,
                             std::optional<std::string> default_value) {
    std::string n(name);
    std::string d(description);
    if (default_value) {
        m_impl->options.add_options()(n, d,
                                      cxxopts::value<std::string>()->default_value(*default_value));
    } else {
        m_impl->options.add_options()(n, d, cxxopts::value<std::string>());
    }
}

void CommandLine::add_int(std::string_view name, std::string_view description,
                          std::optional<int> default_value) {
    std::string n(name);
    std::string d(description);
    if (default_value) {
        m_impl->options.add_options()(
            n, d, cxxopts::value<int>()->default_value(std::to_string(*default_value)));
    } else {
        m_impl->options.add_options()(n, d, cxxopts::value<int>());
    }
}

auto CommandLine::parse(int argc, char* argv[]) -> bool {
    try {
        auto result = m_impl->options.parse(argc, argv);

        auto unknown = result.unmatched();
        if (!unknown.empty()) {
            std::cerr << "Ignoring unknown arguments:";
            for (const auto& arg : unknown) {
                std::cerr << " " << arg;
            }
            std::cerr << std::endl;
        }

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
    if (!m_impl->result) return false;
    try {
        return (*m_impl->result)[std::string(name)].as<bool>();
    } catch (...) {
        return false;
    }
}

auto CommandLine::get_string(std::string_view name, std::string_view default_value) const
    -> std::string {
    if (!m_impl->result) return std::string(default_value);
    try {
        return (*m_impl->result)[std::string(name)].as<std::string>();
    } catch (...) {
        return std::string(default_value);
    }
}

auto CommandLine::get_int(std::string_view name, int default_value) const -> int {
    if (!m_impl->result) return default_value;
    try {
        return (*m_impl->result)[std::string(name)].as<int>();
    } catch (...) {
        return default_value;
    }
}

auto CommandLine::has(std::string_view name) const -> bool {
    if (!m_impl->result) return false;
    try {
        return m_impl->result->count(std::string(name)) > 0;
    } catch (...) {
        return false;
    }
}

}  // namespace pts
