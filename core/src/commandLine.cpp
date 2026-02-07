#include <core/commandLine.h>

#if PTS_ENABLE_PROGRAM_OPTIONS

#include <boost/program_options.hpp>
#include <iostream>
#include <stdexcept>

namespace po = boost::program_options;

namespace pts {

struct CommandLine::Impl {
    po::options_description desc{"Options"};
    po::variables_map vm;
};

CommandLine::CommandLine() : m_impl(std::make_unique<Impl>()) {
    m_impl->desc.add_options()("help,h", "produce help message");
}

CommandLine::~CommandLine() = default;

void CommandLine::add_flag(std::string_view name, std::string_view description) {
    std::string n(name);
    std::string d(description);
    m_impl->desc.add_options()(n.c_str(), po::bool_switch(), d.c_str());
}

void CommandLine::add_string(std::string_view name, std::string_view description,
                             std::optional<std::string> default_value) {
    std::string n(name);
    std::string d(description);
    if (default_value) {
        m_impl->desc.add_options()(
            n.c_str(), po::value<std::string>()->default_value(*default_value), d.c_str());
    } else {
        m_impl->desc.add_options()(n.c_str(), po::value<std::string>(), d.c_str());
    }
}

auto CommandLine::parse(int argc, char* argv[]) -> bool {
    try {
        auto parsed = po::command_line_parser(argc, argv)
                          .options(m_impl->desc)
                          .style(po::command_line_style::default_style &
                                 ~po::command_line_style::allow_guessing)
                          .allow_unregistered()
                          .run();

        po::store(parsed, m_impl->vm);
        po::notify(m_impl->vm);

        auto unknown = po::collect_unrecognized(parsed.options, po::include_positional);
        if (!unknown.empty()) {
            std::cerr << "Ignoring unknown arguments:";
            for (const auto& arg : unknown) {
                std::cerr << " " << arg;
            }
            std::cerr << std::endl;
        }

        if (m_impl->vm.count("help")) {
            std::cout << m_impl->desc << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        throw std::runtime_error("Unknown error parsing command line arguments");
    }
}

auto CommandLine::get_flag(std::string_view name) const -> bool {
    std::string key(name);
    if (m_impl->vm.count(key)) {
        return m_impl->vm[key].as<bool>();
    }
    return false;
}

auto CommandLine::get_string(std::string_view name,
                             std::string_view default_value) const -> std::string {
    std::string key(name);
    if (m_impl->vm.count(key)) {
        return m_impl->vm[key].as<std::string>();
    }
    return std::string(default_value);
}

auto CommandLine::has(std::string_view name) const -> bool {
    return m_impl->vm.count(std::string(name)) > 0;
}

}  // namespace pts

#else  // Emscripten stubs

namespace pts {

CommandLine::CommandLine() = default;
CommandLine::~CommandLine() = default;

void CommandLine::add_flag(std::string_view /*name*/, std::string_view /*description*/) {
}

void CommandLine::add_string(std::string_view /*name*/, std::string_view /*description*/,
                             std::optional<std::string> /*default_value*/) {
}

auto CommandLine::parse(int /*argc*/, char* /*argv*/[]) -> bool {
    return true;
}

auto CommandLine::get_flag(std::string_view /*name*/) const -> bool {
    return false;
}

auto CommandLine::get_string(std::string_view /*name*/,
                             std::string_view default_value) const -> std::string {
    return std::string(default_value);
}

auto CommandLine::has(std::string_view /*name*/) const -> bool {
    return false;
}

}  // namespace pts

#endif
