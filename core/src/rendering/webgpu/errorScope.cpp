#include <core/loggingManager.h>
#include <core/rendering/webgpu/device.h>
#include <core/rendering/webgpu/errorScope.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"

namespace pts::webgpu {

ErrorScope::ErrorScope(const Device& device, WGPUErrorFilter filter, std::string_view logger_name,
                       std::string_view context_label)
    : ErrorScope(device, {filter}, logger_name, context_label) {
}

ErrorScope::ErrorScope(const Device& device, std::initializer_list<WGPUErrorFilter> filters,
                       std::string_view logger_name, std::string_view context_label)
    : m_device(device), m_logger_name(logger_name), m_context_label(context_label) {
    m_scope_count = filters.size();
    m_results.reserve(m_scope_count);

    // Push all error scopes in order
    for (const auto filter : filters) {
        wgpuDevicePushErrorScope(device.handle(), filter);
    }
}

ErrorScope::~ErrorScope() {
    if (!m_popped) {
        pop_and_wait();
        log_all_errors();
    }
}

void ErrorScope::pop_and_throw_if_error() {
    pop_and_wait();

    // Check if any scope captured an error
    bool has_error = false;
    for (const auto& result : m_results) {
        if (result.status != WGPUPopErrorScopeStatus_Success ||
            result.type != WGPUErrorType_NoError) {
            has_error = true;
            break;
        }
    }

    if (has_error) {
        log_all_errors();
        const auto context_label =
            m_context_label.empty() ? std::string_view("WebGPU operation") : m_context_label;
        throw std::runtime_error(std::string("WebGPU ") + std::string(context_label) + " failed");
    }
}

namespace {
struct PopCallbackData {
    ErrorScope::Result* result;
    std::atomic_size_t* pending;
};
}  // namespace

void ErrorScope::pop_and_wait() {
    if (m_popped) {
        return;
    }

    // Pop all scopes in reverse order (LIFO)
    m_results.clear();
    m_results.resize(m_scope_count);

    std::atomic_size_t pending{m_scope_count};
    std::vector<PopCallbackData> callback_data;
    callback_data.reserve(m_scope_count);

    for (std::size_t i = 0; i < m_scope_count; ++i) {
        callback_data.push_back({&m_results[i], &pending});
    }

    for (std::size_t i = 0; i < m_scope_count; ++i) {
        WGPUPopErrorScopeCallbackInfo callback = {};
        callback.mode = WGPUCallbackMode_AllowProcessEvents;
        callback.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type,
                               WGPUStringView message, void* userdata1, void*) {
            auto* data = static_cast<PopCallbackData*>(userdata1);
            data->result->status = status;
            data->result->type = type;
            if (message.data && message.length > 0) {
                data->result->message.assign(message.data, message.length);
            } else {
                data->result->message.clear();
            }
            data->pending->fetch_sub(1, std::memory_order_release);
        };
        callback.userdata1 = &callback_data[i];

        wgpuDevicePopErrorScope(m_device.handle(), callback);
    }

    // Process events until all callbacks complete
    while (pending.load(std::memory_order_acquire) != 0U) {
        wgpuInstanceProcessEvents(m_device.instance());
        std::this_thread::yield();
    }

    m_popped = true;
}

void ErrorScope::log_all_errors() const {
    for (std::size_t i = 0; i < m_results.size(); ++i) {
        const auto& result = m_results[i];
        if (result.status != WGPUPopErrorScopeStatus_Success ||
            result.type != WGPUErrorType_NoError) {
            log_error(result, i);
        }
    }
}

void ErrorScope::log_error(const Result& result, std::size_t scope_index) const {
    const auto& logger_name = m_logger_name.empty() ? k_webgpu_logger_name : m_logger_name;
    const auto context_label =
        m_context_label.empty() ? std::string_view("WebGPU operation") : m_context_label;

    const auto scope_info = m_scope_count > 1
                                ? std::string(" [scope ") + std::to_string(scope_index) + "]"
                                : std::string("");

    if (result.status != WGPUPopErrorScopeStatus_Success) {
        pts::log_or_cerr(logger_name, pts::LogLevel::Error,
                         "{}{} error scope pop failed (status: {})", context_label, scope_info,
                         status_name(result.status));
        return;
    }

    if (result.type != WGPUErrorType_NoError) {
        const std::string_view message =
            result.message.empty() ? std::string_view("(no message)") : result.message;
        pts::log_or_cerr(logger_name, pts::LogLevel::Error, "Failed to create {}{} ({}): {}",
                         context_label, scope_info, error_type_name(result.type), message);
    }
}

}  // namespace pts::webgpu
