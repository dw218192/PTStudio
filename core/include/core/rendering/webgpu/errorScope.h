#pragma once

#include <core/rendering/webgpu/webgpu.h>

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace pts::webgpu {
class Device;

/// Convert seconds to nanoseconds for WebGPU future wait timeout
inline auto to_nanoseconds(std::chrono::seconds timeout) -> std::uint64_t {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count());
}

/// Convert WGPUPopErrorScopeStatus to string representation
inline auto status_name(WGPUPopErrorScopeStatus status) -> const char* {
    switch (status) {
        case WGPUPopErrorScopeStatus_Success:
            return "Success";
        case WGPUPopErrorScopeStatus_Error:
            return "Error";
        default:
            return "Unknown";
    }
}

/// Convert WGPUErrorType to string representation
inline auto error_type_name(WGPUErrorType type) -> const char* {
    switch (type) {
        case WGPUErrorType_Validation:
            return "Validation";
        case WGPUErrorType_OutOfMemory:
            return "OutOfMemory";
        case WGPUErrorType_Internal:
            return "Internal";
        case WGPUErrorType_Unknown:
            return "Unknown";
        case WGPUErrorType_NoError:
            return "NoError";
        default:
            return "Unknown";
    }
}

/// RAII wrapper for WebGPU error scopes.
///
/// Supports nested error scopes via multiple filters. Filters are pushed in order,
/// and popped in reverse order (LIFO). If any scope captures an error, it will be reported.
///
/// Example with single filter:
///   ErrorScope scope(device, WGPUErrorFilter_Validation, "logger", "operation");
///   // ... WebGPU operations ...
///   scope.pop_and_throw_if_error();
///
/// Example with multiple filters (nested scopes):
///   ErrorScope scope(device, {WGPUErrorFilter_Validation, WGPUErrorFilter_OutOfMemory},
///                    "logger", "operation");
///   // ... WebGPU operations ...
///   scope.pop_and_throw_if_error();
class ErrorScope {
   public:
    struct Result {
        WGPUPopErrorScopeStatus status = WGPUPopErrorScopeStatus_Success;
        WGPUErrorType type = WGPUErrorType_NoError;
        std::string message;
    };

    /// Construct with a single error filter
    ErrorScope(const Device& device, WGPUErrorFilter filter, std::string_view logger_name,
               std::string_view context_label);

    /// Construct with multiple error filters (nested scopes).
    /// Filters are pushed in order and popped in reverse (LIFO).
    ErrorScope(const Device& device, std::initializer_list<WGPUErrorFilter> filters,
               std::string_view logger_name, std::string_view context_label);

    NO_COPY_MOVE(ErrorScope);

    ~ErrorScope();

    void pop_and_throw_if_error();

   private:
    void pop_and_wait();
    void log_all_errors() const;
    void log_error(const Result& result, std::size_t scope_index) const;

    const Device& m_device;
    std::string m_logger_name;
    std::string m_context_label;
    std::vector<Result> m_results;
    std::size_t m_scope_count = 0;
    bool m_popped = false;
};

}  // namespace pts::webgpu
