#pragma once

#include <cstdint>

// Interface ID - use this when querying for this interface via PtsHostApi::query_interface
#define TEST_PLUGIN_MATH_INTERFACE_V1_ID "test_plugin.math.v1"

struct TestPluginMathInterfaceV1 {
    int32_t (*multiply)(int32_t a, int32_t b);
    double (*divide)(double a, double b);
};
