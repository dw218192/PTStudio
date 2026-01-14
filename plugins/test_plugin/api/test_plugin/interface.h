#pragma once

#include <cstdint>

// Interface ID - use this when querying for this interface via PtsHostApi::query_interface
#define TEST_PLUGIN_INTERFACE_V1_ID "test_plugin.interface.v1"
#define TEST_PLUGIN_INTERFACE_V1_VERSION 1

struct TestPluginInterfaceV1 {
    uint32_t version;

    const char* (*get_greeting)();
    int32_t (*add_numbers)(int32_t a, int32_t b);
    void (*print_message)(const char* message);
};
