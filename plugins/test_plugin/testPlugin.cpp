#include <core/pluginUtils.h>
#include <test_plugin/interface.h>
#include <test_plugin/math.h>

#include <cstdio>  // For snprintf

// ============================================================================
// Interface ID constants
// ============================================================================
// Interface IDs are now defined in the public headers (interface.h, math.h)
// so that clients know what to query for.

// Create compile-time string constants from the macro definitions
PTS_INTERFACE_ID(kTestInterfaceId, TEST_PLUGIN_INTERFACE_V1_ID);
PTS_INTERFACE_ID(kTestMathInterfaceId, TEST_PLUGIN_MATH_INTERFACE_V1_ID);

// ============================================================================
// Plugin implementation
// ============================================================================

/**
 * Test plugin implementation.
 * Demonstrates:
 * - Basic plugin lifecycle (on_load/on_unload)
 * - Interface exposure via ABI-safe function table
 */
class TestPlugin : public pts::IPlugin {
   public:
    explicit TestPlugin(PtsHostApi* host_api) : pts::IPlugin(host_api) {
        // Constructor should not perform fallible operations
        // Logger creation is done in on_load()
    }

    ~TestPlugin() override {
    }

    bool on_load() override {
        // Create logger - this must succeed for the plugin to be valid
        if (!host_api()) {
            return false;  // No host API, can't initialize
        }

        m_logger = host_api()->create_logger("TestPlugin");
        if (!m_logger) {
            return false;  // Failed to create logger
        }

        // From this point on, logger is always valid
        host_api()->log_info(m_logger, "on_load() called - Plugin is now active!");
        m_call_count = 0;
        return true;
    }

    void on_unload() override {
        host_api()->log_info(m_logger, "on_unload() called - Plugin shutting down...");
        char msg[64];
        snprintf(msg, sizeof(msg), "Total function calls: %d", m_call_count);
        host_api()->log_info(m_logger, msg);
    }

    // Static function to return the test interface function table
    // Plugin instance is automatically available from thread-local storage
    static TestPluginInterfaceV1* get_test_interface() {
        static TestPluginInterfaceV1 interface_table = {
            TEST_PLUGIN_INTERFACE_V1_VERSION, PTS_METHOD(TestPlugin, get_greeting, const char*),
            PTS_METHOD(TestPlugin, add_numbers, int32_t, int32_t, int32_t),
            PTS_METHOD(TestPlugin, print_message, void, const char*)};
        return &interface_table;
    }

    // Static function to return the math interface function table
    static TestPluginMathInterfaceV1* get_math_interface() {
        static TestPluginMathInterfaceV1 interface_table = {
            TEST_PLUGIN_MATH_INTERFACE_V1_VERSION,
            PTS_METHOD(TestPlugin, multiply, int32_t, int32_t, int32_t),
            PTS_METHOD(TestPlugin, divide, double, double, double)};
        return &interface_table;
    }

   private:
    int32_t m_call_count = 0;
    LoggerHandle m_logger = nullptr;

    // Implementation functions - logger is guaranteed valid (precondition)
    const char* get_greeting() {
        m_call_count++;
        host_api()->log_debug(m_logger, "get_greeting() called");
        return "Hello from TestPlugin!";
    }

    int32_t add_numbers(int32_t a, int32_t b) {
        m_call_count++;
        int32_t result = a + b;
        char msg[128];
        snprintf(msg, sizeof(msg), "add_numbers(%d, %d) = %d", a, b, result);
        host_api()->log_debug(m_logger, msg);
        return result;
    }

    void print_message(const char* message) {
        m_call_count++;
        if (message) {
            char msg[256];
            snprintf(msg, sizeof(msg), "print_message: %s", message);
            host_api()->log_info(m_logger, msg);
        }
    }

    int32_t multiply(int32_t a, int32_t b) {
        m_call_count++;
        int32_t result = a * b;
        char msg[128];
        snprintf(msg, sizeof(msg), "multiply(%d, %d) = %d", a, b, result);
        host_api()->log_debug(m_logger, msg);
        return result;
    }

    double divide(double a, double b) {
        m_call_count++;
        if (b == 0.0) {
            host_api()->log_error(m_logger, "divide: Division by zero!");
            return 0.0;
        }
        double result = a / b;
        char msg[128];
        snprintf(msg, sizeof(msg), "divide(%f, %f) = %f", a, b, result);
        host_api()->log_debug(m_logger, msg);
        return result;
    }
};

PTS_PLUGIN_INTERFACES(TestPlugin,
                      PTS_INTERFACE(TestPlugin, kTestInterfaceId, TestPluginInterfaceV1,
                                    TestPlugin::get_test_interface)
                          PTS_INTERFACE(TestPlugin, kTestMathInterfaceId, TestPluginMathInterfaceV1,
                                        TestPlugin::get_math_interface))

// Export the plugin using the convenience macro
PTS_PLUGIN_DEFINE(TestPlugin, PTS_PLUGIN_KIND_SUBSYSTEM, "TestPlugin", "Test Plugin", "1.0.0")
