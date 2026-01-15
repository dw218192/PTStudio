#include <test_plugin/interface.h>
#include <test_plugin/math.h>

#include <core/loggingUtils.hpp>
#include <core/pluginUtils.hpp>
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
    bool on_load() override {
        logger().log_info("on_load() called - Plugin is now active!");
        m_call_count = 0;
        return true;
    }

    void on_unload() override {
        if (!logger().is_valid()) {
            return;
        }

        logger().log_info("on_unload() called - Plugin shutting down...");
        logger().log_info("Total function calls: {}", m_call_count);
    }

    // Static function to return the test interface function table
    // Plugin instance is automatically available from thread-local storage
    static TestPluginInterfaceV1* get_test_interface() {
        static TestPluginInterfaceV1 interface_table = {
            PTS_METHOD(TestPlugin, get_greeting, const char*),
            PTS_METHOD(TestPlugin, add_numbers, int32_t, int32_t, int32_t),
            PTS_METHOD(TestPlugin, print_message, void, const char*)};
        return &interface_table;
    }

    // Static function to return the math interface function table
    static TestPluginMathInterfaceV1* get_math_interface() {
        static TestPluginMathInterfaceV1 interface_table = {
            PTS_METHOD(TestPlugin, multiply, int32_t, int32_t, int32_t),
            PTS_METHOD(TestPlugin, divide, double, double, double)};
        return &interface_table;
    }

   private:
    int32_t m_call_count = 0;

    // Implementation functions - logger is guaranteed valid (precondition)
    const char* get_greeting() {
        m_call_count++;
        logger().log_debug("get_greeting() called");
        return "Hello from TestPlugin!";
    }

    int32_t add_numbers(int32_t a, int32_t b) {
        m_call_count++;
        int32_t result = a + b;
        logger().log_debug("add_numbers({}, {}) = {}", a, b, result);
        return result;
    }

    void print_message(const char* message) {
        m_call_count++;
        if (message) {
            logger().log_info("print_message: {}", message);
        }
    }

    int32_t multiply(int32_t a, int32_t b) {
        m_call_count++;
        int32_t result = a * b;
        logger().log_debug("multiply({}, {}) = {}", a, b, result);
        return result;
    }

    double divide(double a, double b) {
        m_call_count++;
        if (b == 0.0) {
            logger().log_error("divide: Division by zero!");
            return 0.0;
        }
        double result = a / b;
        logger().log_debug("divide({}, {}) = {}", a, b, result);
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
