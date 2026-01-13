#include <core/plugin.h>

#include <iostream>

/**
 * Simple test plugin implementation.
 * Demonstrates the minimal plugin interface.
 */
class TestPlugin : public IPlugin {
   public:
    TestPlugin() {
        std::cout << "[TestPlugin] Constructor called" << std::endl;
    }

    ~TestPlugin() override {
        std::cout << "[TestPlugin] Destructor called" << std::endl;
    }

    void on_load() override {
        std::cout << "[TestPlugin] on_load() called - Plugin is now active!" << std::endl;
    }

    void on_unload() override {
        std::cout << "[TestPlugin] on_unload() called - Plugin shutting down..." << std::endl;
    }
};

// Export the plugin using the convenience macro
PTS_PLUGIN_DEFINE(TestPlugin, PTS_PLUGIN_KIND_TEST, "TestPlugin", "Test Plugin", "1.0.0")
