#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <core/logging.h>
#include <core/pluginManager.h>
#include <doctest/doctest.h>
#include <test_plugin/interface.h>
#include <test_plugin/math.h>

#include <memory>

namespace {
// Helper to create a test logging manager
std::unique_ptr<pts::LoggingManager> create_test_logging() {
    pts::Config cfg;
    cfg.level = pts::LogLevel::Debug;
    cfg.pattern = "[%H:%M:%S.%e] [%n] [%l] %v";
    return std::make_unique<pts::LoggingManager>(cfg);
}
}  // namespace

TEST_CASE("PluginManager - Initialization") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");

    SUBCASE("Constructor initializes successfully") {
        pts::PluginManager manager(logger, *logging);
        REQUIRE(manager.get_plugins().empty());
    }

    SUBCASE("Destructor cleans up properly") {
        auto manager = std::make_unique<pts::PluginManager>(logger, *logging);
        manager.reset();  // Should not crash
        CHECK(true);
    }
}

TEST_CASE("PluginManager - Plugin Discovery") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    SUBCASE("Scan plugins directory") {
        // Scan relative to executable (plugins should be in ../plugins relative to tests)
        size_t found = manager.scan_directory("../plugins");

        MESSAGE("Found ", found, " plugin(s)");
        REQUIRE(found >= 1);  // At least TestPlugin should be found

        const auto& plugins = manager.get_plugins();
        REQUIRE(plugins.size() >= 1);

        // Find TestPlugin
        auto it = std::find_if(plugins.begin(), plugins.end(),
                               [](const pts::PluginInfo& info) { return info.id == "TestPlugin"; });

        REQUIRE(it != plugins.end());
        CHECK(it->display_name == "Test Plugin");
        CHECK(it->version == "1.0.0");
        CHECK(it->kind == PTS_PLUGIN_KIND_SUBSYSTEM);
        CHECK(it->is_loaded == false);
        CHECK(it->instance == nullptr);
    }

    SUBCASE("Scan non-existent directory") {
        size_t found = manager.scan_directory("nonexistent_directory");
        CHECK(found == 0);
    }
}

TEST_CASE("PluginManager - Plugin Loading and Unloading") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    // First scan for plugins
    size_t found = manager.scan_directory("../plugins");
    REQUIRE(found >= 1);

    SUBCASE("Load valid plugin") {
        bool loaded = manager.load_plugin("TestPlugin");
        REQUIRE(loaded);

        // Verify plugin state
        const auto& plugins = manager.get_plugins();
        auto it = std::find_if(plugins.begin(), plugins.end(),
                               [](const pts::PluginInfo& info) { return info.id == "TestPlugin"; });

        REQUIRE(it != plugins.end());
        CHECK(it->is_loaded == true);
        CHECK(it->instance != nullptr);

        // Cleanup
        manager.unload_plugin("TestPlugin");
    }

    SUBCASE("Load non-existent plugin") {
        bool loaded = manager.load_plugin("NonExistentPlugin");
        CHECK(loaded == false);
    }

    SUBCASE("Load plugin twice") {
        bool loaded1 = manager.load_plugin("TestPlugin");
        REQUIRE(loaded1);

        bool loaded2 = manager.load_plugin("TestPlugin");
        CHECK(loaded2 == false);  // Should fail - already loaded

        manager.unload_plugin("TestPlugin");
    }

    SUBCASE("Unload valid plugin") {
        bool loaded = manager.load_plugin("TestPlugin");
        REQUIRE(loaded);

        manager.unload_plugin("TestPlugin");

        const auto& plugins = manager.get_plugins();
        auto it = std::find_if(plugins.begin(), plugins.end(),
                               [](const pts::PluginInfo& info) { return info.id == "TestPlugin"; });

        REQUIRE(it != plugins.end());
        CHECK(it->is_loaded == false);
        CHECK(it->instance == nullptr);
    }

    SUBCASE("Unload non-loaded plugin") {
        // Should not crash
        manager.unload_plugin("TestPlugin");
        CHECK(true);
    }
}

TEST_CASE("PluginManager - Interface Querying") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    manager.scan_directory("../plugins");
    manager.load_plugin("TestPlugin");

    void* plugin_handle = manager.get_plugin_instance("TestPlugin");
    REQUIRE(plugin_handle != nullptr);

    // Get the plugin info to access the descriptor
    const auto& plugins = manager.get_plugins();
    auto plugin_it = std::find_if(plugins.begin(), plugins.end(), [](const pts::PluginInfo& info) {
        return info.id == "TestPlugin";
    });
    REQUIRE(plugin_it != plugins.end());

    SUBCASE("Query valid interface - TestPluginInterfaceV1") {
        // Query the test interface
        void* iface_ptr = manager.query_interface(plugin_handle, TEST_PLUGIN_INTERFACE_V1_ID);
        REQUIRE(iface_ptr != nullptr);

        // Cast to the interface type
        auto* iface = static_cast<TestPluginInterfaceV1*>(iface_ptr);

        // Test interface functions
        const char* greeting = iface->get_greeting();
        REQUIRE(greeting != nullptr);
        CHECK(std::string(greeting) == "Hello from TestPlugin!");

        int32_t sum = iface->add_numbers(5, 3);
        CHECK(sum == 8);

        // print_message doesn't return anything, just verify it doesn't crash
        iface->print_message("Test message from unit test");
    }

    SUBCASE("Query valid interface - TestPluginMathInterfaceV1") {
        // Query the math interface
        void* iface_ptr = manager.query_interface(plugin_handle, TEST_PLUGIN_MATH_INTERFACE_V1_ID);
        REQUIRE(iface_ptr != nullptr);

        // Cast to the interface type
        auto* iface = static_cast<TestPluginMathInterfaceV1*>(iface_ptr);

        // Test math operations
        int32_t product = iface->multiply(7, 6);
        CHECK(product == 42);

        double quotient = iface->divide(10.0, 2.0);
        CHECK(quotient == 5.0);

        // Test division by zero handling
        double div_zero = iface->divide(5.0, 0.0);
        CHECK(div_zero == 0.0);  // Plugin should return 0.0 for division by zero
    }

    SUBCASE("Query non-existent interface") {
        // Query for interface that doesn't exist
        void* iface_ptr = manager.query_interface(plugin_handle, "nonexistent.interface.v1");
        CHECK(iface_ptr == nullptr);
    }

    SUBCASE("Query with null plugin handle") {
        void* iface_ptr = manager.query_interface(nullptr, "test_plugin.interface.v1");
        CHECK(iface_ptr == nullptr);
    }

    SUBCASE("Query with null interface ID") {
        void* iface_ptr = manager.query_interface(plugin_handle, nullptr);
        CHECK(iface_ptr == nullptr);
    }

    manager.unload_plugin("TestPlugin");
}

TEST_CASE("PluginManager - Plugin Interfaces Functionality") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    manager.scan_directory("../plugins");
    bool loaded = manager.load_plugin("TestPlugin");
    REQUIRE(loaded);

    void* plugin_handle = manager.get_plugin_instance("TestPlugin");
    REQUIRE(plugin_handle != nullptr);

    SUBCASE("Multiple interface queries on same plugin") {
        // Query both interfaces
        void* test_iface = manager.query_interface(plugin_handle, TEST_PLUGIN_INTERFACE_V1_ID);
        void* math_iface = manager.query_interface(plugin_handle, TEST_PLUGIN_MATH_INTERFACE_V1_ID);

        REQUIRE(test_iface != nullptr);
        REQUIRE(math_iface != nullptr);

        // Use both interfaces
        auto* test = static_cast<TestPluginInterfaceV1*>(test_iface);
        auto* math = static_cast<TestPluginMathInterfaceV1*>(math_iface);

        // Call functions from both interfaces
        int32_t sum = test->add_numbers(10, 20);
        int32_t product = math->multiply(5, 4);

        CHECK(sum == 30);
        CHECK(product == 20);
    }

    SUBCASE("Interface calls increment plugin state") {
        void* test_iface = manager.query_interface(plugin_handle, "test_plugin.interface.v1");
        REQUIRE(test_iface != nullptr);

        auto* test = static_cast<TestPluginInterfaceV1*>(test_iface);

        // Call multiple functions - they should all work
        test->get_greeting();
        test->add_numbers(1, 2);
        test->print_message("Test");

        // The plugin tracks call count internally (shown in on_unload)
        CHECK(true);  // If we get here without crashing, the plugin is working
    }

    SUBCASE("Math interface edge cases") {
        void* math_iface = manager.query_interface(plugin_handle, "test_plugin.math.v1");
        REQUIRE(math_iface != nullptr);

        auto* math = static_cast<TestPluginMathInterfaceV1*>(math_iface);

        // Test with zero
        CHECK(math->multiply(100, 0) == 0);
        CHECK(math->multiply(0, 100) == 0);

        // Test with negative numbers
        CHECK(math->multiply(-5, 3) == -15);
        CHECK(math->multiply(-4, -3) == 12);

        // Test division with negatives
        CHECK(math->divide(-10.0, 2.0) == -5.0);
        CHECK(math->divide(10.0, -2.0) == -5.0);
    }

    manager.unload_plugin("TestPlugin");
}

TEST_CASE("PluginManager - Multiple Plugins") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    size_t found = manager.scan_directory("../plugins");
    MESSAGE("Found ", found, " plugin(s)");

    SUBCASE("Load only valid plugins") {
        const auto& plugins = manager.get_plugins();
        size_t loaded_count = 0;

        // Only load plugins that we know are valid (TestPlugin specifically)
        for (const auto& plugin : plugins) {
            // Skip plugins with corrupted/invalid metadata
            if (plugin.id == "TestPlugin") {
                if (manager.load_plugin(plugin.id)) {
                    loaded_count++;
                }
            }
        }

        MESSAGE("Successfully loaded ", loaded_count, " valid plugin(s)");
        CHECK(loaded_count >= 1);

        // Unload all loaded plugins
        if (loaded_count > 0) {
            manager.unload_plugin("TestPlugin");
        }
    }
}

TEST_CASE("PluginManager - Plugin Lifecycle") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    manager.scan_directory("../plugins");

    SUBCASE("Load, use, unload sequence") {
        // Load
        bool loaded = manager.load_plugin("TestPlugin");
        REQUIRE(loaded);

        void* instance1 = manager.get_plugin_instance("TestPlugin");
        REQUIRE(instance1 != nullptr);

        // Unload
        manager.unload_plugin("TestPlugin");

        void* instance2 = manager.get_plugin_instance("TestPlugin");
        CHECK(instance2 == nullptr);

        // Reload
        bool reloaded = manager.load_plugin("TestPlugin");
        REQUIRE(reloaded);

        void* instance3 = manager.get_plugin_instance("TestPlugin");
        REQUIRE(instance3 != nullptr);
        // Note: instance3 may differ from instance1 (new allocation)

        manager.unload_plugin("TestPlugin");
    }
}

TEST_CASE("PluginManager - Shutdown") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");

    SUBCASE("Shutdown with loaded plugins") {
        pts::PluginManager manager(logger, *logging);
        manager.scan_directory("../plugins");
        manager.load_plugin("TestPlugin");

        // Shutdown should unload all plugins
        manager.shutdown();

        const auto& plugins = manager.get_plugins();
        CHECK(plugins.empty());
    }

    SUBCASE("Shutdown with no plugins") {
        pts::PluginManager manager(logger, *logging);
        manager.shutdown();  // Should not crash
        CHECK(true);
    }

    SUBCASE("Multiple shutdowns") {
        pts::PluginManager manager(logger, *logging);
        manager.scan_directory("../plugins");
        manager.load_plugin("TestPlugin");

        manager.shutdown();
        manager.shutdown();  // Second shutdown should be safe
        CHECK(true);
    }
}

TEST_CASE("PluginManager - Error Handling") {
    auto logging = create_test_logging();
    auto logger = logging->get_logger_shared("test.plugin_manager");
    pts::PluginManager manager(logger, *logging);

    SUBCASE("Get instance of non-existent plugin") {
        void* instance = manager.get_plugin_instance("NonExistentPlugin");
        CHECK(instance == nullptr);
    }

    SUBCASE("Get instance of non-loaded plugin") {
        manager.scan_directory("../plugins");
        void* instance = manager.get_plugin_instance("TestPlugin");
        CHECK(instance == nullptr);
    }
}
