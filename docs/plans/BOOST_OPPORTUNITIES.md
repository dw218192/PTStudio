# Boost Library Opportunities in PTStudio

This document identifies areas in the PTStudio codebase where Boost libraries could be leveraged to improve functionality, reduce code complexity, or provide better cross-platform support.

## Currently Enabled Boost Components

- **Boost.Describe** - Enabled for reflection/metadata support
- **Boost.DLL** - Enabled for dynamic library loading

## High-Priority Opportunities

### 1. **Boost.Describe** (Already Enabled)
**Location**: `core/include/core/reflection.h`, `core/include/core/typeTraitsUtil.h`

**Current State**: 
- Custom reflection framework using macros (`BEGIN_REFLECT`, `FIELD`, `END_REFLECT`)
- Manual type trait implementations
- Custom serialization/deserialization logic

**Boost Opportunity**:
- Replace or enhance custom reflection macros with `BOOST_DESCRIBE_STRUCT` and `BOOST_DESCRIBE_ENUM`
- Use `boost::describe::describe_members` for automatic member introspection
- Leverage `boost::describe::has_describe_enumerators` for enum reflection
- Simplify type trait utilities using Boost.Describe's built-in traits

**Example Usage**:
```cpp
// Instead of custom FIELD macros, use:
BOOST_DESCRIBE_STRUCT(SceneObject, (Object), 
    (m_world_transform, Transform),
    (m_local_transform, Transform),
    (m_flags, EditFlags)
)
```

**Files to Update**:
- `core/include/core/reflection.h` - Replace reflection macros
- `core/include/core/typeTraitsUtil.h` - Use Boost.Describe traits
- `core/src/jsonArchive.cpp` - Simplify serialization using Boost.Describe

---

### 2. **Boost.DLL** (Already Enabled)
**Location**: No current usage found

**Current State**: 
- No dynamic library loading detected in codebase
- Static linking for all components

**Boost Opportunity**:
- Load renderer plugins dynamically (e.g., different Vulkan/OpenGL renderers)
- Hot-reload shader compilers or asset loaders
- Plugin system for custom scene objects or materials

**Example Usage**:
```cpp
// Load a renderer plugin
boost::dll::shared_library lib("vulkan_renderer_plugin");
auto create_renderer = lib.get<Renderer*()>("create_renderer");
```

**Potential Files**:
- `core/include/core/renderer.h` - Plugin interface
- `editor/src/editorApplication.cpp` - Plugin loading

---

### 3. **Boost.Filesystem**
**Location**: `core/src/archive.cpp`, `editor/src/editorApplication.cpp`

**Current State**:
- Uses `std::filesystem::exists()` in one place (`editorApplication.cpp:283`)
- Manual file I/O with `std::ifstream`/`std::ofstream` in `archive.cpp`
- String-based file paths (`std::string_view file`)

**Boost Opportunity**:
- Better cross-platform path handling
- Path manipulation utilities (extension, stem, parent_path)
- Directory iteration for asset discovery
- More robust error handling

**Example Usage**:
```cpp
// In archive.cpp
auto PTS::Archive::load_file(std::string_view file, ...) {
    boost::filesystem::path file_path(file);
    if (!boost::filesystem::exists(file_path)) {
        return TL_ERROR("File not found: " + file_path.string());
    }
    // Use boost::filesystem::load_string_file or similar
}
```

**Files to Update**:
- `core/src/archive.cpp` - Replace manual file I/O
- `editor/src/editorApplication.cpp` - Use Boost.Filesystem for path operations
- `core/src/imgui/fileDialogue.cpp` - Path manipulation

---

### 4. **Boost.StringAlgo**
**Location**: `core/include/core/stringManip.h`, various string operations

**Current State**:
- Custom compile-time string manipulation utilities
- Manual string operations throughout codebase
- String formatting with `fmt::format`

**Boost Opportunity**:
- String splitting, trimming, case conversion
- Replace custom compile-time string utilities where runtime is acceptable
- Better string matching and searching

**Example Usage**:
```cpp
// String splitting for file paths, config parsing
std::vector<std::string> parts;
boost::split(parts, path_string, boost::is_any_of("/\\"));

// Case-insensitive comparison
if (boost::iequals(extension, ".json")) { ... }

// Trim whitespace
boost::trim(config_line);
```

**Files to Update**:
- `core/include/core/stringManip.h` - Add runtime string utilities
- `core/src/jsonArchive.cpp` - String parsing utilities
- Any file with manual string parsing

---

### 5. **Boost.ProgramOptions**
**Location**: Currently no C++ command-line parsing (Python uses argparse)

**Current State**:
- No C++ command-line argument parsing
- All CLI handled by Python scripts

**Boost Opportunity**:
- Add C++ command-line interface for editor
- Configuration file parsing
- Runtime option management

**Example Usage**:
```cpp
// In editor/src/main.cpp or editorApplication.cpp
namespace po = boost::program_options;
po::options_description desc("PTStudio Editor Options");
desc.add_options()
    ("scene,s", po::value<std::string>(), "Scene file to load")
    ("headless", "Run without GUI")
    ("verbose,v", "Verbose logging");
```

**Potential Files**:
- `editor/src/main.cpp` - Add CLI parsing
- `editor/src/editorApplication.cpp` - Use parsed options

---

## Medium-Priority Opportunities

### 6. **Boost.Serialization**
**Location**: `core/src/jsonArchive.cpp`, `core/include/core/archive.h`

**Current State**:
- Custom JSON serialization using `nlohmann_json`
- Manual serialization/deserialization for each type

**Boost Opportunity**:
- Binary serialization format (faster, smaller)
- Text/XML serialization alternatives
- Versioning support for serialized data
- Can work alongside JSON for different use cases

**Example Usage**:
```cpp
// Binary archive for faster loading
boost::archive::binary_oarchive oa(ofstream);
oa << scene << camera;
```

**Files to Update**:
- `core/include/core/archive.h` - Add binary archive implementation
- `core/src/jsonArchive.cpp` - Add Boost.Serialization archive

---

### 7. **Boost.TypeTraits**
**Location**: `core/include/core/typeTraitsUtil.h`

**Current State**:
- Custom type trait implementations
- Manual SFINAE patterns

**Boost Opportunity**:
- More comprehensive type traits
- Better C++17/20 compatibility traits
- Simplified type checking code

**Example Usage**:
```cpp
// Replace custom traits with Boost equivalents
#include <boost/type_traits.hpp>
// Use boost::is_base_of, boost::is_pointer, etc.
```

**Files to Update**:
- `core/include/core/typeTraitsUtil.h` - Replace custom traits

---

### 8. **Boost.Container**
**Location**: Various container usage throughout codebase

**Current State**:
- Standard library containers (`std::vector`, `std::map`, etc.)

**Boost Opportunity**:
- `boost::container::flat_map` for better cache locality
- `boost::container::small_vector` for small buffer optimization
- `boost::container::stable_vector` for pointer stability

**Example Usage**:
```cpp
// For frequently accessed, small maps
boost::container::flat_map<ObjectID, Object*> id_to_object;
```

**Potential Files**:
- `core/src/jsonArchive.cpp` - `g_id_to_object`, `g_pointer_to_id` maps
- Any performance-critical container usage

---

### 9. **Boost.Stacktrace**
**Location**: Error handling throughout codebase

**Current State**:
- Error messages with file/line info (`TL_ERROR` macro)
- No stack trace on errors

**Boost Opportunity**:
- Stack traces in error messages for debugging
- Better crash reporting
- Debug information in logs

**Example Usage**:
```cpp
// In utils.h TL_ERROR macro
#include <boost/stacktrace.hpp>
#define TL_ERROR(msg, ...) \
    tl::unexpected { \
        fmt::format(msg, __VA_ARGS__) + \
        "\nStack trace:\n" + \
        boost::stacktrace::to_string(boost::stacktrace::stacktrace()) \
    }
```

**Files to Update**:
- `core/include/core/utils.h` - Add stack traces to errors
- `core/src/logging.cpp` - Stack trace logging

---

## Lower-Priority Opportunities

### 10. **Boost.System**
**Location**: Error handling with `tl::expected<std::string>`

**Current State**:
- String-based error messages
- No error code system

**Boost Opportunity**:
- Structured error codes
- Better error categorization
- Integration with `std::error_code`

**Note**: May conflict with current `tl::expected` usage pattern

---

### 11. **Boost.Regex** (Currently Disabled)
**Location**: No regex usage found

**Current State**:
- No regular expression usage

**Boost Opportunity**:
- Shader code parsing/validation
- Config file parsing
- Asset path validation

**Example Usage**:
```cpp
// Validate shader file paths
boost::regex shader_pattern(R"(.*\.(glsl|vert|frag|comp)$)");
if (boost::regex_match(path, shader_pattern)) { ... }
```

---

### 12. **Boost.Algorithm**
**Location**: Various algorithm usage

**Current State**:
- Standard library algorithms

**Boost Opportunity**:
- Additional search algorithms
- String algorithms (complements Boost.StringAlgo)
- Range algorithms

---

## Summary of Recommended Actions

### Immediate (High Impact, Low Effort)
1. **Enable Boost.Filesystem** - Replace manual file I/O in `archive.cpp`
2. **Use Boost.Describe** - Start migrating reflection macros
3. **Add Boost.Stacktrace** - Improve error debugging

### Short-term (Medium Impact)
4. **Boost.StringAlgo** - Replace manual string operations
5. **Boost.DLL** - Implement plugin system for renderers
6. **Boost.Serialization** - Add binary archive format

### Long-term (Lower Priority)
7. **Boost.ProgramOptions** - Add C++ CLI if needed
8. **Boost.Container** - Optimize performance-critical containers
9. **Boost.TypeTraits** - Simplify type trait code

## Configuration Changes Needed

To enable additional Boost components, update `conanfile.py`:

```python
default_options = {
    # ... existing options ...
    # Enable additional Boost components
    "boost:without_filesystem": False,  # Enable Boost.Filesystem
    "boost:without_stacktrace": False,   # Enable Boost.Stacktrace
    "boost:without_serialization": False, # Enable Boost.Serialization
    "boost:without_program_options": False, # Enable Boost.ProgramOptions
    # Keep describe and dll enabled
    "boost:without_describe": False,
    "boost:without_dll": False,
}
```

## Notes

- **Boost.Describe** and **Boost.DLL** are already enabled but not yet used in the codebase
- Most Boost libraries are header-only or have minimal dependencies
- Consider build time impact when enabling many Boost components
- Some Boost libraries (like Filesystem) may require linking against compiled libraries

