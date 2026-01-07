#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <GL/glew.h>
#include <doctest/doctest.h>

#include <vector>

#include "glBuffer.h"
#include "glFrameBuffer.h"
#include "glRenderBuffer.h"
#include "glResource.h"
#include "glTexture.h"
#include "glVertexArray.h"
#include "shader.h"
#include "uniformVar.h"

using namespace PTS;

namespace {
struct TestResource : GLResource {
    TestResource() : GLResource(0) {
    }
};
}  // namespace

TEST_CASE("GLResource basic functionality") {
    TestResource res;
    CHECK_FALSE(res.valid());
    CHECK(res.handle() == 0);
}

TEST_CASE("GLBuffer interface") {
    // Test that GLBuffer has expected static methods
    // Note: create() requires OpenGL context, so we can't test it here
    // without proper setup

    SUBCASE("Type aliases exist") {
        // Verify that GLBufferRef is defined
        static_assert(std::is_same_v<GLBufferRef, UniqueGLResRef<GLBuffer>>);
    }
}

TEST_CASE("GLResourceDeleter") {
    GLResourceDeleter deleter;
    auto* res = new TestResource();

    // Should not crash (though actual deletion requires proper OpenGL context)
    deleter(res);
}

// Add more tests as needed for other components
// Note: Most GL wrapper functionality requires an active OpenGL context
// For full integration tests, you would need to:
// 1. Initialize GLFW window
// 2. Initialize GLEW
// 3. Create OpenGL context
// 4. Then test actual GL resource creation and manipulation
