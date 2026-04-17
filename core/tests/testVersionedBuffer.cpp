#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NOMINMAX
#include <core/rendering/versionedBuffer.h>
#include <core/rendering/webgpu/device.h>
#include <doctest/doctest.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstdint>
#include <utility>

using pts::rendering::ImportedBuffer;
using pts::rendering::VersionedBuffer;

namespace {

auto create_test_logger() -> std::shared_ptr<spdlog::logger> {
    auto logger = spdlog::get("versioned_buffer_test");
    if (!logger) {
        logger = spdlog::stdout_color_mt("versioned_buffer_test");
    }
    logger->set_level(spdlog::level::warn);
    return logger;
}

constexpr auto k_storage_copy_dst =
    static_cast<WGPUBufferUsage>(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);

}  // namespace

TEST_CASE("VersionedBuffer - default-constructed is empty") {
    VersionedBuffer<uint32_t> buf;
    CHECK_FALSE(buf.is_valid());
    CHECK(buf.handle() == nullptr);
    CHECK(buf.size_bytes() == 0);
    CHECK(buf.version() == 0);
}

TEST_CASE("VersionedBuffer - construction allocates and version starts at 0") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 4, k_storage_copy_dst, "vb_initial");
    CHECK(buf.is_valid());
    CHECK(buf.handle() != nullptr);
    CHECK(buf.size_bytes() == 4 * sizeof(uint32_t));
    CHECK(buf.version() == 0);
}

TEST_CASE("VersionedBuffer - write bumps version by exactly 1") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 4, k_storage_copy_dst);
    std::array<uint32_t, 4> data = {1, 2, 3, 4};

    CHECK(buf.version() == 0);
    buf.write(device.queue(), data.data(), data.size());
    CHECK(buf.version() == 1);
    buf.write(device.queue(), data.data(), data.size());
    CHECK(buf.version() == 2);
}

TEST_CASE("VersionedBuffer - identical writes bump each time (no value diffing)") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 2, k_storage_copy_dst);
    std::array<uint32_t, 2> data = {42, 42};

    buf.write(device.queue(), data.data(), data.size());
    auto v1 = buf.version();
    buf.write(device.queue(), data.data(), data.size());
    CHECK(buf.version() == v1 + 1);
}

TEST_CASE("VersionedBuffer - imported() matches internal state") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 3, k_storage_copy_dst);
    std::array<uint32_t, 3> data = {7, 8, 9};
    buf.write(device.queue(), data.data(), data.size());

    ImportedBuffer imp = buf.imported();
    CHECK(imp.handle == buf.handle());
    CHECK(imp.size_bytes == buf.size_bytes());
    CHECK(imp.version == buf.version());
}

TEST_CASE("VersionedBuffer - move construction preserves handle and version") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> src(device.handle(), 2, k_storage_copy_dst);
    std::array<uint32_t, 2> data = {1, 2};
    src.write(device.queue(), data.data(), data.size());
    auto src_handle = src.handle();
    auto src_version = src.version();
    auto src_size = src.size_bytes();

    VersionedBuffer<uint32_t> dst(std::move(src));
    CHECK(dst.handle() == src_handle);
    CHECK(dst.version() == src_version);
    CHECK(dst.size_bytes() == src_size);
    CHECK_FALSE(src.is_valid());
    CHECK(src.handle() == nullptr);
    CHECK(src.size_bytes() == 0);
}

TEST_CASE("VersionedBuffer - move assignment releases target and transfers source") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> dst(device.handle(), 1, k_storage_copy_dst);
    VersionedBuffer<uint32_t> src(device.handle(), 2, k_storage_copy_dst);
    std::array<uint32_t, 2> data = {5, 6};
    src.write(device.queue(), data.data(), data.size());
    auto src_handle = src.handle();
    auto src_size = src.size_bytes();
    auto src_version = src.version();

    dst = std::move(src);
    CHECK(dst.handle() == src_handle);
    CHECK(dst.size_bytes() == src_size);
    CHECK(dst.version() == src_version);
    CHECK_FALSE(src.is_valid());
}

TEST_CASE("VersionedBuffer - write grows when count exceeds initial capacity") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 2, k_storage_copy_dst);
    auto initial_handle = buf.handle();
    auto initial_size = buf.size_bytes();
    CHECK(initial_size == 2 * sizeof(uint32_t));

    std::array<uint32_t, 8> large = {1, 2, 3, 4, 5, 6, 7, 8};
    buf.write(device.queue(), large.data(), large.size());

    CHECK(buf.size_bytes() >= 8 * sizeof(uint32_t));
    CHECK(buf.handle() != initial_handle);
    CHECK(buf.version() == 1);
}

TEST_CASE("VersionedBuffer - write with count == 0 bumps version, skips upload") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 1, k_storage_copy_dst);
    buf.write(device.queue(), nullptr, 0);
    CHECK(buf.version() == 1);
    CHECK(buf.is_valid());
}

TEST_CASE("VersionedBuffer - write_at performs partial upload and bumps version") {
    auto logger = create_test_logger();
    auto device = pts::webgpu::Device::create(logger);

    VersionedBuffer<uint32_t> buf(device.handle(), 4, k_storage_copy_dst);
    std::array<uint32_t, 4> full = {1, 2, 3, 4};
    buf.write(device.queue(), full.data(), full.size());
    auto v = buf.version();

    uint32_t patch = 99;
    buf.write_at(device.queue(), 2, &patch, 1);
    CHECK(buf.version() == v + 1);
    CHECK(buf.size_bytes() == 4 * sizeof(uint32_t));
}
