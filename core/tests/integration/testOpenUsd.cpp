#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <pxr/usd/usd/stage.h>

TEST_CASE("OpenUSD - Create in-memory stage") {
    auto stage = pxr::UsdStage::CreateInMemory();
    CHECK(stage);
}
