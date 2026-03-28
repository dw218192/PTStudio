#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdUtils/usdzPackage.h>

#include <cstdio>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: usdz_pack <input.usda> <output.usdz>\n");
        return 1;
    }
    bool ok = pxr::UsdUtilsCreateNewUsdzPackage(pxr::SdfAssetPath(argv[1]), argv[2]);
    if (!ok) {
        std::fprintf(stderr, "Failed to package %s -> %s\n", argv[1], argv[2]);
    }
    return ok ? 0 : 1;
}
