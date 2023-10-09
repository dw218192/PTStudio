#pragma once
#include <string_view>


// TODO
enum BuiltinAsset {
    BuiltinAsset_DefaultShader,
    BuiltinAsset_Count
};

struct AssetManager {
    AssetManager();
    ~AssetManager();

    void load(std::string_view path);
    void save(std::string_view path);
};