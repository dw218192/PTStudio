#include <core/diagnostics.h>
#include <core/imgui/fileDialogue.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#else
#include <portable-file-dialogs.h>

#include <fstream>
#include <sstream>
#endif

#if defined(__EMSCRIPTEN__)

namespace {
std::function<void(ImGui::FileDialogueResult)> s_pending_callback;

extern "C" {
EMSCRIPTEN_KEEPALIVE
void pts_file_dialog_callback(const char* name, const char* data, int size) {
    if (s_pending_callback) {
        ImGui::FileDialogueResult result;
        result.name = name ? name : "";
        result.contents = std::string(data, static_cast<size_t>(size));
        auto cb = std::move(s_pending_callback);
        s_pending_callback = nullptr;
        cb(std::move(result));
    }
}
}
}  // namespace

void ImGui::FileDialogueAsync(FileDialogueMode mode, const std::string& accept,
                              std::function<void(FileDialogueResult)> on_result) {
    PTS_UNUSED(mode);
    s_pending_callback = std::move(on_result);

    // clang-format off
    EM_ASM({
        var accept = UTF8ToString($0);
        var input = document.createElement('input');
        input.type = 'file';
        if (accept) input.accept = accept;
        input.onchange = function(e) {
            var file = e.target.files[0];
            if (!file) return;
            var reader = new FileReader();
            reader.onload = function() {
                var data = new Uint8Array(reader.result);
                var nameLen = lengthBytesUTF8(file.name) + 1;
                var namePtr = _malloc(nameLen);
                stringToUTF8(file.name, namePtr, nameLen);
                var dataPtr = _malloc(data.length);
                HEAPU8.set(data, dataPtr);
                _pts_file_dialog_callback(namePtr, dataPtr, data.length);
                _free(namePtr);
                _free(dataPtr);
            };
            reader.readAsArrayBuffer(file);
        };
        input.click();
    }, accept.c_str());
    // clang-format on
}

#else

namespace {
auto open_file_dialog(ImGui::FileDialogueMode mode) -> std::string {
    std::vector<std::string> filters = {"Scene Files", "*.usda *.usdc *.usd", "All Files", "*"};
    if (mode == ImGui::FileDialogueMode::Open) {
        auto selection = pfd::open_file("Open File", "", filters).result();
        if (!selection.empty()) return selection[0];
    } else {
        auto result = pfd::save_file("Save File", "", filters).result();
        if (!result.empty()) return result;
    }
    return {};
}
}  // namespace

void ImGui::FileDialogueAsync(FileDialogueMode mode, const std::string& accept,
                              std::function<void(FileDialogueResult)> on_result) {
    PTS_UNUSED(accept);
    auto path = open_file_dialog(mode);
    if (path.empty()) return;

    std::ifstream file(path, std::ios::binary);
    if (!file) return;

    std::ostringstream ss;
    ss << file.rdbuf();

    FileDialogueResult result;
    result.name = path;
    result.contents = ss.str();
    on_result(std::move(result));
}

#endif
