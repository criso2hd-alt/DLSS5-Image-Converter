// DLSS5 Scene Capture: a ReShade add-on that saves a screenshot together with
// the game's depth, for the DLSS 5 Image Converter's multi-shot 3D scenes.
//
// Why depth: without it the converter guesses depth from each photo with Depth
// Anything, every guess is warped differently, and a scene built from several
// shots ghosts. The game already knows the real depth.
//
// Why its own capture, not ReShade's screenshot key: the first versions
// piggybacked on ReShade's screenshot and read the depth afterwards. On
// Windows 11, PrintScreen opens the Snipping Tool, which minimises the game,
// so the depth came back empty (4 of 29 shots) or, worse, from after the
// camera had moved. Now one key press grabs colour and depth on the same
// frame, in this add-on, and nothing else is involved.
//
// Depth still comes from ReShade's own choice of depth buffer (with the per-game
// options players already tune): DLSS5Capture.fx copies it, raw, into a float
// texture every frame, and this add-on reads that texture back.
//
// Output, in the chosen folder:
//   <game> <date> <time>_<n>.png          the frame as shown, after effects
//   <game> <date> <time>_<n>.depth.f32    width*height float32, top row first
//   <game> <date> <time>_<n>.depth.json   size, format and provenance

#include <Windows.h>
#include <ShObjIdl.h>
#include <shellapi.h>
#include <wincodec.h>
#include <wrl/client.h>

#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace reshade::api;
using clock_type = std::chrono::steady_clock;

extern "C" __declspec(dllexport) const char *NAME = "DLSS5 Scene Capture";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Captures a screenshot together with the game's depth on one key press, for the DLSS 5 "
    "Image Converter's multi-shot 3D scenes. Needs the DLSS5 Depth Capture effect enabled.";

static constexpr const char *kEffect = "DLSS5Capture.fx";
static constexpr const char *kTexture = "DLSS5_DepthCopy";
static constexpr const char *kViewTexture = "DLSS5_DepthView";
static constexpr const char *kVersion = "0.4.0";
static constexpr const char *kSection = "DLSS5_SCENE_CAPTURE";

// --- settings (persisted in ReShade.ini under [DLSS5_SCENE_CAPTURE]) ----------

struct Settings
{
    std::string folder;              // UTF-8
    uint32_t key = VK_F10;           // F10: unused by ReShade and RenoDX by default
    bool ctrl = false, shift = false, alt = false;
    bool banner = true;
    bool preview = false;            // built-in depth preview window
    float preview_size = 0.3f;       // fraction of the screen width
};

static Settings g_settings;
static bool g_loaded = false;

static std::wstring widen(const std::string &text)
{
    if (text.empty())
        return {};
    const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), int(text.size()), nullptr, 0);
    std::wstring out(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), int(text.size()), out.data(), size);
    return out;
}

static std::string narrow(const std::wstring &text)
{
    if (text.empty())
        return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), int(text.size()), nullptr, 0, nullptr, nullptr);
    std::string out(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), int(text.size()), out.data(), size, nullptr, nullptr);
    return out;
}

static std::filesystem::path game_path()
{
    wchar_t buffer[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    return std::filesystem::path(buffer);
}

static void load_settings(effect_runtime *runtime)
{
    if (g_loaded)
        return;
    g_loaded = true;
    char value[1024] = {};
    size_t size = sizeof(value);
    if (reshade::get_config_value(runtime, kSection, "Folder", value, &size) && value[0])
        g_settings.folder = value;
    else
        g_settings.folder = narrow((game_path().parent_path() / L"DLSS5 Captures").wstring());
    reshade::get_config_value(runtime, kSection, "Key", g_settings.key);
    reshade::get_config_value(runtime, kSection, "Ctrl", g_settings.ctrl);
    reshade::get_config_value(runtime, kSection, "Shift", g_settings.shift);
    reshade::get_config_value(runtime, kSection, "Alt", g_settings.alt);
    reshade::get_config_value(runtime, kSection, "Banner", g_settings.banner);
    reshade::get_config_value(runtime, kSection, "Preview", g_settings.preview);
    reshade::get_config_value(runtime, kSection, "PreviewSize", g_settings.preview_size);
    if (g_settings.preview_size < 0.1f || g_settings.preview_size > 0.9f)
        g_settings.preview_size = 0.3f;
    if (g_settings.key == 0 || g_settings.key > 0xFE)
        g_settings.key = VK_F10;
}

static void save_settings(effect_runtime *runtime)
{
    reshade::set_config_value(runtime, kSection, "Folder", g_settings.folder.c_str());
    reshade::set_config_value(runtime, kSection, "Key", g_settings.key);
    reshade::set_config_value(runtime, kSection, "Ctrl", g_settings.ctrl);
    reshade::set_config_value(runtime, kSection, "Shift", g_settings.shift);
    reshade::set_config_value(runtime, kSection, "Alt", g_settings.alt);
    reshade::set_config_value(runtime, kSection, "Banner", g_settings.banner);
    reshade::set_config_value(runtime, kSection, "Preview", g_settings.preview);
    reshade::set_config_value(runtime, kSection, "PreviewSize", g_settings.preview_size);
}

static std::string key_name(uint32_t key)
{
    if (key >= VK_F1 && key <= VK_F24)
        return "F" + std::to_string(key - VK_F1 + 1);
    // Keys whose scan code is ambiguous without the "extended" flag.
    switch (key)
    {
    case VK_INSERT: return "Insert";
    case VK_DELETE: return "Delete";
    case VK_HOME: return "Home";
    case VK_END: return "End";
    case VK_PRIOR: return "Page Up";
    case VK_NEXT: return "Page Down";
    case VK_LEFT: return "Left";
    case VK_RIGHT: return "Right";
    case VK_UP: return "Up";
    case VK_DOWN: return "Down";
    case VK_SNAPSHOT: return "Print Screen";
    case VK_PAUSE: return "Pause";
    }
    wchar_t name[64] = {};
    const LONG scan = LONG(MapVirtualKeyW(key, MAPVK_VK_TO_VSC)) << 16;
    if (GetKeyNameTextW(scan, name, 64) > 0)
        return narrow(name);
    char fallback[16];
    std::snprintf(fallback, sizeof(fallback), "Key 0x%02X", key);
    return fallback;
}

static std::string binding_name()
{
    std::string name;
    if (g_settings.ctrl) name += "Ctrl + ";
    if (g_settings.shift) name += "Shift + ";
    if (g_settings.alt) name += "Alt + ";
    return name + key_name(g_settings.key);
}

// --- status banner ------------------------------------------------------------

enum class Status { idle, capturing, done, failed };

static std::mutex g_lock;
static Status g_status = Status::idle;
static std::string g_message;
static clock_type::time_point g_status_until;
static std::atomic<bool> g_busy{false};
static std::atomic<int> g_count{0};

static void log_line(reshade::log::level level, const std::string &text)
{
    reshade::log::message(level, text.c_str());
}

static void set_status(Status status, const std::string &message, int seconds)
{
    std::lock_guard<std::mutex> lock(g_lock);
    g_status = status;
    g_message = message;
    g_status_until = clock_type::now() + std::chrono::seconds(seconds);
}

// --- grabbing the frame ---------------------------------------------------------

struct Frame
{
    std::vector<uint8_t> bgr;        // colour, 3 bytes per pixel, what the PNG gets
    uint32_t width = 0, height = 0;
    std::vector<float> depth;        // raw depth values
    uint32_t depth_width = 0, depth_height = 0;
};

static bool grab_colour(effect_runtime *runtime, Frame &frame, std::string &problem)
{
    runtime->get_screenshot_width_and_height(&frame.width, &frame.height);
    const resource back_buffer = runtime->get_current_back_buffer();
    const format fmt = runtime->get_device()->get_resource_desc(back_buffer).texture.format;
    const format typed = format_to_default_typed(fmt, 0);

    // capture_screenshot writes the back buffer's own pixel format, converted
    // here to BGR for the PNG. 10-bit is common in Unreal Engine games (Stray
    // uses it) and is still ordinary SDR, so it is scaled down to 8 bits.
    // Floating-point HDR back buffers are refused rather than saved with
    // wrong colours.
    const bool rgba = typed == format::r8g8b8a8_unorm || typed == format::r8g8b8x8_unorm;
    const bool bgra = typed == format::b8g8r8a8_unorm || typed == format::b8g8r8x8_unorm;
    const bool rgb10 = typed == format::r10g10b10a2_unorm;
    const bool bgr10 = typed == format::b10g10r10a2_unorm;
    if (!rgba && !bgra && !rgb10 && !bgr10)
    {
        problem = "This game's back buffer format is not supported yet (HDR?). Depth was not captured either.";
        return false;
    }
    std::vector<uint8_t> pixels(size_t(frame.width) * frame.height * 4);
    if (!runtime->capture_screenshot(pixels.data()))
    {
        problem = "ReShade could not capture the frame.";
        return false;
    }
    frame.bgr.resize(size_t(frame.width) * frame.height * 3);
    if (rgba || bgra)
    {
        for (size_t i = 0, o = 0; i < pixels.size(); i += 4, o += 3)
        {
            frame.bgr[o + 0] = pixels[i + (rgba ? 2 : 0)];
            frame.bgr[o + 1] = pixels[i + 1];
            frame.bgr[o + 2] = pixels[i + (rgba ? 0 : 2)];
        }
    }
    else
    {
        // Packed little-endian: first channel in bits 0-9, second 10-19, third
        // 20-29, alpha 30-31. The top 8 of each 10 bits make the 8-bit value.
        for (size_t i = 0, o = 0; i < pixels.size(); i += 4, o += 3)
        {
            uint32_t word;
            std::memcpy(&word, &pixels[i], 4);
            const uint8_t first = uint8_t((word >> 2) & 0xFF);
            const uint8_t second = uint8_t((word >> 12) & 0xFF);
            const uint8_t third = uint8_t((word >> 22) & 0xFF);
            frame.bgr[o + 0] = rgb10 ? third : first;     // blue
            frame.bgr[o + 1] = second;                    // green
            frame.bgr[o + 2] = rgb10 ? first : third;     // red
        }
    }
    return true;
}

static bool grab_depth(effect_runtime *runtime, Frame &frame, std::string &problem)
{
    effect_texture_variable variable = runtime->find_texture_variable(kEffect, kTexture);
    if (variable.handle == 0)
        variable = runtime->find_texture_variable(nullptr, kTexture);
    if (variable.handle == 0)
    {
        problem = "Enable the \"DLSS5 Depth Capture\" effect in ReShade to capture depth.";
        return false;
    }
    resource_view view = {}, view_srgb = {};
    runtime->get_texture_binding(variable, &view, &view_srgb);
    if (view.handle == 0)
    {
        problem = "The DLSS5 Depth Capture effect is not running yet.";
        return false;
    }

    device *const dev = runtime->get_device();
    const resource texture = dev->get_resource_from_view(view);
    const resource_desc desc = dev->get_resource_desc(texture);
    frame.depth_width = desc.texture.width;
    frame.depth_height = desc.texture.height;

    frame.depth.resize(size_t(frame.depth_width) * frame.depth_height);
    command_queue *const queue = runtime->get_command_queue();
    command_list *const cmd_list = queue->get_immediate_command_list();

    // Two readback routes. Texture-to-buffer copies are the D3D12/Vulkan way,
    // but ReShade documents them as unavailable on some APIs, and on D3D11
    // (Stray) the copy silently did nothing: every capture came back as zeros
    // while ReShade's own depth view was fine. There a CPU-readable texture is
    // the right target instead.
    if (dev->check_capability(device_caps::copy_buffer_to_texture))
    {
        // D3D12 needs every buffer row to start on a 256-byte boundary (64
        // floats), so rows are padded for the copy and stripped afterwards.
        const uint32_t row_length = (frame.depth_width + 63u) & ~63u;
        const uint64_t size = uint64_t(row_length) * frame.depth_height * sizeof(float);
        resource buffer = {};
        if (!dev->create_resource(resource_desc(size, memory_heap::readback, resource_usage::copy_dest),
                                  nullptr, resource_usage::copy_dest, &buffer))
        {
            problem = "Could not create the depth readback buffer.";
            return false;
        }
        cmd_list->barrier(texture, resource_usage::shader_resource, resource_usage::copy_source);
        cmd_list->copy_texture_to_buffer(texture, 0, nullptr, buffer, 0, row_length, frame.depth_height);
        cmd_list->barrier(texture, resource_usage::copy_source, resource_usage::shader_resource);
        queue->flush_immediate_command_list();
        queue->wait_idle();

        void *mapped = nullptr;
        const bool ok = dev->map_buffer_region(buffer, 0, size, map_access::read_only, &mapped) && mapped;
        if (ok)
        {
            const float *source = static_cast<const float *>(mapped);
            for (uint32_t y = 0; y < frame.depth_height; ++y)
                std::memcpy(&frame.depth[size_t(y) * frame.depth_width], source + size_t(y) * row_length,
                            frame.depth_width * sizeof(float));
            dev->unmap_buffer_region(buffer);
        }
        else
        {
            problem = "Could not read the depth back from the GPU.";
        }
        dev->destroy_resource(buffer);
        return ok;
    }

    resource staging = {};
    if (!dev->create_resource(resource_desc(frame.depth_width, frame.depth_height, 1, 1, format::r32_float, 1,
                                            memory_heap::readback, resource_usage::copy_dest),
                              nullptr, resource_usage::copy_dest, &staging))
    {
        problem = "Could not create the depth readback texture.";
        return false;
    }
    cmd_list->barrier(texture, resource_usage::shader_resource, resource_usage::copy_source);
    cmd_list->copy_resource(texture, staging);
    cmd_list->barrier(texture, resource_usage::copy_source, resource_usage::shader_resource);
    queue->flush_immediate_command_list();
    queue->wait_idle();

    subresource_data mapped = {};
    const bool ok = dev->map_texture_region(staging, 0, nullptr, map_access::read_only, &mapped) && mapped.data;
    if (ok)
    {
        const uint8_t *source = static_cast<const uint8_t *>(mapped.data);
        for (uint32_t y = 0; y < frame.depth_height; ++y)
            std::memcpy(&frame.depth[size_t(y) * frame.depth_width], source + size_t(y) * mapped.row_pitch,
                        frame.depth_width * sizeof(float));
        dev->unmap_texture_region(staging, 0);
    }
    else
    {
        problem = "Could not read the depth back from the GPU.";
    }
    dev->destroy_resource(staging);
    return ok;
}

// --- writing (background thread) -----------------------------------------------

// PNG through Windows' own imaging component: no library to ship, and far
// faster than a hand-rolled deflate on a 4K frame.
static bool write_png(const std::filesystem::path &path, const Frame &frame)
{
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool ok = false;
    {
        ComPtr<IWICImagingFactory> factory;
        ComPtr<IWICStream> stream;
        ComPtr<IWICBitmapEncoder> encoder;
        ComPtr<IWICBitmapFrameEncode> target;
        ComPtr<IPropertyBag2> options;
        WICPixelFormatGUID pixel_format = GUID_WICPixelFormat24bppBGR;
        ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(&factory))) &&
             SUCCEEDED(factory->CreateStream(&stream)) &&
             SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
             SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
             SUCCEEDED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache)) &&
             SUCCEEDED(encoder->CreateNewFrame(&target, &options)) &&
             SUCCEEDED(target->Initialize(options.Get())) &&
             SUCCEEDED(target->SetSize(frame.width, frame.height)) &&
             SUCCEEDED(target->SetPixelFormat(&pixel_format)) &&
             IsEqualGUID(pixel_format, GUID_WICPixelFormat24bppBGR) &&
             SUCCEEDED(target->WritePixels(frame.height, frame.width * 3, UINT(frame.bgr.size()),
                                           const_cast<BYTE *>(frame.bgr.data()))) &&
             SUCCEEDED(target->Commit()) &&
             SUCCEEDED(encoder->Commit());
    }
    if (SUCCEEDED(init))
        CoUninitialize();
    return ok;
}

static std::string json_escape(const std::string &text)
{
    std::string out;
    for (char c : text)
    {
        if (c == '\\' || c == '"')
            out += '\\';
        out += c;
    }
    return out;
}

static void write_frame(Frame frame, std::filesystem::path png)
{
    std::string failure;
    std::error_code error;
    std::filesystem::create_directories(png.parent_path(), error);

    if (!write_png(png, frame))
        failure = "Could not write " + narrow(png.filename().wstring());

    float lowest = 1e30f, highest = -1e30f;
    size_t at_zero = 0, at_one = 0;
    if (failure.empty())
    {
        const std::filesystem::path depth_path = std::filesystem::path(png).replace_extension(".depth.f32");
        if (FILE *file = _wfopen(depth_path.c_str(), L"wb"))
        {
            std::fwrite(frame.depth.data(), sizeof(float), frame.depth.size(), file);
            std::fclose(file);
        }
        else
        {
            failure = "Could not write the depth file.";
        }
        // A few numbers about the raw values, so a reader can tell reversed Z
        // (sky at 0) from standard Z (sky at 1), or spot an empty capture,
        // without loading the whole file.
        for (float value : frame.depth)
        {
            lowest = value < lowest ? value : lowest;
            highest = value > highest ? value : highest;
            at_zero += value <= 1e-7f;
            at_one += value >= 1.0f - 1e-7f;
        }
        const double count = double(frame.depth.size());
        const std::filesystem::path info_path = std::filesystem::path(png).replace_extension(".depth.json");
        if (FILE *info = _wfopen(info_path.c_str(), L"wb"))
        {
            std::fprintf(info,
                "{\n"
                "  \"version\": \"%s\",\n"
                "  \"screenshot\": \"%s\",\n"
                "  \"width\": %u,\n"
                "  \"height\": %u,\n"
                "  \"screenshot_width\": %u,\n"
                "  \"screenshot_height\": %u,\n"
                "  \"format\": \"float32\",\n"
                "  \"layout\": \"row-major, top row first, little-endian\",\n"
                "  \"source\": \"ReShade's selected depth buffer, raw (not linearised)\",\n"
                "  \"same_frame_as_screenshot\": true,\n"
                "  \"min\": %.9g,\n"
                "  \"max\": %.9g,\n"
                "  \"fraction_at_zero\": %.6f,\n"
                "  \"fraction_at_one\": %.6f\n"
                "}\n",
                kVersion, json_escape(narrow(png.filename().wstring())).c_str(),
                frame.depth_width, frame.depth_height, frame.width, frame.height,
                lowest, highest, double(at_zero) / count, double(at_one) / count);
            std::fclose(info);
        }
    }

    if (!failure.empty())
    {
        log_line(reshade::log::level::error, failure);
        set_status(Status::failed, failure, 8);
    }
    else if (highest <= 0.0f)
    {
        log_line(reshade::log::level::warning, "Saved " + narrow(png.filename().wstring()) +
                                                   " but its depth is empty.");
        set_status(Status::failed, "Saved, but the depth was empty. Check ReShade's Generic Depth settings.", 8);
    }
    else
    {
        ++g_count;
        log_line(reshade::log::level::info, "Saved " + narrow(png.filename().wstring()) + " with depth");
        set_status(Status::done, "Depth capture complete. You can take another screenshot.", 4);
    }
    g_busy = false;
}

static std::filesystem::path next_path()
{
    static int counter = 0;
    const std::time_t now = std::time(nullptr);
    std::tm local = {};
    localtime_s(&local, &now);
    wchar_t stamp[64];
    std::wcsftime(stamp, 64, L"%Y-%m-%d %H-%M-%S", &local);
    const std::wstring name = game_path().stem().wstring() + L" " + stamp + L"_" +
                              std::to_wstring(++counter) + L".png";
    return std::filesystem::path(widen(g_settings.folder)) / name;
}

// --- the capture key -------------------------------------------------------------

static bool g_listening = false;     // settings UI is waiting for a new key

static bool binding_pressed(effect_runtime *runtime)
{
    if (g_listening || !runtime->is_key_pressed(g_settings.key))
        return false;
    return runtime->is_key_down(VK_CONTROL) == g_settings.ctrl &&
           runtime->is_key_down(VK_SHIFT) == g_settings.shift &&
           runtime->is_key_down(VK_MENU) == g_settings.alt;
}

// After ReShade's effects, before its overlay: the frame exactly as the player
// sees it, without the banner or any ReShade window in it. The depth copy has
// run by now too, since it is one of those effects.
static void on_finish_effects(effect_runtime *runtime, command_list *, resource_view, resource_view)
{
    load_settings(runtime);
    if (!binding_pressed(runtime))
        return;
    if (g_busy)
    {
        set_status(Status::capturing, "Still saving the previous capture. Wait until complete...", 60);
        return;
    }

    Frame frame;
    std::string problem;
    if (!grab_depth(runtime, frame, problem) || !grab_colour(runtime, frame, problem))
    {
        log_line(reshade::log::level::warning, "Capture failed: " + problem);
        set_status(Status::failed, problem, 8);
        return;
    }
    g_busy = true;
    set_status(Status::capturing, "Capturing depth buffer. Wait until complete...", 60);
    std::thread(write_frame, std::move(frame), next_path()).detach();
}

// --- on-screen banner ------------------------------------------------------------

// Picture-in-picture view of exactly what a capture will save. Drawn in the
// overlay, which comes after the capture point, so it never ends up in a
// screenshot.
static void draw_preview(effect_runtime *runtime)
{
    effect_texture_variable variable = runtime->find_texture_variable(kEffect, kViewTexture);
    resource_view view = {}, view_srgb = {};
    if (variable.handle != 0)
        runtime->get_texture_binding(variable, &view, &view_srgb);

    const ImVec2 screen = ImGui::GetIO().DisplaySize;
    const float width = screen.x * g_settings.preview_size;
    const float height = width * (screen.y / std::max(screen.x, 1.0f));
    ImGui::SetNextWindowPos(ImVec2(screen.x - 20.0f, 20.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.85f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##dlss5_depth_preview", nullptr, flags))
    {
        ImGui::TextUnformatted("Depth that will be saved (near = white, sky = blue)");
        if (view.handle != 0)
            ImGui::Image(ImTextureRef(static_cast<ImTextureID>(view.handle)), ImVec2(width, height),
                         ImVec2(0, 0), ImVec2(1, 1));
        else
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f),
                               "Enable the DLSS5 Depth Capture effect to see the depth.");
    }
    ImGui::End();
}

static void on_overlay(effect_runtime *runtime)
{
    if (g_settings.preview)
        draw_preview(runtime);
    if (!g_settings.banner)
        return;
    Status status;
    std::string message;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        if (g_status == Status::idle)
            return;
        if (g_status != Status::capturing && clock_type::now() > g_status_until)
        {
            g_status = Status::idle;
            return;
        }
        status = g_status;
        message = g_message;
    }
    // GetIO rather than GetMainViewport: ReShade's function table for add-ons
    // does not expose the viewport call.
    const ImVec2 screen = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(screen.x * 0.5f, 40.0f), ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.75f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##dlss5_scene_capture", nullptr, flags))
    {
        const ImVec4 colour = status == Status::done     ? ImVec4(0.45f, 0.95f, 0.45f, 1.0f)
                              : status == Status::failed ? ImVec4(1.0f, 0.45f, 0.4f, 1.0f)
                                                         : ImVec4(1.0f, 0.85f, 0.3f, 1.0f);
        ImGui::TextColored(colour, "%s", message.c_str());
    }
    ImGui::End();
}

// --- settings panel (ReShade's Add-ons tab) -----------------------------------------

// The folder dialog runs on its own thread: it is modal, and on the render
// thread it would freeze the game while open.
static std::atomic<bool> g_browsing{false};
static std::mutex g_browse_lock;
static std::string g_browse_result;

static void browse_thread(std::wstring start)
{
    const HRESULT init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    {
        ComPtr<IFileOpenDialog> dialog;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog))))
        {
            DWORD options = 0;
            dialog->GetOptions(&options);
            dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
            dialog->SetTitle(L"Folder for DLSS5 scene captures");
            ComPtr<IShellItem> folder;
            if (!start.empty() && SUCCEEDED(SHCreateItemFromParsingName(start.c_str(), nullptr, IID_PPV_ARGS(&folder))))
                dialog->SetFolder(folder.Get());
            if (SUCCEEDED(dialog->Show(nullptr)))
            {
                ComPtr<IShellItem> result;
                PWSTR path = nullptr;
                if (SUCCEEDED(dialog->GetResult(&result)) &&
                    SUCCEEDED(result->GetDisplayName(SIGDN_FILESYSPATH, &path)))
                {
                    std::lock_guard<std::mutex> lock(g_browse_lock);
                    g_browse_result = narrow(path);
                    CoTaskMemFree(path);
                }
            }
        }
    }
    if (SUCCEEDED(init))
        CoUninitialize();
    g_browsing = false;
}

static void on_settings(effect_runtime *runtime)
{
    load_settings(runtime);

    {
        std::lock_guard<std::mutex> lock(g_browse_lock);
        if (!g_browse_result.empty())
        {
            g_settings.folder = g_browse_result;
            g_browse_result.clear();
            save_settings(runtime);
        }
    }

    ImGui::TextWrapped("One key press saves the frame and the game's depth side by side, for the "
                       "converter's multi-shot 3D scenes. Keep the \"DLSS5 Depth Capture\" effect enabled.");
    ImGui::Separator();

    // Output folder.
    ImGui::TextUnformatted("Save to");
    static char folder[1024] = {};
    static std::string shown;
    if (shown != g_settings.folder)
    {
        std::snprintf(folder, sizeof(folder), "%s", g_settings.folder.c_str());
        shown = g_settings.folder;
    }
    ImGui::SetNextItemWidth(-200.0f);
    if (ImGui::InputText("##folder", folder, sizeof(folder), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        g_settings.folder = folder;
        shown = g_settings.folder;
        save_settings(runtime);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Type a folder and press Enter, or use Browse.");
    ImGui::SameLine();
    ImGui::BeginDisabled(g_browsing);
    if (ImGui::Button("Browse..."))
    {
        g_browsing = true;
        std::thread(browse_thread, widen(g_settings.folder)).detach();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open"))
    {
        std::error_code error;
        std::filesystem::create_directories(widen(g_settings.folder), error);
        ShellExecuteW(nullptr, L"open", widen(g_settings.folder).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }
    if (g_browsing)
        ImGui::TextDisabled("Choose a folder in the window that opened (it may be behind the game).");

    // Capture key.
    ImGui::Spacing();
    ImGui::TextUnformatted("Capture key");
    const std::string label = g_listening ? std::string("Press a key... (Esc to cancel)") : binding_name();
    if (ImGui::Button((label + "##bind").c_str(), ImVec2(260.0f, 0.0f)))
        g_listening = true;
    if (g_listening)
    {
        if (runtime->is_key_pressed(VK_ESCAPE))
        {
            g_listening = false;
        }
        else
        {
            // Mouse buttons (1-6) and bare modifiers are not bindable; a modifier
            // held while pressing the key becomes part of the binding.
            for (uint32_t key = 0x08; key <= 0xFE; ++key)
            {
                if (key == VK_SHIFT || key == VK_CONTROL || key == VK_MENU || key == VK_LSHIFT ||
                    key == VK_RSHIFT || key == VK_LCONTROL || key == VK_RCONTROL || key == VK_LMENU ||
                    key == VK_RMENU || key == VK_LWIN || key == VK_RWIN || key == VK_ESCAPE)
                    continue;
                if (runtime->is_key_pressed(key))
                {
                    g_settings.key = key;
                    g_settings.ctrl = runtime->is_key_down(VK_CONTROL);
                    g_settings.shift = runtime->is_key_down(VK_SHIFT);
                    g_settings.alt = runtime->is_key_down(VK_MENU);
                    g_listening = false;
                    save_settings(runtime);
                    break;
                }
            }
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Avoid keys the game or ReShade already use.");

    ImGui::Spacing();
    if (ImGui::Checkbox("Show the capturing / complete banner", &g_settings.banner))
        save_settings(runtime);
    if (ImGui::Checkbox("Show depth preview", &g_settings.preview))
        save_settings(runtime);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("A small window with exactly the depth a capture will save. If it shows the "
                          "scene, captures will have depth. It is never in the saved image.");
    if (g_settings.preview)
    {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(160.0f);
        float percent = g_settings.preview_size * 100.0f;
        if (ImGui::SliderFloat("Size##preview", &percent, 10.0f, 90.0f, "%.0f%%", 0))
        {
            g_settings.preview_size = percent / 100.0f;
            save_settings(runtime);
        }
    }

    ImGui::Separator();
    ImGui::Text("Captured this session: %d", g_count.load());
    if (g_busy)
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "Saving...");
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        reshade::register_event<reshade::addon_event::reshade_finish_effects>(on_finish_effects);
        reshade::register_event<reshade::addon_event::reshade_overlay>(on_overlay);
        reshade::register_overlay(nullptr, on_settings);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_overlay(nullptr, on_settings);
        reshade::unregister_addon(module);
        break;
    }
    return TRUE;
}
