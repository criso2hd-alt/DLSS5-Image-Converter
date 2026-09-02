// dlss5_eval — evaluate DLSS (and, via the RenoDX add-on, the DLSS 5 neural
// renderer) over a synthetic single-image frame contract.
//
// Why this exists as a separate process
// -------------------------------------
// DLSS 5's neural pass is not an image API. `nvngx_dlssnr.dll` is an NGX snippet
// that the RenoDX ReShade add-on injects into a DLSS Super Resolution
// evaluation. So the only supported way to reach it is to *be* something that
// evaluates DLSS. This program is the smallest possible such thing: a D3D12
// device, a hidden swapchain so ReShade attaches and loads the add-on, and one
// DLSS feature in DLAA mode that we feed the same still image several times.
//
// The feature handle is created once and reused. That is the point of the whole
// process model: DLSS's temporal history lives inside that handle, and a
// one-shot-per-frame design would reset the accumulator on every pass and make
// multi-frame evaluation pointless.
//
// Protocol (stdin/stdout, one line each way, see evaluator.py):
//   <- READY <notes>
//   -> FRAME <colour.bin> <jitter_x> <jitter_y> <reset 0|1>
//   <- FRAME_OK <index>
//   -> WRITE <out.bin>
//   <- WRITE_OK <bytes>
//   -> QUIT
//   <- BYE

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"

using Microsoft::WRL::ComPtr;

namespace {

// A project-specific NGX application id. NVIDIA hands these out for shipping
// titles; for local tooling any stable non-zero value works, and keeping it
// stable matters because NGX caches per-app state between runs.
constexpr unsigned long long kAppId = 0x44'4C'53'35'49'4D'47'01ull;

// PCI vendor id for NVIDIA, used to choose the adapter deliberately rather than
// taking whichever one Windows enumerates first.
constexpr UINT kNvidiaVendorId = 0x10DE;

struct Options {
    int width = 0;
    int height = 0;
    int frames = 1;
    std::wstring depth_path;
    std::wstring motion_path;
    bool reversed_depth = false;
    float intensity = 0.65f;
    float skin = 0.45f;
    float local_tone = 0.40f;
    float structure = 0.50f;
    bool probe = false;
};

void Emit(const std::string& line) {
    std::cout << line << std::endl;  // endl: the parent blocks on readline().
    std::cout.flush();
}

[[noreturn]] void Fail(const std::string& message) {
    Emit("ERROR " + message);
    std::exit(1);
}

void Require(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::ostringstream out;
        out << what << " failed (0x" << std::hex << static_cast<unsigned>(hr) << ")";
        Fail(out.str());
    }
}

void RequireNgx(NVSDK_NGX_Result result, const char* what) {
    if (NVSDK_NGX_FAILED(result)) {
        std::ostringstream out;
        out << what << " failed: ";
        // GetNGXResultAsString returns wide text; narrow it crudely, it is ASCII.
        const wchar_t* text = GetNGXResultAsString(result);
        for (const wchar_t* p = text; p && *p; ++p) out << static_cast<char>(*p);
        Fail(out.str());
    }
}

std::wstring Widen(const std::string& text) {
    if (text.empty()) return {};
    int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), wide.data(), size);
    return wide;
}

std::string Narrow(const std::wstring& text) {
    if (text.empty()) return {};
    int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()),
                                   nullptr, 0, nullptr, nullptr);
    std::string narrow(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), narrow.data(),
                        size, nullptr, nullptr);
    return narrow;
}

std::string Trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

// Peel `count` whitespace-separated fields off the *end* of a line, leaving
// whatever precedes them as `head`.
//
// Commands carry a path followed by fixed numeric fields, and the path is the
// part that can contain spaces — "C:\Program Files\...", or the release folder
// of anyone whose user name has a space in it. Reading the path with `>>` takes
// only the text before its first space and then fails to open a file whose name
// is a truncation of the real one, which reads as a corrupt contract rather
// than as a quoting bug. Counting from the end is unambiguous instead.
bool SplitTrailingFields(const std::string& text, int count, std::string& head,
                         std::vector<std::string>& tail) {
    tail.clear();
    std::string rest = Trim(text);
    for (int i = 0; i < count; ++i) {
        const auto pos = rest.find_last_of(" \t");
        if (pos == std::string::npos) return false;
        tail.insert(tail.begin(), rest.substr(pos + 1));
        rest = Trim(rest.substr(0, pos));
    }
    head = rest;
    return !head.empty();
}

std::vector<uint8_t> ReadFile(const std::wstring& path, size_t expected) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) Fail("Could not open a contract plane for reading: " + Narrow(path));
    const auto size = static_cast<size_t>(file.tellg());
    if (size != expected) {
        std::ostringstream out;
        out << "Contract plane is " << size << " bytes, expected " << expected
            << ". The harness and the contract disagree about the frame size.";
        Fail(out.str());
    }
    std::vector<uint8_t> data(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

// --- D3D12 plumbing ---------------------------------------------------------

struct Texture {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Resource> upload;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT bytes_per_pixel = 0;
};

class Harness {
public:
    void Initialise(const Options& options);
    void UploadColour(const std::wstring& path);
    void Evaluate(float jitter_x, float jitter_y, bool reset);
    size_t WriteOutput(const std::wstring& path);
    void ProbeWarmUp();
    std::string Probe();
    void Shutdown();

private:
    Texture CreateTexture(DXGI_FORMAT format, UINT bytes_per_pixel, D3D12_RESOURCE_FLAGS flags,
                          D3D12_RESOURCE_STATES state);
    void UploadTexture(Texture& texture, const std::vector<uint8_t>& data);
    void Execute();
    void CreateHiddenWindow();

    Options options_{};
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<IDXGISwapChain1> swapchain_;
    HANDLE fence_event_ = nullptr;
    UINT64 fence_value_ = 0;
    HWND window_ = nullptr;

    Texture colour_, depth_, motion_, output_;
    ComPtr<ID3D12Resource> readback_;
    UINT readback_row_pitch_ = 0;

    //: The adapter the device was actually created on, so diagnostics report
    //: the GPU in use rather than the one that happened to enumerate first.
    std::wstring adapter_description_;
    bool adapter_is_nvidia_ = false;
    std::string probe_note_;

    NVSDK_NGX_Parameter* params_ = nullptr;
    NVSDK_NGX_Handle* feature_ = nullptr;
    bool ngx_ready_ = false;
    int frame_index_ = 0;
};

void Harness::CreateHiddenWindow() {
    // ReShade attaches through a proxy DXGI/D3D12 DLL and only starts once a
    // swapchain exists. No swapchain means no add-on, which means a plain DLAA
    // resolve and no neural pass — the failure mode is a subtly disappointing
    // image rather than an error, so the window is not optional.
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"Dlss5EvalHost";
    RegisterClassExW(&wc);
    window_ = CreateWindowExW(0, wc.lpszClassName, L"dlss5_eval", WS_OVERLAPPEDWINDOW, 0, 0,
                              64, 64, nullptr, nullptr, wc.hInstance, nullptr);
    if (!window_) Fail("Could not create the host window ReShade needs to attach to.");
    // Deliberately never shown. It exists so the swapchain is legal, not to be
    // looked at; SW_HIDE keeps it off the taskbar and out of the user's way.
    ShowWindow(window_, SW_HIDE);
}

Texture Harness::CreateTexture(DXGI_FORMAT format, UINT bytes_per_pixel,
                               D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    Texture texture;
    texture.format = format;
    texture.bytes_per_pixel = bytes_per_pixel;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = static_cast<UINT64>(options_.width);
    desc.Height = static_cast<UINT>(options_.height);
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;

    Require(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                            IID_PPV_ARGS(&texture.resource)),
            "CreateCommittedResource(texture)");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);

    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.Format = DXGI_FORMAT_UNKNOWN;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Require(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                            IID_PPV_ARGS(&texture.upload)),
            "CreateCommittedResource(upload)");
    return texture;
}

void Harness::UploadTexture(Texture& texture, const std::vector<uint8_t>& data) {
    D3D12_RESOURCE_DESC desc = texture.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);

    uint8_t* mapped = nullptr;
    D3D12_RANGE none{0, 0};
    Require(texture.upload->Map(0, &none, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    // Row by row: D3D12 pads every row to 256 bytes, so a straight memcpy of a
    // tightly packed source shears the image diagonally.
    for (UINT y = 0; y < rows; ++y) {
        std::memcpy(mapped + footprint.Footprint.RowPitch * y,
                    data.data() + row_bytes * y, static_cast<size_t>(row_bytes));
    }
    texture.upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = texture.resource.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = texture.upload.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;

    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = texture.resource.Get();
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list_->ResourceBarrier(1, &to_copy);
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
    list_->ResourceBarrier(1, &to_copy);
}

void Harness::Execute() {
    Require(list_->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    const UINT64 target = ++fence_value_;
    Require(queue_->Signal(fence_.Get(), target), "Queue::Signal");
    if (fence_->GetCompletedValue() < target) {
        Require(fence_->SetEventOnCompletion(target, fence_event_), "SetEventOnCompletion");
        WaitForSingleObject(fence_event_, INFINITE);
    }
    Require(allocator_->Reset(), "Allocator::Reset");
    Require(list_->Reset(allocator_.Get(), nullptr), "CommandList::Reset");
}

void Harness::Initialise(const Options& options) {
    options_ = options;

    ComPtr<IDXGIFactory4> factory;
    Require(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    // The default adapter, deliberately.
    //
    // Selecting the NVIDIA adapter explicitly looks like the obvious fix for a
    // machine that has an integrated GPU as adapter 0 - and it was tried here -
    // but passing any explicit adapter to D3D12CreateDevice makes NGX refuse the
    // resulting device with FAIL_FeatureNotSupported, on a card that works
    // perfectly through the default path. Verified with and without ReShade
    // loaded, so it is not the proxy wrapping the factory.
    //
    // So the device stays on the default adapter, and the LUID lookup below only
    // *reports* which GPU that turned out to be. If it is not the one the user
    // expects, the fix is Windows' own per-application setting
    // (Settings > Display > Graphics), which changes what "default" means.
    Require(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device_)),
            "D3D12CreateDevice");

    // Name the adapter the device actually landed on. Reporting whichever one
    // enumerates first would happily print an RTX name while the device ran on
    // an integrated GPU, which is the exact confusion this exists to prevent.
    {
        const LUID luid = device_->GetAdapterLuid();
        ComPtr<IDXGIAdapter1> in_use;
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&in_use))) &&
            SUCCEEDED(in_use->GetDesc1(&desc))) {
            adapter_description_ = desc.Description;
            adapter_is_nvidia_ = desc.VendorId == kNvidiaVendorId;
        }
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Require(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
    Require(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
            "CreateCommandAllocator");
    Require(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                       IID_PPV_ARGS(&list_)),
            "CreateCommandList");
    Require(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    CreateHiddenWindow();
    DXGI_SWAP_CHAIN_DESC1 sc{};
    sc.Width = 64;
    sc.Height = 64;
    sc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.BufferCount = 2;
    sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc.SampleDesc.Count = 1;
    // A failure here is survivable: without ReShade we still get a DLAA resolve,
    // which is a useful diagnostic rather than a reason to abort.
    factory->CreateSwapChainForHwnd(queue_.Get(), window_, &sc, nullptr, nullptr, &swapchain_);

    const NVSDK_NGX_Result init = NVSDK_NGX_D3D12_Init(kAppId, L".", device_.Get());
    if (NVSDK_NGX_FAILED(init)) {
        // "FeatureNotSupported" reads as "your GPU is too old", and sometimes is
        // — but it is also what comes back when the device was created on an
        // integrated GPU. Naming the adapter turns an opaque NGX code into
        // something the user can act on.
        std::ostringstream why;
        const wchar_t* raw = GetNGXResultAsString(init);
        why << "NGX could not start on \"" << Narrow(adapter_description_) << "\" ("
            << Narrow(raw ? raw : L"") << "). ";
        if (init == NVSDK_NGX_Result_FAIL_FeatureNotSupported) {
            if (!adapter_is_nvidia_) {
                why << "That is not an NVIDIA GPU, so DLSS cannot run on it. The "
                       "app is using this machine's default graphics adapter. "
                       "Force it onto the NVIDIA card in Windows: Settings > "
                       "System > Display > Graphics, add DLSS5Converter.exe and "
                       "engine\dlss5_eval.exe, and set both to High performance.";
            } else {
                why << "This NVIDIA GPU does not support DLSS. It needs an RTX "
                       "card - GTX 10 and 16-series have CUDA but no tensor "
                       "cores, so depth estimation works and DLSS cannot.";
            }
        } else if (init == NVSDK_NGX_Result_FAIL_OutOfDate) {
            why << "The driver is too old for this DLSS runtime. Update it.";
        } else {
            const wchar_t* text = GetNGXResultAsString(init);
            why << Narrow(text ? text : L"");
        }
        Fail(why.str());
    }
    ngx_ready_ = true;
    RequireNgx(NVSDK_NGX_D3D12_GetCapabilityParameters(&params_),
               "NVSDK_NGX_D3D12_GetCapabilityParameters");

    // --probe returns here deliberately, *before* the availability check below.
    // Diagnosing an unavailable runtime is the entire job of --probe, so it has
    // to live long enough to print "dlss_available: 0"; aborting with an ERROR
    // line instead tells the user nothing about which piece is missing.
    if (options_.probe) return;

    int available = 0;
    params_->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
    if (!available) {
        Fail("NGX reports DLSS Super Resolution as unavailable, so there is no "
             "evaluation for the neural pass to ride on. Usually nvngx_dlss.dll "
             "is not beside this executable yet. Run --probe for the breakdown.");
    }

    colour_ = CreateTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    depth_ = CreateTexture(DXGI_FORMAT_R32_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    motion_ = CreateTexture(DXGI_FORMAT_R16G16_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    output_ = CreateTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const size_t pixels = static_cast<size_t>(options_.width) * options_.height;
    UploadTexture(depth_, ReadFile(options_.depth_path, pixels * 4));
    UploadTexture(motion_, ReadFile(options_.motion_path, pixels * 4));
    Execute();

    NVSDK_NGX_DLSS_Create_Params create{};
    create.Feature.InWidth = static_cast<unsigned>(options_.width);
    create.Feature.InHeight = static_cast<unsigned>(options_.height);
    // DLAA: render resolution equals output resolution. There is nothing to
    // upscale here — we want the neural pass, not more pixels.
    create.Feature.InTargetWidth = static_cast<unsigned>(options_.width);
    create.Feature.InTargetHeight = static_cast<unsigned>(options_.height);
    create.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
    create.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
        // Our colour plane is linear with no tonemap applied, so let DLSS derive
        // its own exposure rather than trusting a value we would have to invent.
        NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
        (options_.reversed_depth ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0);

    RequireNgx(NGX_D3D12_CREATE_DLSS_EXT(list_.Get(), 1, 1, &feature_, params_, &create),
               "NGX_D3D12_CREATE_DLSS_EXT");
    Execute();

    // The neural add-on's knobs used to be set here as NGX parameters, and
    // again in main() as environment variables, because there was no public
    // DLSS 5 header and one of the two was assumed to be live. Measurement said
    // neither is: the add-on reads ReShade.ini, once, at startup. Both dead
    // paths are gone, and runtime.write_addon_config owns the settings now.
    // The --intensity family of flags is kept only so the harness stays
    // runnable by hand with the same arguments the pipeline has always passed.
}

void Harness::UploadColour(const std::wstring& path) {
    const size_t pixels = static_cast<size_t>(options_.width) * options_.height;
    UploadTexture(colour_, ReadFile(path, pixels * 8));
}

void Harness::Evaluate(float jitter_x, float jitter_y, bool reset) {
    NVSDK_NGX_D3D12_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = colour_.resource.Get();
    eval.Feature.pInOutput = output_.resource.Get();
    eval.pInDepth = depth_.resource.Get();
    eval.pInMotionVectors = motion_.resource.Get();
    // Sign convention matches DLSS5_Feed.fx: the offset describes where the
    // sample sits relative to the pixel centre, and we resampled the colour
    // plane by the same amount before handing it over.
    eval.InJitterOffsetX = jitter_x;
    eval.InJitterOffsetY = jitter_y;
    eval.InRenderSubrectDimensions.Width = static_cast<unsigned>(options_.width);
    eval.InRenderSubrectDimensions.Height = static_cast<unsigned>(options_.height);
    // Only the first pass resets. Every later pass is the accumulation we are
    // running multiple frames to build, so resetting again would throw it away.
    eval.InReset = reset ? 1 : 0;
    // Motion vectors are already in pixels, so no further scaling.
    eval.InMVScaleX = 1.0f;
    eval.InMVScaleY = 1.0f;

    RequireNgx(NGX_D3D12_EVALUATE_DLSS_EXT(list_.Get(), feature_, params_, &eval),
               "NGX_D3D12_EVALUATE_DLSS_EXT");
    Execute();

    // Present so ReShade advances a frame. The add-on does its work around the
    // evaluation, but a runtime that never presents looks hung to it and some
    // builds defer their first-frame setup until after the initial Present.
    if (swapchain_) swapchain_->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    ++frame_index_;
}

size_t Harness::WriteOutput(const std::wstring& path) {
    D3D12_RESOURCE_DESC desc = output_.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);

    if (!readback_) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.Format = DXGI_FORMAT_UNKNOWN;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Require(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback_)),
                "CreateCommittedResource(readback)");
    }

    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = output_.resource.Get();
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list_->ResourceBarrier(1, &to_copy);

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = output_.resource.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback_.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    std::swap(to_copy.Transition.StateBefore, to_copy.Transition.StateAfter);
    list_->ResourceBarrier(1, &to_copy);
    Execute();

    uint8_t* mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(total)};
    Require(readback_->Map(0, &range, reinterpret_cast<void**>(&mapped)), "Map(readback)");
    std::ofstream file(path, std::ios::binary);
    if (!file) Fail("Could not open the output file for writing.");
    // Strip the 256-byte row padding on the way out so Python can reshape the
    // buffer directly instead of having to know D3D12's alignment rules.
    for (UINT y = 0; y < rows; ++y) {
        file.write(reinterpret_cast<const char*>(mapped + footprint.Footprint.RowPitch * y),
                   static_cast<std::streamsize>(row_bytes));
    }
    file.close();
    D3D12_RANGE nothing{0, 0};
    readback_->Unmap(0, &nothing);
    return static_cast<size_t>(row_bytes) * rows;
}

void Harness::ProbeWarmUp() {
    // Run a real, tiny evaluation before reporting module state.
    //
    // Without this the probe is not just incomplete, it is misleading. The
    // add-on pre-loads nvngx_dlssnr.dll at device init on some setups and
    // defers to "retry lazily on first evaluate" on others — and a probe that
    // returns before creating a feature never reaches the lazy path, so it
    // reports dlssnr_module_loaded: 0 on a machine where a conversion would
    // have worked perfectly.
    //
    // Two frames, not one: the add-on installs its NGX hooks during the first
    // evaluate, so that one cannot be intercepted.
    options_.width = 64;
    options_.height = 64;
    const size_t pixels = 64 * 64;

    colour_ = CreateTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    depth_ = CreateTexture(DXGI_FORMAT_R32_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    motion_ = CreateTexture(DXGI_FORMAT_R16G16_FLOAT, 4, D3D12_RESOURCE_FLAG_NONE,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    output_ = CreateTexture(DXGI_FORMAT_R16G16B16A16_FLOAT, 8,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    UploadTexture(colour_, std::vector<uint8_t>(pixels * 8, 0));
    UploadTexture(depth_, std::vector<uint8_t>(pixels * 4, 0));
    UploadTexture(motion_, std::vector<uint8_t>(pixels * 4, 0));
    Execute();

    NVSDK_NGX_DLSS_Create_Params create{};
    create.Feature.InWidth = 64;
    create.Feature.InHeight = 64;
    create.Feature.InTargetWidth = 64;
    create.Feature.InTargetHeight = 64;
    create.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
    create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                  NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                                  NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;

    const NVSDK_NGX_Result created =
        NGX_D3D12_CREATE_DLSS_EXT(list_.Get(), 1, 1, &feature_, params_, &create);
    if (NVSDK_NGX_FAILED(created)) {
        const wchar_t* text = GetNGXResultAsString(created);
        probe_note_ = "feature creation failed: " + Narrow(text ? text : L"");
        return;
    }
    Execute();

    for (int i = 0; i < 2; ++i) {
        NVSDK_NGX_D3D12_DLSS_Eval_Params eval{};
        eval.Feature.pInColor = colour_.resource.Get();
        eval.Feature.pInOutput = output_.resource.Get();
        eval.pInDepth = depth_.resource.Get();
        eval.pInMotionVectors = motion_.resource.Get();
        eval.InJitterOffsetX = 0.0f;
        eval.InJitterOffsetY = 0.0f;
        eval.InRenderSubrectDimensions.Width = 64;
        eval.InRenderSubrectDimensions.Height = 64;
        eval.InReset = i == 0 ? 1 : 0;
        eval.InMVScaleX = 1.0f;
        eval.InMVScaleY = 1.0f;
        const NVSDK_NGX_Result evaluated =
            NGX_D3D12_EVALUATE_DLSS_EXT(list_.Get(), feature_, params_, &eval);
        if (NVSDK_NGX_FAILED(evaluated)) {
            const wchar_t* text = GetNGXResultAsString(evaluated);
            probe_note_ = "evaluation failed: " + Narrow(text ? text : L"");
            return;
        }
        Execute();
        if (swapchain_) swapchain_->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    }
    probe_note_ = "ok";
}

std::string Harness::Probe() {
    std::ostringstream out;
    // The adapter the device was actually created on. Reporting whichever
    // adapter enumerates first would happily print an RTX name while the
    // device ran on an integrated GPU, which is the exact confusion this
    // line exists to prevent.
    out << "adapter: " << Narrow(adapter_description_) << "\n";

    int value = 0;
    params_->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &value);
    out << "dlss_available: " << value << "\n";
    value = 0;
    params_->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &value);
    out << "needs_driver_update: " << value << "\n";

    // Whether the neural add-on attached at all. If this stays 0 the pipeline
    // still runs and still produces a picture — a plain DLAA resolve — which is
    // the single most confusing failure this tool has, so it is checked first.
    //
    // Asked the honest way: the add-on is a ReShade module, not an NGX one, and
    // it publishes no NGX parameter. This used to read a guessed parameter name
    // ("DLSS5.NeuralRendering.Available") that nothing anywhere sets, so it
    // reported 0 even with the add-on demonstrably loaded and running.
    out << "neural_addon_loaded: "
        << (GetModuleHandleW(L"renodx-dlss5.addon64") ? 1 : 0) << "\n";

    // ReShade ships as a proxy named after whichever DLL it stands in for —
    // dxgi.dll for us — so nothing in the process is ever called ReShade64.dll,
    // and looking for that name reported 0 unconditionally. Probing for one of
    // its add-on API exports identifies it whatever the file was renamed to.
    const HMODULE proxy = GetModuleHandleW(L"dxgi.dll");
    out << "reshade_proxy_loaded: "
        << (proxy && GetProcAddress(proxy, "ReShadeRegisterAddon") ? 1 : 0) << "\n";
    out << "dlssnr_module_loaded: " << (GetModuleHandleW(L"nvngx_dlssnr.dll") ? 1 : 0) << "\n";
    // The line above only means anything because ProbeWarmUp ran a real
    // evaluation first; this says whether that evaluation actually worked.
    if (!probe_note_.empty()) out << "test_evaluation: " << probe_note_ << "\n";
    return out.str();
}

void Harness::Shutdown() {
    if (feature_) {
        NVSDK_NGX_D3D12_ReleaseFeature(feature_);
        feature_ = nullptr;
    }
    if (ngx_ready_) {
        NVSDK_NGX_D3D12_DestroyParameters(params_);
        NVSDK_NGX_D3D12_Shutdown1(device_.Get());
        ngx_ready_ = false;
    }
    if (fence_event_) CloseHandle(fence_event_);
    if (window_) DestroyWindow(window_);
}

Options Parse(int argc, char** argv) {
    Options options;
    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) Fail("Missing value for " + std::string(argv[i]));
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--width") options.width = std::stoi(next(i));
        else if (flag == "--height") options.height = std::stoi(next(i));
        else if (flag == "--frames") options.frames = std::stoi(next(i));
        else if (flag == "--depth") options.depth_path = Widen(next(i));
        else if (flag == "--motion") options.motion_path = Widen(next(i));
        else if (flag == "--reversed-depth") options.reversed_depth = true;
        else if (flag == "--intensity") options.intensity = std::stof(next(i));
        else if (flag == "--skin") options.skin = std::stof(next(i));
        else if (flag == "--local-tone") options.local_tone = std::stof(next(i));
        else if (flag == "--structure") options.structure = std::stof(next(i));
        else if (flag == "--probe") options.probe = true;
        else Fail("Unknown argument: " + flag);
    }
    if (!options.probe && (options.width <= 0 || options.height <= 0)) {
        Fail("--width and --height are required.");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    Options options = Parse(argc, argv);

    Harness harness;
    harness.Initialise(options);

    if (options.probe) {
        harness.ProbeWarmUp();
        std::cout << harness.Probe();
        std::cout.flush();
        harness.Shutdown();
        return 0;
    }

    Emit("READY DLSS feature created in DLAA mode");

    std::string line;
    int evaluated = 0;
    while (std::getline(std::cin, line)) {
        std::istringstream parts(line);
        std::string command;
        parts >> command;
        if (command == "FRAME") {
            std::string rest;
            std::getline(parts, rest);
            std::string path;
            std::vector<std::string> fields;
            if (!SplitTrailingFields(rest, 3, path, fields)) {
                Fail("Malformed FRAME: expected <colour path> <jitter x> <jitter y> <reset>.");
            }
            const float jx = std::stof(fields[0]);
            const float jy = std::stof(fields[1]);
            const int reset = std::stoi(fields[2]);
            harness.UploadColour(Widen(path));
            harness.Evaluate(jx, jy, reset != 0);
            Emit("FRAME_OK " + std::to_string(++evaluated));
        } else if (command == "WRITE") {
            std::string rest;
            std::getline(parts, rest);
            // The whole remainder is the path — nothing follows it.
            const std::string path = Trim(rest);
            if (path.empty()) Fail("Malformed WRITE: expected an output path.");
            Emit("WRITE_OK " + std::to_string(harness.WriteOutput(Widen(path))));
        } else if (command == "QUIT" || command.empty()) {
            break;
        } else {
            Fail("Unknown command: " + command);
        }
    }

    harness.Shutdown();
    Emit("BYE");
    return 0;
}
