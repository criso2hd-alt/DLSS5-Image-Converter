// DLSS5Capture.fx - companion to the "DLSS5 Scene Capture" add-on.
//
// Copies the depth buffer ReShade has picked for this game, raw, into a 32-bit
// float texture every frame, so the add-on can read it back when a capture is
// taken. It draws nothing to the screen.
//
// It also keeps a small viewable version (DLSS5_DepthView) for the add-on's
// built-in preview window. That preview shows the very texture the add-on
// saves, so if the preview shows depth, the capture will have depth.
// ReShade's DisplayDepth reads ReShade's depth directly and needs the right
// preprocessor settings, which is how it could look fine while captures came
// back empty.
//
// If the depth is wrong (flat, or the wrong buffer), tune ReShade's Add-ons
// tab, "Generic Depth". The saved value is raw; it is interpreted on the
// converter's side.

texture DepthBufferTex : DEPTH;
sampler DepthSampler
{
    Texture = DepthBufferTex;
    // Point sampling: blending across a depth edge would invent a surface
    // halfway between a face and the wall behind it.
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

texture DLSS5_DepthCopy
{
    Width = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = R32F;
};
sampler DepthCopySampler
{
    Texture = DLSS5_DepthCopy;
    MagFilter = POINT;
    MinFilter = POINT;
    MipFilter = POINT;
};

// 1 when this game stores reversed Z (near = 1, sky = 0), 0 for standard Z.
texture DLSS5_Reversed { Width = 1; Height = 1; Format = R8; };
sampler ReversedSampler { Texture = DLSS5_Reversed; MagFilter = POINT; MinFilter = POINT; MipFilter = POINT; };

// Half resolution is plenty for a preview window.
texture DLSS5_DepthView { Width = BUFFER_WIDTH / 2; Height = BUFFER_HEIGHT / 2; Format = RGBA8; };

void VS_Fullscreen(in uint id : SV_VertexID, out float4 position : SV_Position,
                   out float2 uv : TEXCOORD)
{
    uv.x = (id == 2) ? 2.0 : 0.0;
    uv.y = (id == 1) ? 2.0 : 0.0;
    position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}

float PS_CopyDepth(float4 position : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    return tex2Dlod(DepthSampler, float4(uv, 0.0, 0.0)).x;
}

// Decide reversed vs standard from a grid of samples. Sky and far geometry
// pile up at 0 in reversed Z and at 1 in standard Z; with no sky in view, the
// bulk of an ordinary scene still sits near 0 in reversed and near 1 in
// standard, so the mean breaks the tie.
float PS_DetectReversed(float4 position : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float at_zero = 0.0, at_one = 0.0, total = 0.0;
    [loop] for (int y = 0; y < 16; ++y)
    {
        [loop] for (int x = 0; x < 16; ++x)
        {
            const float d = tex2Dlod(DepthCopySampler, float4((x + 0.5) / 16.0, (y + 0.5) / 16.0, 0.0, 0.0)).x;
            at_zero += d <= 1e-7;
            at_one += d >= 1.0 - 1e-7;
            total += d;
        }
    }
    if (abs(at_zero - at_one) >= 2.0)
        return at_zero > at_one ? 1.0 : 0.0;
    return total / 256.0 < 0.5 ? 1.0 : 0.0;
}

// Near is white, far is dark, sky is dark blue. Both depth conventions store
// roughly near / distance as their "closeness" (reversed directly, standard as
// 1 - d), so one log scale over five decades shows any scene with contrast.
float4 PS_DepthView(float4 position : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    const float d = tex2Dlod(DepthCopySampler, float4(uv, 0.0, 0.0)).x;
    const bool reversed = tex2Dlod(ReversedSampler, float4(0.5, 0.5, 0.0, 0.0)).x > 0.5;
    const float closeness = reversed ? d : 1.0 - d;
    if (closeness <= 1e-7)
        return float4(0.05, 0.08, 0.22, 1.0);
    const float v = saturate(1.0 + log10(closeness) / 5.0);
    return float4(v, v, v, 1.0);
}

technique DLSS5_DepthCapture <
    ui_label = "DLSS5 Depth Capture";
    ui_tooltip = "Keeps a copy of the game's depth so the DLSS5 Scene Capture add-on can save it "
                 "with every capture, and feeds its preview window. Draws nothing on screen.";
    enabled = true;
>
{
    pass Copy
    {
        VertexShader = VS_Fullscreen;
        PixelShader = PS_CopyDepth;
        RenderTarget = DLSS5_DepthCopy;
    }
    pass Detect
    {
        VertexShader = VS_Fullscreen;
        PixelShader = PS_DetectReversed;
        RenderTarget = DLSS5_Reversed;
    }
    pass View
    {
        VertexShader = VS_Fullscreen;
        PixelShader = PS_DepthView;
        RenderTarget = DLSS5_DepthView;
    }
}
