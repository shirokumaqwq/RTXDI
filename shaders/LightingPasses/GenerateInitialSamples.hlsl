/***************************************************************************
 # Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#pragma pack_matrix(row_major)

#include "RtxdiApplicationBridge.hlsli"

#include <rtxdi/ResamplingFunctions.hlsli>

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif

    const RTXDI_ResamplingRuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 1);
    RAB_RandomSamplerState tileRng = RAB_InitRandomSampler(pixelPosition / RTXDI_TILE_SIZE_IN_PIXELS, 1);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
        g_Const.numPrimaryRegirSamples,
        g_Const.numPrimaryLocalLightSamples,
        g_Const.numPrimaryInfiniteLightSamples,
        g_Const.numPrimaryEnvironmentSamples,
        g_Const.numPrimaryBrdfSamples,
        g_Const.brdfCutoff,
        0.001f);

    RAB_LightSample lightSample;
    RTXDI_Reservoir reservoir = RTXDI_SampleLightsForSurface(rng, tileRng, surface,
        sampleParams, params, lightSample);

    if (g_Const.enableInitialVisibility && RTXDI_IsValidReservoir(reservoir))
    {
        if (!RAB_GetConservativeVisibility(surface, lightSample))
        {
            RTXDI_StoreVisibilityInReservoir(reservoir, 0, true);
        }
    }

    RAB_SplitRadiance splitRadiance = RAB_GetLightSampleSplitRadianceForSurface(lightSample, surface);

    if(g_Const.colorDenoiserMode == 1)
    {
        reservoir.colorWeight = RTXDI_GetReservoirInvPdf(reservoir) * splitRadiance.diffuse;
    }
    else if(g_Const.colorDenoiserMode == 2)
    {
        reservoir.colorWeight =  RTXDI_GetReservoirInvPdf(reservoir) * (splitRadiance.diffuse * surface.diffuseAlbedo + splitRadiance.specular);
    }

    RTXDI_StoreReservoir(reservoir, params, GlobalIndex, g_Const.initialOutputBufferIndex);
}