/***************************************************************************
 # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#pragma pack_matrix(row_major)

#include "RtxdiApplicationBridge.hlsli"

#include <rtxdi/GIResamplingFunctions.hlsli>


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
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams);

    if (any(pixelPosition > int2(g_Const.view.viewportSize)))
        return;

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(GlobalIndex, 8);
    
    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);
    
    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.runtimeParams, reservoirPosition, g_Const.spatialInputBufferIndex);

    if (RAB_IsSurfaceValid(primarySurface)) {
        RTXDI_GISpatialResamplingParameters sparams;

        sparams.sourceBufferIndex = g_Const.spatialInputBufferIndex;
        sparams.biasCorrectionMode = g_Const.spatialBiasCorrection;
        sparams.depthThreshold = g_Const.spatialDepthThreshold;
        sparams.normalThreshold = g_Const.spatialNormalThreshold; 
        sparams.numSamples = g_Const.numSpatialSamples;
        sparams.samplingRadius = g_Const.spatialSamplingRadius;

        // Execute resampling.
        reservoir = RTXDI_GISpatialResampling(pixelPosition, primarySurface, reservoir, rng, sparams, g_Const.runtimeParams);
    }

    RTXDI_StoreGIReservoir(reservoir, g_Const.runtimeParams, reservoirPosition, g_Const.spatialOutputBufferIndex);
}
