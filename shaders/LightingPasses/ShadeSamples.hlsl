/***************************************************************************
 # Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef WITH_NRD
#define NRD_HEADER_ONLY
#include <NRD.hlsli>
#endif

#include "ShadingHelpers.hlsli"

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif

    const RTXDI_ResamplingRuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_Reservoir reservoir = RTXDI_LoadReservoir(params, GlobalIndex, g_Const.shadeInputBufferIndex);

    float3 diffuse = 0;
    float3 specular = 0;
    float lightDistance = 0;
    float2 currLuminance = 0;
    float3 visibilityFactor = float3(1.0, 1.0, 1.0);

    // TODO: use external parameter
    int MultiSampleSize = g_Const.colorDenoiserMode == 4 ? 3 : 1;
    int MultiSampleRadius = MultiSampleSize / 2;
    const float normalThreshold = 0.6f;
    const float depthThreshold = 0.1f;
    const bool enableMaterialSimilarityTest = true;

    int ValidNeghborNum = 0;
    for(int i = - MultiSampleRadius; i <= MultiSampleRadius; i++)
    {
        for(int j = - MultiSampleRadius; j <= MultiSampleRadius; j++)
        {
            int2 neighborPixelPosition = int2(pixelPosition) + int2(i, j);
            neighborPixelPosition = RAB_ClampSamplePositionIntoView(neighborPixelPosition, false);

            RAB_Surface neighborSurface = RAB_GetGBufferSurface(neighborPixelPosition, false);

            if (!RAB_IsSurfaceValid(neighborSurface))
                continue;

            if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(neighborSurface), 
                RAB_GetSurfaceLinearDepth(surface), RAB_GetSurfaceLinearDepth(neighborSurface), 
                normalThreshold, depthThreshold)) // TODO: use external parameter
                continue;

            if (enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(surface, neighborSurface)) // TODO: use external parameter
                continue;

            uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(neighborPixelPosition, params);
            RTXDI_Reservoir neighborReservoir = RTXDI_LoadReservoir(params, neighborReservoirPos, g_Const.shadeInputBufferIndex);

            if(RTXDI_IsValidReservoir(neighborReservoir))
            {
                ValidNeghborNum++;
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(neighborReservoir), false);

                RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo,
                    surface, RTXDI_GetReservoirSampleUV(neighborReservoir));

                float3 neighborDiffuse = 0;
                float3 neighborSpecular = 0;
                float neighborLightDistance = 0;    
                float3 neighborVisibilityFactor = float3(1.0, 1.0, 1.0);


                bool needToStore = ShadeSurfaceWithLightSample(neighborReservoir, surface, lightSample,
                    /* previousFrameTLAS = */ false, /* enableVisibilityReuse = */ true, neighborDiffuse, neighborSpecular, neighborLightDistance, neighborVisibilityFactor);

                if(i == 0 && j == 0)
                {
                    currLuminance = float2(calcLuminance(neighborDiffuse * surface.diffuseAlbedo), calcLuminance(neighborSpecular));
                    lightDistance = neighborLightDistance;
                    visibilityFactor = neighborVisibilityFactor;

                    if (needToStore)
                    {   
                        RTXDI_StoreReservoir(neighborReservoir, params, GlobalIndex, g_Const.shadeInputBufferIndex);
                    }                    
                }

                neighborSpecular = DemodulateSpecular(surface.specularF0, neighborSpecular);

                diffuse += neighborDiffuse;
                specular += neighborSpecular;
            }
        }
    }
    
    diffuse = ValidNeghborNum == 0 ? 0 : diffuse / ValidNeghborNum;
    specular = ValidNeghborNum == 0 ? 0 : specular / ValidNeghborNum;

    // if (RTXDI_IsValidReservoir(reservoir))
    // {
    //     RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(reservoir), false);

    //     RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo,
    //         surface, RTXDI_GetReservoirSampleUV(reservoir));

    //     bool needToStore = ShadeSurfaceWithLightSample(reservoir, surface, lightSample,
    //         /* previousFrameTLAS = */ false, /* enableVisibilityReuse = */ true, diffuse, specular, lightDistance, visibilityFactor);
    
    //     currLuminance = float2(calcLuminance(diffuse * surface.diffuseAlbedo), calcLuminance(specular));
    
    //     specular = DemodulateSpecular(surface.specularF0, specular);

    //     if (needToStore)
    //     {
    //         RTXDI_StoreReservoir(reservoir, params, GlobalIndex, g_Const.shadeInputBufferIndex);
    //     }
    // }

    // Store the sampled lighting luminance for the gradient pass.
    // Discard the pixels where the visibility was reused, as gradients need actual visibility.
    u_RestirLuminance[GlobalIndex] = currLuminance * (reservoir.age > 0 ? 0 : 1);
    
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    if (g_Const.visualizeRegirCells)
    {
        diffuse *= RTXDI_VisualizeReGIRCells(g_Const.runtimeParams, RAB_GetSurfaceWorldPos(surface));
    }
#endif

    if(g_Const.colorDenoiserMode == 1)
    {
        float3 colorWeight = reservoir.colorWeight * visibilityFactor;
        float denominator = calcLuminance(colorWeight);
        diffuse = (denominator == 0.0) ? diffuse : calcLuminance(diffuse) * colorWeight / denominator;
    }
    else if(g_Const.colorDenoiserMode == 2)
    {
        float3 colorWeight = reservoir.colorWeight * visibilityFactor;
        float denominator = calcLuminance(colorWeight);
        diffuse = (denominator == 0.0) ? diffuse : dot(currLuminance, float2(1, 1)) * colorWeight / denominator;
        specular = (denominator == 0.0) ? specular : 0;
    }

    StoreShadingOutput(GlobalIndex, pixelPosition, 
        surface.viewDepth, surface.roughness, diffuse, specular, lightDistance, true, g_Const.enableDenoiserInputPacking);
}
