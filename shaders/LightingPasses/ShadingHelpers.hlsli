/***************************************************************************
 # Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef SHADING_HELPERS_HLSLI
#define SHADING_HELPERS_HLSLI

#ifdef RESERVOIR_HLSLI

bool ShadeSurfaceWithLightSample(
    inout RTXDI_Reservoir reservoir,
    RAB_Surface surface,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance)
{
    diffuse = 0;
    specular = 0;
    lightDistance = 0;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    if (g_Const.enableFinalVisibility)
    {
        float3 visibility = 0;
        bool visibilityReused = false;

        if (g_Const.reuseFinalVisibility && enableVisibilityReuse)
        {
            RTXDI_VisibilityReuseParameters rparams;
            rparams.maxAge = g_Const.finalVisibilityMaxAge;
            rparams.maxDistance = g_Const.finalVisibilityMaxDistance;

            visibilityReused = RTXDI_GetReservoirVisibility(reservoir, rparams, visibility);
        }

        if (!visibilityReused)
        {
            if (previousFrameTLAS && g_Const.enablePreviousTLAS)
                visibility = GetFinalVisibility(PrevSceneBVH, surface, lightSample.position);
            else
                visibility = GetFinalVisibility(SceneBVH, surface, lightSample.position);
            RTXDI_StoreVisibilityInReservoir(reservoir, visibility, g_Const.discardInvisibleSamples);
            needToStore = true;
        }

        lightSample.radiance *= visibility;
    }

    lightSample.radiance *= RTXDI_GetReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (any(lightSample.radiance > 0))
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}

bool ShadeSurfaceWithLightSample(
    inout RTXDI_Reservoir reservoir,
    RAB_Surface surface,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance,
    out float3 visibility)
{
    diffuse = 0;
    specular = 0;
    lightDistance = 0;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    visibility = float3(1.0, 1.0, 1.0);
    if (g_Const.enableFinalVisibility)
    {
        bool visibilityReused = false;

        if (g_Const.reuseFinalVisibility && enableVisibilityReuse)
        {
            RTXDI_VisibilityReuseParameters rparams;
            rparams.maxAge = g_Const.finalVisibilityMaxAge;
            rparams.maxDistance = g_Const.finalVisibilityMaxDistance;

            visibilityReused = RTXDI_GetReservoirVisibility(reservoir, rparams, visibility);
        }

        if (!visibilityReused)
        {
            if (previousFrameTLAS && g_Const.enablePreviousTLAS)
                visibility = GetFinalVisibility(PrevSceneBVH, surface, lightSample.position);
            else
                visibility = GetFinalVisibility(SceneBVH, surface, lightSample.position);
            RTXDI_StoreVisibilityInReservoir(reservoir, visibility, g_Const.discardInvisibleSamples);
            needToStore = true;
        }

        lightSample.radiance *= visibility;
    }

    lightSample.radiance *= RTXDI_GetReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (any(lightSample.radiance > 0))
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}

#endif // RESERVOIR_HLSLI

float3 DemodulateSpecular(float3 surfaceSpecularF0, float3 specular)
{
    return specular / max(0.01, surfaceSpecularF0);
}


void StoreShadingOutput(
    uint2 reservoirPosition,
    uint2 pixelPosition,
    float viewDepth,
    float roughness,
    float3 diffuse,
    float3 specular,
    float lightDistance,
    bool isFirstPass,
    bool isLastPass)
{
    uint2 lightingTexturePos = (g_Const.denoiserMode != DENOISER_MODE_OFF)
        ? reservoirPosition
        : pixelPosition;

    float diffuseHitT = lightDistance;
    float specularHitT = lightDistance;

    if (!isFirstPass)
    {
        float4 priorDiffuse = u_DiffuseLighting[lightingTexturePos];
        float4 priorSpecular = u_SpecularLighting[lightingTexturePos];

        if (calcLuminance(diffuse) > calcLuminance(priorDiffuse.rgb) || lightDistance == 0)
            diffuseHitT = priorDiffuse.w;

        if (calcLuminance(specular) > calcLuminance(priorSpecular.rgb) || lightDistance == 0)
            specularHitT = priorSpecular.w;
        
        diffuse += priorDiffuse.rgb;
        specular += priorSpecular.rgb;
    }

    if (g_Const.denoiserMode == DENOISER_MODE_OFF && g_Const.runtimeParams.activeCheckerboardField != 0 && isLastPass)
    {
        int2 otherFieldPixelPosition = pixelPosition;
        otherFieldPixelPosition.x += (g_Const.runtimeParams.activeCheckerboardField == 1) == ((pixelPosition.y & 1) != 0)
            ? 1 : -1;

        if (g_Const.denoiserMode == DENOISER_MODE_RELAX || g_Const.enableAccumulation)
        {
            diffuse *= 2;
            specular *= 2;

            u_DiffuseLighting[otherFieldPixelPosition] = 0;
            u_SpecularLighting[otherFieldPixelPosition] = 0;
        }
        else // g_Const.denoiserMode == DENOISER_MODE_OFF
        {
            u_DiffuseLighting[otherFieldPixelPosition] = float4(diffuse, 0);
            u_SpecularLighting[otherFieldPixelPosition] = float4(specular, 0);
        }
    }

#if WITH_NRD
    if(g_Const.denoiserMode != DENOISER_MODE_OFF && isLastPass)
    {
        const bool useReLAX = (g_Const.denoiserMode == DENOISER_MODE_RELAX);
 
        if (useReLAX)
        {
            u_DiffuseLighting[lightingTexturePos] = RELAX_FrontEnd_PackRadianceAndHitDist(diffuse, diffuseHitT);
            u_SpecularLighting[lightingTexturePos] = RELAX_FrontEnd_PackRadianceAndHitDist(specular, specularHitT);
        }
        else
        {
            float diffNormDist = REBLUR_FrontEnd_GetNormHitDist(diffuseHitT, viewDepth, g_Const.reblurDiffHitDistParams);
            u_DiffuseLighting[lightingTexturePos] = REBLUR_FrontEnd_PackRadianceAndHitDist(diffuse, diffNormDist);
            
            float specNormDist = REBLUR_FrontEnd_GetNormHitDist(specularHitT, viewDepth, g_Const.reblurSpecHitDistParams, roughness);
            u_SpecularLighting[lightingTexturePos] = REBLUR_FrontEnd_PackRadianceAndHitDist(specular, specNormDist);
        }
    }
    else
#endif
    {
        u_DiffuseLighting[lightingTexturePos] = float4(diffuse, diffuseHitT);
        u_SpecularLighting[lightingTexturePos] = float4(specular, specularHitT);
    }
}

#endif // SHADING_HELPERS_HLSLI