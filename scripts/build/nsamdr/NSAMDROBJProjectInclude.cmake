# NSAMDR V10.8.5 standalone DX11 preview injection.
#
# The preview deliberately does NOT attach itself to TrinityALTest_dx11 and does
# NOT link TrinityAL_dx11.  The previous diagnostic path rebuilt the full
# TrinityAL library even though NSAMDRPreviewApplication only needs raw D3D11
# interfaces.  That dragged the legacy Nsight Aftermath SDK into a diagnostic
# viewer and caused compile failures unrelated to NSAMDR.

if(NOT COMMAND nsamdr_create_standalone_obj_preview)
    function(nsamdr_create_standalone_obj_preview)
        if(NOT WIN32)
            message(FATAL_ERROR "NSAMDR standalone OBJ preview currently requires Windows")
        endif()

        if(TARGET NSAMDRPreview_dx11)
            return()
        endif()

        set(_NSAMDR_OBJ_DIR "${CMAKE_SOURCE_DIR}/trinityal/tests/nsamdr")
        set(_NSAMDR_STANDALONE_DIR "${_NSAMDR_OBJ_DIR}/standalone")
        set(_NSAMDR_PREVIEW_HLSL "${_NSAMDR_OBJ_DIR}/NSAMDRPreview.hlsl")
        set(_NSAMDR_ICON_ICO "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewIcon.ico")
        set(_NSAMDR_ICON_PNG "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewIcon.png")

        set(_NSAMDR_OBJ_SOURCES
            "${_NSAMDR_STANDALONE_DIR}/NSAMDRStandalonePreview.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewTypes.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewUtilities.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRCameraController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMeshProcessor.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRInputController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRShaderLibrary.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRAssetProcessor.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRSceneController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRRenderPipeline.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewRenderer.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewProcessing.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPanel.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewApplication.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRWindowIcon.cpp")

        set(_NSAMDR_OBJ_HEADERS
            "${_NSAMDR_STANDALONE_DIR}/StdAfx.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPlatform.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewTypes.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewUtilities.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRCameraController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMeshProcessor.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRInputController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRShaderLibrary.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRAssetProcessor.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRSceneController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRRenderPipeline.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewRenderer.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewProcessing.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPanel.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewApplication.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRWindowIcon.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewResource.h")

        foreach(_required_file IN ITEMS
                ${_NSAMDR_OBJ_SOURCES}
                ${_NSAMDR_OBJ_HEADERS}
                "${_NSAMDR_PREVIEW_HLSL}"
                "${_NSAMDR_ICON_ICO}"
                "${_NSAMDR_ICON_PNG}")
            if(NOT EXISTS "${_required_file}")
                message(FATAL_ERROR "Missing NSAMDR standalone preview file: ${_required_file}")
            endif()
        endforeach()

        find_package(imgui CONFIG REQUIRED)
        find_package(GTest CONFIG REQUIRED)

        file(TO_NATIVE_PATH "${_NSAMDR_ICON_ICO}" _NSAMDR_ICON_NATIVE)
        string(REPLACE "\\" "\\\\" _NSAMDR_ICON_ESCAPED "${_NSAMDR_ICON_NATIVE}")
        set(_NSAMDR_ICON_RC "${CMAKE_BINARY_DIR}/nsamdr-preview/NSAMDRPreviewIcon.generated.rc")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/nsamdr-preview")
        file(WRITE "${_NSAMDR_ICON_RC}" "1 ICON \"${_NSAMDR_ICON_ESCAPED}\"\n101 ICON \"${_NSAMDR_ICON_ESCAPED}\"\n")
        set_source_files_properties("${_NSAMDR_ICON_RC}" PROPERTIES GENERATED TRUE)

        add_executable(NSAMDRPreview_dx11
            ${_NSAMDR_OBJ_SOURCES}
            ${_NSAMDR_OBJ_HEADERS}
            "${_NSAMDR_ICON_RC}")

        # Every legacy NSAMDR preview .cpp includes "StdAfx.h".  Put the
        # standalone replacement first so the target never sees
        # trinityal/StdAfx.h (which includes GFSDK_Aftermath.h).
        target_include_directories(NSAMDRPreview_dx11 BEFORE PRIVATE
            "${_NSAMDR_STANDALONE_DIR}"
            "${_NSAMDR_OBJ_DIR}")

        target_compile_definitions(NSAMDRPreview_dx11 PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
            TRINITY_PLATFORM=2
            TRINITY_DIRECTX11=2
            NSAMDR_OBJ_PREVIEW=1)

        file(TO_CMAKE_PATH "${_NSAMDR_PREVIEW_HLSL}" _NSAMDR_PREVIEW_SHADER_PATH)
        target_compile_definitions(NSAMDRPreview_dx11 PRIVATE
            NSAMDR_PREVIEW_SHADER_PATH="${_NSAMDR_PREVIEW_SHADER_PATH}")

        target_compile_features(NSAMDRPreview_dx11 PRIVATE cxx_std_17)
        target_link_libraries(NSAMDRPreview_dx11 PRIVATE
            imgui::imgui
            GTest::gtest
            GTest::gtest_main
            d3d11
            dxgi
            dxguid
            d3dcompiler
            windowscodecs
            ole32
            user32
            gdi32
            shell32)

        set_target_properties(NSAMDRPreview_dx11 PROPERTIES
            OUTPUT_NAME "NSAMDRPreview_dx11"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nsamdr-preview"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/nsamdr-preview"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/nsamdr-preview"
            RUNTIME_OUTPUT_DIRECTORY_INTERNAL "${CMAKE_BINARY_DIR}/nsamdr-preview"
            RUNTIME_OUTPUT_DIRECTORY_TRINITYDEV "${CMAKE_BINARY_DIR}/nsamdr-preview"
            PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nsamdr-preview")

        source_group("NSAMDR/Source" FILES ${_NSAMDR_OBJ_SOURCES})
        source_group("NSAMDR/Headers" FILES ${_NSAMDR_OBJ_HEADERS})

        message(STATUS "Configured standalone NSAMDRPreview_dx11 (no TrinityAL_dx11 / no Nsight Aftermath)")
    endfunction()
endif()

get_property(_nsamdr_standalone_deferred GLOBAL PROPERTY NSAMDR_STANDALONE_OBJ_DEFERRED)
if(NOT _nsamdr_standalone_deferred)
    set_property(GLOBAL PROPERTY NSAMDR_STANDALONE_OBJ_DEFERRED TRUE)
    cmake_language(DEFER CALL nsamdr_create_standalone_obj_preview)
endif()
