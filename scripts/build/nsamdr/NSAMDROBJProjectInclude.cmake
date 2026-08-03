# Opt-in CMake injection for the Granny-free NSAMDR real-OBJ preview.
# It is supplied only through CMAKE_PROJECT_INCLUDE by the launcher.

if(NOT COMMAND nsamdr_attach_obj_preview)
    function(nsamdr_attach_obj_preview)
        if(NOT WIN32)
            message(FATAL_ERROR "NSAMDR OBJ preview currently requires Windows")
        endif()
        if(NOT BUILD_DX11)
            message(FATAL_ERROR "NSAMDR OBJ preview requires BUILD_DX11=ON")
        endif()
        if(NOT TARGET TrinityALTest_dx11)
            message(FATAL_ERROR "NSAMDR OBJ preview could not find TrinityALTest_dx11")
        endif()

        get_target_property(_already_attached TrinityALTest_dx11 NSAMDR_OBJ_PREVIEW_ATTACHED)
        if(_already_attached)
            return()
        endif()

        set(_NSAMDR_OBJ_DIR "${CMAKE_SOURCE_DIR}/trinityal/tests/nsamdr")
        set(_NSAMDR_OBJ_SOURCES
            "${_NSAMDR_OBJ_DIR}/NSAMDRShipPreview.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewTypes.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewUtilities.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRCameraController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMeshProcessor.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRInputController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRShaderLibrary.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRStrategyModes.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMode3Pipeline.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRAssetProcessor.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRTrainingController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRSceneController.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRRenderPipeline.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewRenderer.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewProcessing.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPanel.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewApplication.cpp"
            "${_NSAMDR_OBJ_DIR}/NSAMDRWindowIcon.cpp")
        set(_NSAMDR_OBJ_HEADERS
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPlatform.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewTypes.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewUtilities.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRCameraController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMeshProcessor.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRInputController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRShaderLibrary.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRStrategyModes.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRMode3Pipeline.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRAssetProcessor.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRTrainingController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRSceneController.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRRenderPipeline.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewRenderer.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewProcessing.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewPanel.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewApplication.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRWindowIcon.h"
            "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewResource.h")
        set(_NSAMDR_ICON_ICO "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewIcon.ico")
        set(_NSAMDR_ICON_PNG "${_NSAMDR_OBJ_DIR}/NSAMDRPreviewIcon.png")
        set(_NSAMDR_OBJ_SOURCE "${_NSAMDR_OBJ_DIR}/NSAMDRShipPreview.cpp")
        # The HLSL is loaded and compiled explicitly at runtime with
        # D3DCompileFromFile. Do not add it to TrinityALTest_dx11 sources.
        set(_NSAMDR_PREVIEW_HLSL "${_NSAMDR_OBJ_DIR}/NSAMDRPreview.hlsl")
        foreach(_required_file IN ITEMS
                "${_NSAMDR_OBJ_SOURCE}"
                "${_NSAMDR_PREVIEW_HLSL}"
                ${_NSAMDR_OBJ_SOURCES}
                ${_NSAMDR_OBJ_HEADERS}
                "${_NSAMDR_ICON_ICO}"
                "${_NSAMDR_ICON_PNG}")
            if(NOT EXISTS "${_required_file}")
                message(FATAL_ERROR "Missing NSAMDR OBJ source/support file: ${_required_file}")
            endif()
        endforeach()

        find_package(imgui CONFIG REQUIRED)

        # Older local overlays may already list this source in trinityal/tests/CMakeLists.txt.
        # Compare absolute source paths before adding it again.
        get_target_property(_existing_sources TrinityALTest_dx11 SOURCES)
        set(_source_present FALSE)
        foreach(_existing_source IN LISTS _existing_sources)
            if(IS_ABSOLUTE "${_existing_source}")
                set(_existing_absolute "${_existing_source}")
            else()
                get_filename_component(
                    _existing_absolute
                    "${CMAKE_SOURCE_DIR}/trinityal/tests/${_existing_source}"
                    ABSOLUTE)
            endif()
            get_filename_component(_existing_absolute "${_existing_absolute}" REALPATH)
            get_filename_component(_required_absolute "${_NSAMDR_OBJ_SOURCE}" REALPATH)
            if(_existing_absolute STREQUAL _required_absolute)
                set(_source_present TRUE)
                break()
            endif()
        endforeach()

        if(_source_present)
            list(REMOVE_ITEM _NSAMDR_OBJ_SOURCES "${_NSAMDR_OBJ_SOURCE}")
        endif()
        file(TO_NATIVE_PATH "${_NSAMDR_ICON_ICO}" _NSAMDR_ICON_NATIVE)
        string(REPLACE "\\" "\\\\" _NSAMDR_ICON_ESCAPED "${_NSAMDR_ICON_NATIVE}")
        set(_NSAMDR_ICON_RC "${CMAKE_CURRENT_BINARY_DIR}/NSAMDRPreviewIcon.generated.rc")
        file(WRITE "${_NSAMDR_ICON_RC}" "1 ICON \"${_NSAMDR_ICON_ESCAPED}\"\n101 ICON \"${_NSAMDR_ICON_ESCAPED}\"\n")
        set_source_files_properties("${_NSAMDR_ICON_RC}" PROPERTIES GENERATED TRUE)

        target_sources(TrinityALTest_dx11 PRIVATE
            ${_NSAMDR_OBJ_SOURCES}
            ${_NSAMDR_OBJ_HEADERS}
            "${_NSAMDR_ICON_RC}"
            "${_NSAMDR_ICON_ICO}"
            "${_NSAMDR_ICON_PNG}")
        set_source_files_properties(${_NSAMDR_OBJ_HEADERS} "${_NSAMDR_ICON_ICO}" "${_NSAMDR_ICON_PNG}" PROPERTIES HEADER_FILE_ONLY TRUE)
        source_group("NSAMDR/Source" FILES ${_NSAMDR_OBJ_SOURCES})
        source_group("NSAMDR/Headers" FILES ${_NSAMDR_OBJ_HEADERS})

        target_link_libraries(TrinityALTest_dx11 PRIVATE
            imgui::imgui
            d3dcompiler
            windowscodecs)
        file(TO_CMAKE_PATH "${_NSAMDR_PREVIEW_HLSL}" _NSAMDR_PREVIEW_SHADER_PATH)
        target_compile_definitions(TrinityALTest_dx11 PRIVATE
            NSAMDR_OBJ_PREVIEW=1
            NSAMDR_PREVIEW_SHADER_PATH="${_NSAMDR_PREVIEW_SHADER_PATH}")
        set_property(TARGET TrinityALTest_dx11 PROPERTY NSAMDR_OBJ_PREVIEW_ATTACHED TRUE)

        message(STATUS "Configured Granny-free NSAMDR real-OBJ preview on TrinityALTest_dx11")
    endfunction()
endif()

# The first project() is the repository root. Defer until the end of that
# directory, after trinityal/tests has created TrinityALTest_dx11.
get_property(_nsamdr_obj_deferred GLOBAL PROPERTY NSAMDR_OBJ_DEFERRED)
if(NOT _nsamdr_obj_deferred)
    set_property(GLOBAL PROPERTY NSAMDR_OBJ_DEFERRED TRUE)
    cmake_language(DEFER CALL nsamdr_attach_obj_preview)
endif()
