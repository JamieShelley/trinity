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

        set(_NSAMDR_OBJ_SOURCE "${CMAKE_SOURCE_DIR}/trinityal/tests/nsamdr/NSAMDRShipPreview.cpp")
        if(NOT EXISTS "${_NSAMDR_OBJ_SOURCE}")
            message(FATAL_ERROR "Missing NSAMDR OBJ source: ${_NSAMDR_OBJ_SOURCE}")
        endif()

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

        if(NOT _source_present)
            target_sources(TrinityALTest_dx11 PRIVATE "${_NSAMDR_OBJ_SOURCE}")
        endif()

        target_link_libraries(TrinityALTest_dx11 PRIVATE
            imgui::imgui
            d3dcompiler
            windowscodecs)
        target_compile_definitions(TrinityALTest_dx11 PRIVATE NSAMDR_OBJ_PREVIEW=1)
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
