# Opt-in CMake injection for the local NSAMDR full-Trinity real-ship viewer.
# Supplied only through CMAKE_PROJECT_INCLUDE by the NSAMDR launcher.

if(PROJECT_NAME STREQUAL "trinity")
    function(nsamdr_attach_real_ship_viewer)
        if(NOT WIN32)
            message(FATAL_ERROR "NSAMDRRealShipViewer currently requires Windows")
        endif()
        if(NOT BUILD_DX11)
            message(FATAL_ERROR "NSAMDRRealShipViewer requires BUILD_DX11=ON")
        endif()
        if(NOT WITH_GRANNY)
            message(FATAL_ERROR "NSAMDRRealShipViewer requires WITH_GRANNY=ON for real .gr2 ship geometry")
        endif()
        if(NOT TARGET trinity_dx11)
            message(FATAL_ERROR "NSAMDRRealShipViewer could not find the full-Trinity trinity_dx11 target")
        endif()
        if(TARGET NSAMDRRealShipViewer)
            return()
        endif()

        set(_NSAMDR_DIR "${CMAKE_SOURCE_DIR}/trinity/tools/nsamdr")
        foreach(_required_file
            NSAMDRRealShipViewer.cpp
            NSAMDRRealShipViewer.h
            NSAMDRRealShipViewerMain.cpp)
            if(NOT EXISTS "${_NSAMDR_DIR}/${_required_file}")
                message(FATAL_ERROR "Missing NSAMDR real-ship source: ${_NSAMDR_DIR}/${_required_file}")
            endif()
        endforeach()

        find_package(imgui CONFIG REQUIRED)

        # The implementation is compiled into full Trinity so it can use private
        # EveSOF, EveSpaceScene and real material pipeline types.
        target_sources(trinity_dx11 PRIVATE
            "${_NSAMDR_DIR}/NSAMDRRealShipViewer.cpp"
            "${_NSAMDR_DIR}/NSAMDRRealShipViewer.h"
        )
        target_include_directories(trinity_dx11 PRIVATE
            "${CMAKE_SOURCE_DIR}/vendor/github.com/carbonengine/blue/src"
            "${_NSAMDR_DIR}"
        )
        target_link_libraries(trinity_dx11 PRIVATE imgui::imgui)
        target_compile_definitions(trinity_dx11 PRIVATE
            NSAMDR_REALSHIP_VIEWER=1
            NSAMDR_REALSHIP_VIEWER_BUILD=1
        )

        ccp_add_executable(NSAMDRRealShipViewer
            "${_NSAMDR_DIR}/NSAMDRRealShipViewerMain.cpp"
            "${_NSAMDR_DIR}/NSAMDRRealShipViewer.h"
        )
        target_include_directories(NSAMDRRealShipViewer PRIVATE "${_NSAMDR_DIR}")
        target_link_libraries(NSAMDRRealShipViewer PRIVATE trinity_dx11)
        target_compile_definitions(NSAMDRRealShipViewer PRIVATE TRINITY_PLATFORM=TRINITY_DIRECTX11)

        set(_NSAMDR_OUTPUT_DIR
            "${CMAKE_BINARY_DIR}/carbon/autobuild/NSAMDRRealShipViewer/${CCP_PLATFORM}/${CCP_ARCHITECTURE}/${CCP_TOOLSET}/$<$<CONFIG:DEBUG>:>")
        set_target_properties(NSAMDRRealShipViewer PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${_NSAMDR_OUTPUT_DIR}"
            PDB_OUTPUT_DIRECTORY "${_NSAMDR_OUTPUT_DIR}"
            FOLDER "Trinity/Tools"
        )

        add_custom_command(TARGET NSAMDRRealShipViewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:NSAMDRRealShipViewer>
                $<TARGET_FILE_DIR:NSAMDRRealShipViewer>
            COMMAND_EXPAND_LISTS
        )

        message(STATUS "Configured NSAMDRRealShipViewer against full Trinity DX11 + EveSOF + Granny")
    endfunction()

    # CMAKE_PROJECT_INCLUDE runs immediately after project(trinity), before the
    # trinity_dx11 target is declared. Defer attachment to the end of this directory.
    cmake_language(DEFER CALL nsamdr_attach_real_ship_viewer)
endif()
