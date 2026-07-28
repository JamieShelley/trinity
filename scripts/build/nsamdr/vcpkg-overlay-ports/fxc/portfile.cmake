if(NOT VCPKG_TARGET_IS_WINDOWS)
    message(FATAL_ERROR "The local FXC overlay supports Windows only.")
endif()

set(_fxc_search_dirs)

if(DEFINED ENV{WindowsSdkVerBinPath} AND NOT "$ENV{WindowsSdkVerBinPath}" STREQUAL "")
    list(APPEND _fxc_search_dirs
        "$ENV{WindowsSdkVerBinPath}/x64"
        "$ENV{WindowsSdkVerBinPath}/x86"
        "$ENV{WindowsSdkVerBinPath}")
endif()

if(DEFINED ENV{WindowsSdkDir} AND DEFINED ENV{WindowsSDKVersion})
    list(APPEND _fxc_search_dirs
        "$ENV{WindowsSdkDir}/bin/$ENV{WindowsSDKVersion}/x64"
        "$ENV{WindowsSdkDir}/bin/$ENV{WindowsSDKVersion}/x86")
endif()

file(GLOB _windows_sdk_versions LIST_DIRECTORIES true
    "C:/Program Files (x86)/Windows Kits/10/bin/*")
list(SORT _windows_sdk_versions COMPARE NATURAL ORDER DESCENDING)

foreach(_sdk_version_dir IN LISTS _windows_sdk_versions)
    if(IS_DIRECTORY "${_sdk_version_dir}")
        list(APPEND _fxc_search_dirs
            "${_sdk_version_dir}/x64"
            "${_sdk_version_dir}/x86")
    endif()
endforeach()

set(_fxc_exe "")
set(_d3dcompiler_dll "")

foreach(_candidate_dir IN LISTS _fxc_search_dirs)
    if(EXISTS "${_candidate_dir}/fxc.exe" AND
       EXISTS "${_candidate_dir}/D3DCompiler_47.dll")
        set(_fxc_exe "${_candidate_dir}/fxc.exe")
        set(_d3dcompiler_dll "${_candidate_dir}/D3DCompiler_47.dll")
        break()
    endif()
endforeach()

if(_fxc_exe STREQUAL "")
    message(FATAL_ERROR
        "Could not locate fxc.exe and D3DCompiler_47.dll in the installed Windows 10 SDK. "
        "Install a Windows 10 SDK component in Visual Studio Installer.")
endif()

message(STATUS "Using Windows SDK FXC: ${_fxc_exe}")

file(MAKE_DIRECTORY
    "${CURRENT_PACKAGES_DIR}/tools/${PORT}"
    "${CURRENT_PACKAGES_DIR}/bin"
    "${CURRENT_PACKAGES_DIR}/share/${PORT}")

file(COPY "${_fxc_exe}" DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")
file(COPY "${_d3dcompiler_dll}" DESTINATION "${CURRENT_PACKAGES_DIR}/bin")

vcpkg_copy_tool_dependencies("${CURRENT_PACKAGES_DIR}/tools/${PORT}")

file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/usage"
"The fxc executable is available through the fxc CMake package/tool path.\n")
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright"
"FXC and D3DCompiler_47.dll are copied from the locally installed Microsoft Windows SDK.\n")
