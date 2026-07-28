#pragma once

#if defined(_WIN32)
#  if defined(NSAMDR_REALSHIP_VIEWER_BUILD)
#    define NSAMDR_VIEWER_API __declspec(dllexport)
#  else
#    define NSAMDR_VIEWER_API __declspec(dllimport)
#  endif
#else
#  define NSAMDR_VIEWER_API
#endif

extern "C" NSAMDR_VIEWER_API int NSAMDR_RunRealShipViewer( int argc, wchar_t** argv );
