// Copyright © 2026

#include "NSAMDRRealShipViewer.h"

#if defined(_WIN32)
#include <Windows.h>

int wmain( int argc, wchar_t** argv )
{
    return NSAMDR_RunRealShipViewer( argc, argv );
}
#else
int main()
{
    return 1;
}
#endif
