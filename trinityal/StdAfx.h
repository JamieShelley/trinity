// Copyright © 2023 CCP ehf.

#ifndef TrinityALTest_StdAfx_H
#define TrinityALTest_StdAfx_H

#ifdef _MSC_VER
#pragma warning( push, 3 )
#endif

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX //don't want that evil microsoft macro
#endif
#include <windows.h>
#endif

#include <Windows.h>
typedef HWND Tr2WindowHandle;
#elif defined( __APPLE__ )
#include <objc/objc-runtime.h>
typedef id Tr2WindowHandle;
#else
#include <cstdint>
typedef uintptr_t Tr2WindowHandle;
#endif

#ifdef _MSC_VER
#pragma warning( pop )
#endif

// clang-format off
#define INCLUDE_SHADER_CODE( name ) CCP_STRINGIZE( SHADER_PATH/name.h )
// clang-format on

#if ( TRINITY_PLATFORM == TRINITY_DIRECTX11 || TRINITY_PLATFORM == TRINITY_DIRECTX12 )
#include <GFSDK_Aftermath.h>
#endif
}

inline AssertionResult IsHRESULTSuccess( const char* expr, HRESULT hr )
{
	if( SUCCEEDED( hr ) )
	{
		return AssertionSuccess();
	}
	return HRESULTFailureHelper( expr, "succeeds", hr );
}

inline AssertionResult IsHRESULTFailure( const char* expr, HRESULT hr )
{
	if( FAILED( hr ) )
	{
		return AssertionSuccess();
	}
	return HRESULTFailureHelper( expr, "succeeds", hr );
}
}
}

#define EXPECT_HRESULT_SUCCEEDED( expr ) EXPECT_PRED_FORMAT1( ::testing::internal::IsHRESULTSuccess, ( expr ) )
#define ASSERT_HRESULT_SUCCEEDED( expr ) ASSERT_PRED_FORMAT1( ::testing::internal::IsHRESULTSuccess, ( expr ) )
#define EXPECT_HRESULT_FAILED( expr ) EXPECT_PRED_FORMAT1( ::testing::internal::IsHRESULTFailure, ( expr ) )
#define ASSERT_HRESULT_FAILED( expr ) ASSERT_PRED_FORMAT1( ::testing::internal::IsHRESULTFailure, ( expr ) )

#endif

#endif
