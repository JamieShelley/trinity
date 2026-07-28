// Copyright © 2026
// Local full-Trinity NSAMDR real-ship preview.

#include "StdAfx.h"
#include "NSAMDRRealShipViewer.h"

#if defined(_WIN32) && TRINITY_PLATFORM == TRINITY_DIRECTX11

#include "TriDevice.h"
#include "TriView.h"
#include "TriProjection.h"
#include "Tr2Renderer.h"
#include "Tr2Variable.h"
#include "RenderJob/Tr2RenderJobs.h"
#include "RenderJob/TriRenderJob.h"
#include "RenderJob/TriStepClear.h"
#include "RenderJob/TriStepSetView.h"
#include "RenderJob/TriStepSetProjection.h"
#include "RenderJob/TriStepRenderScene.h"
#include "Eve/EveSpaceScene.h"
#include "Eve/IEveSpaceObject2.h"
#include "Eve/SpaceObjectFactory/EveSOF.h"
#include "Eve/SpaceObjectFactory/EveSOFData.h"

#include <Blue.h>
#include <BlueRegistration.h>
#include <BlueResMan.h>
#include <BluePaths.h>
#include <RemoteFileCache.h>
#include <ResourceLoading.h>
#include <IBluePaths.h>
#include <IBlueOS.h>

#include <Python.h>

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include <d3d11.h>
#include <windowsx.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cfloat>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler( HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam );
extern void InitializeTrinity();

using Microsoft::WRL::ComPtr;

namespace
{

constexpr int kWindowWidth = 1440;
constexpr int kWindowHeight = 900;
constexpr float kPi = 3.14159265358979323846f;

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 Add( const Vec3& a, const Vec3& b )
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

Vec3 Sub( const Vec3& a, const Vec3& b )
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

Vec3 Mul( const Vec3& value, float scale )
{
    return { value.x * scale, value.y * scale, value.z * scale };
}

float Length( const Vec3& value )
{
    return std::sqrt( value.x * value.x + value.y * value.y + value.z * value.z );
}

Vec3 Normalize( const Vec3& value )
{
    const float length = Length( value );
    if( length <= 1.0e-6f )
    {
        return { 0.0f, 0.0f, 1.0f };
    }
    return Mul( value, 1.0f / length );
}

Vec3 Cross( const Vec3& a, const Vec3& b )
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

std::string ToLower( std::string value )
{
    std::transform( value.begin(), value.end(), value.begin(), []( unsigned char c ) {
        return static_cast<char>( std::tolower( c ) );
    } );
    return value;
}

bool ContainsInsensitive( const std::string& value, const std::string& filter )
{
    if( filter.empty() )
    {
        return true;
    }
    return ToLower( value ).find( ToLower( filter ) ) != std::string::npos;
}

std::wstring GetArgument( int argc, wchar_t** argv, const wchar_t* name )
{
    for( int index = 1; index + 1 < argc; ++index )
    {
        if( _wcsicmp( argv[index], name ) == 0 )
        {
            return argv[index + 1];
        }
    }
    return {};
}

std::wstring GetEnvironmentWide( const wchar_t* name )
{
    const DWORD required = GetEnvironmentVariableW( name, nullptr, 0 );
    if( required == 0 )
    {
        return {};
    }
    std::wstring result( required, L'\0' );
    GetEnvironmentVariableW( name, result.data(), required );
    if( !result.empty() && result.back() == L'\0' )
    {
        result.pop_back();
    }
    return result;
}

bool ReadBinaryTextFile( const std::wstring& path, std::string& contents )
{
    std::ifstream input( path, std::ios::binary );
    if( !input )
    {
        return false;
    }
    input.seekg( 0, std::ios::end );
    const std::streamoff size = input.tellg();
    input.seekg( 0, std::ios::beg );
    if( size <= 0 )
    {
        return false;
    }
    contents.resize( static_cast<size_t>( size ) );
    input.read( contents.data(), size );
    return input.good() || input.eof();
}

std::vector<std::string> FindSofCandidates( const std::string& indexContents )
{
    std::vector<std::string> candidates;
    std::istringstream stream( indexContents );
    std::string line;
    while( std::getline( stream, line ) )
    {
        const size_t comma = line.find( ',' );
        if( comma == std::string::npos )
        {
            continue;
        }
        std::string path = line.substr( 0, comma );
        const std::string lower = ToLower( path );
        const bool isPersisted = lower.size() >= 6 &&
            ( lower.rfind( ".black" ) == lower.size() - 6 || lower.rfind( ".red" ) == lower.size() - 4 );
        const bool looksLikeSof = lower.find( "spaceobjectfactory" ) != std::string::npos ||
            lower.find( "/sof/" ) != std::string::npos ||
            lower.find( "sofdata" ) != std::string::npos;
        if( isPersisted && looksLikeSof )
        {
            if( lower.rfind( "res:/", 0 ) != 0 )
            {
                path = "res:/" + path;
            }
            candidates.push_back( path );
        }
    }

    std::sort( candidates.begin(), candidates.end(), []( const std::string& a, const std::string& b ) {
        const auto rank = []( const std::string& value ) {
            const std::string lower = ToLower( value );
            int score = 1000;
            if( lower.find( "spaceobjectfactory/data" ) != std::string::npos ) score -= 500;
            if( lower.find( "sofdata" ) != std::string::npos ) score -= 250;
            if( lower.find( "data.black" ) != std::string::npos ) score -= 100;
            score += static_cast<int>( value.size() );
            return score;
        };
        return rank( a ) < rank( b );
    } );
    candidates.erase( std::unique( candidates.begin(), candidates.end() ), candidates.end() );
    return candidates;
}

bool AppendPythonPath( PyConfig& config, const std::wstring& path )
{
    if( path.empty() )
    {
        return true;
    }
    const PyStatus status = PyWideStringList_Append( &config.module_search_paths, path.c_str() );
    return !PyStatus_Exception( status );
}

bool InitializePython( std::string& error )
{
    if( Py_IsInitialized() != 0 )
    {
        return true;
    }

    PyPreConfig preConfig;
    PyPreConfig_InitIsolatedConfig( &preConfig );
    PyStatus status = Py_PreInitialize( &preConfig );
    if( PyStatus_Exception( status ) )
    {
        error = status.err_msg ? status.err_msg : "Py_PreInitialize failed";
        return false;
    }

    const std::wstring importPath = GetEnvironmentWide( L"NSAMDR_IMPORT_PATH" );
    const std::wstring stdlibPath = GetEnvironmentWide( L"NSAMDR_PYTHON_STDLIB" );

    PyConfig config;
    PyConfig_InitIsolatedConfig( &config );
    bool pathOk = AppendPythonPath( config, L"." );
    if( !importPath.empty() )
    {
        pathOk = pathOk && AppendPythonPath( config, importPath + L"/bin/python" );
        pathOk = pathOk && AppendPythonPath( config, importPath + L"/bin" );
        pathOk = pathOk && AppendPythonPath( config, importPath + L"/lib" );
    }
    if( !stdlibPath.empty() )
    {
        pathOk = pathOk && AppendPythonPath( config, stdlibPath );
    }
    config.module_search_paths_set = 1;

    if( !pathOk )
    {
        PyConfig_Clear( &config );
        error = "Could not configure Python module search paths";
        return false;
    }

    status = Py_InitializeFromConfig( &config );
    PyConfig_Clear( &config );
    if( PyStatus_Exception( status ) )
    {
        error = status.err_msg ? status.err_msg : "Py_InitializeFromConfig failed";
        return false;
    }

    if( !InstallImportHook() )
    {
        error = "Blue InstallImportHook failed";
        return false;
    }

    PyObject* scheduler = PyImport_ImportModule( "scheduler" );
    if( scheduler == nullptr )
    {
        PyErr_Print();
        error = "Could not import scheduler";
        return false;
    }
    Py_DECREF( scheduler );

    PyObject* carbonSocket = PyImport_ImportModule( "_carbonsocket" );
    if( carbonSocket == nullptr )
    {
        PyErr_Print();
        error = "Could not import _carbonsocket";
        return false;
    }
    Py_DECREF( carbonSocket );
    return true;
}

class PreviewScene final : public EveSpaceScene
{
public:
    PreviewScene() = default;

    void ConfigureLighting(
        float keyIntensity,
        float fillIntensity,
        float rimIntensity,
        float backgroundBrightness,
        float yaw,
        float pitch )
    {
        const float cp = std::cos( pitch );
        Vec3 direction = Normalize( { cp * std::cos( yaw ), std::sin( pitch ), cp * std::sin( yaw ) } );
        m_sunData.DirWorld = Vector3( direction.x, direction.y, direction.z );
        // EVE's space material path exposes one directional sun plus ambient and
        // environment reflection. Use those three independent terms as a studio
        // key, cool fill and broad rim/reflection light.
        m_sunColor = Color( 1.00f * keyIntensity, 0.95f * keyIntensity, 0.86f * keyIntensity, 1.0f );
        m_sunData.DiffuseColor = m_sunColor;
        m_ambientColor = Color( 0.14f * fillIntensity, 0.23f * fillIntensity, 0.36f * fillIntensity, 1.0f );
        m_fogColor = Color(
            0.010f * backgroundBrightness,
            0.020f * backgroundBrightness,
            0.040f * backgroundBrightness,
            1.0f );
        m_fogStart = 1000000.0f;
        m_fogEnd = 2000000.0f;
        m_fogMax = 0.0f;
        m_nebulaIntensity = 0.0f;
        m_backgroundReflectionIntensity = rimIntensity;
        m_backgroundRenderingEnabled = false;
        m_display = true;
        m_update = true;
    }
};

struct CatalogHull
{
    EveSOFDataHull* data = nullptr;
    std::string searchText;
};

struct ViewerApp
{
    HWND hwnd = nullptr;
    bool running = true;
    bool rightDragging = false;
    bool middleDragging = false;
    POINT previousMouse = {};

    Vec3 cameraTarget = {};
    float cameraDistance = 1200.0f;
    float cameraYaw = -0.65f;
    float cameraPitch = 0.22f;
    float orbitSensitivity = 0.006f;
    float panSensitivity = 0.0017f;
    float zoomSensitivity = 0.13f;
    float framedRadius = 400.0f;

    float keyIntensity = 1.85f;
    float fillIntensity = 0.82f;
    float rimIntensity = 0.42f;
    float backgroundBrightness = 0.65f;
    float lightYaw = -0.72f;
    float lightPitch = -0.58f;
    float exposure = 1.0f;
    int nsamdrMode = 0;
    int previousEnabledMode = 1;
    float nsamdrStrength = 1.0f;

    char hullFilter[128] = "raven";
    char factionFilter[128] = "caldari";
    char raceFilter[128] = "caldari";
    char dnaBuffer[512] = {};

    std::string status = "Initialising";
    std::string resourceIndexContents;
    std::vector<std::string> sofCandidates;
    int selectedSofCandidate = 0;

    EveSOFPtr sof;
    EveSOFDataPtr sofData;
    IRootPtr currentShipRoot;
    IEveSpaceObject2Ptr currentShip;
    PreviewScene* scene = nullptr;

    std::vector<CatalogHull> hulls;
    std::vector<EveSOFDataFaction*> factions;
    std::vector<EveSOFDataRace*> races;
    int selectedHull = -1;
    int selectedFaction = -1;
    int selectedRace = -1;

    TriViewPtr view;
    TriProjectionPtr projection;
    Tr2RenderJobsPtr renderJobs;
    TriRenderJobPtr renderJob;
    TriStepClearPtr clearStep;
    TriStepSetViewPtr viewStep;
    TriStepSetProjectionPtr projectionStep;
    TriStepRenderScenePtr sceneStep;

    Tr2Variable nsamdrSettings;
    ComPtr<ID3D11RenderTargetView> imguiRenderTarget;
};

ViewerApp* g_app = nullptr;

void ResetCamera( ViewerApp& app )
{
    app.cameraDistance = std::max( app.framedRadius * 2.8f, 10.0f );
    app.cameraYaw = -0.65f;
    app.cameraPitch = 0.22f;
}

void UpdateCamera( ViewerApp& app )
{
    const float cp = std::cos( app.cameraPitch );
    const Vec3 forward = Normalize( {
        cp * std::cos( app.cameraYaw ),
        std::sin( app.cameraPitch ),
        cp * std::sin( app.cameraYaw ),
    } );
    const Vec3 eye = Sub( app.cameraTarget, Mul( forward, app.cameraDistance ) );
    app.view->SetLookAtPosition(
        Vector3( eye.x, eye.y, eye.z ),
        Vector3( app.cameraTarget.x, app.cameraTarget.y, app.cameraTarget.z ),
        Vector3( 0.0f, 1.0f, 0.0f ) );
    app.projection->PerspectiveFov( 52.0f * kPi / 180.0f, static_cast<float>( kWindowWidth ) / static_cast<float>( kWindowHeight ), 0.1f, std::max( 1000000.0f, app.cameraDistance * 100.0f ) );
}

void PanCamera( ViewerApp& app, float dx, float dy )
{
    const float cp = std::cos( app.cameraPitch );
    const Vec3 forward = Normalize( {
        cp * std::cos( app.cameraYaw ),
        std::sin( app.cameraPitch ),
        cp * std::sin( app.cameraYaw ),
    } );
    const Vec3 right = Normalize( Cross( forward, { 0.0f, 1.0f, 0.0f } ) );
    const Vec3 up = Normalize( Cross( right, forward ) );
    const float scale = app.cameraDistance * app.panSensitivity;
    app.cameraTarget = Add( app.cameraTarget, Add( Mul( right, -dx * scale ), Mul( up, dy * scale ) ) );
}

void FrameCurrentShip( ViewerApp& app )
{
    Vector4 sphere;
    if( app.currentShip && app.currentShip->GetBoundingSphere( sphere ) )
    {
        app.cameraTarget = { sphere.x, sphere.y, sphere.z };
        app.framedRadius = std::max( sphere.w, 1.0f );
    }
    else
    {
        app.cameraTarget = {};
        app.framedRadius = 400.0f;
    }
    ResetCamera( app );
}

void ClearSceneObjects( ViewerApp& app )
{
    if( app.scene == nullptr )
    {
        return;
    }
    while( app.scene->Objects().GetSize() > 0 )
    {
        app.scene->Objects().Remove( 0 );
    }
    app.currentShip = nullptr;
    app.currentShipRoot = nullptr;
}

bool BuildShipFromDna( ViewerApp& app, const std::string& dna )
{
    if( !app.sof )
    {
        app.status = "SOF has not been loaded";
        return false;
    }

    IRootPtr built = app.sof->BuildFromDNA( dna.c_str() );
    if( !built )
    {
        app.status = "EveSOF rejected DNA: " + dna;
        return false;
    }

    IEveSpaceObject2Ptr ship = BlueCastPtr( built->GetRawRoot() );
    if( !ship )
    {
        app.status = "DNA did not produce an IEveSpaceObject2: " + dna;
        return false;
    }

    ClearSceneObjects( app );
    app.currentShipRoot = built;
    app.currentShip = ship;
    app.scene->Objects().Insert( -1, ship->GetRawRoot() );
    strncpy_s( app.dnaBuffer, dna.c_str(), _TRUNCATE );
    FrameCurrentShip( app );
    app.status = "Loaded real SOF ship: " + dna;
    return true;
}

std::string SelectedDna( const ViewerApp& app )
{
    if( app.selectedHull < 0 || app.selectedHull >= static_cast<int>( app.hulls.size() ) ||
        app.selectedFaction < 0 || app.selectedFaction >= static_cast<int>( app.factions.size() ) ||
        app.selectedRace < 0 || app.selectedRace >= static_cast<int>( app.races.size() ) )
    {
        return {};
    }
    return app.hulls[app.selectedHull].data->m_name + ":" +
        app.factions[app.selectedFaction]->m_name + ":" +
        app.races[app.selectedRace]->m_name;
}

void SelectDefaults( ViewerApp& app )
{
    auto findHull = [&]( const char* needle ) {
        const std::string search = ToLower( needle );
        for( size_t index = 0; index < app.hulls.size(); ++index )
        {
            if( app.hulls[index].searchText.find( search ) != std::string::npos )
            {
                return static_cast<int>( index );
            }
        }
        return app.hulls.empty() ? -1 : 0;
    };
    auto findFaction = [&]( const char* needle ) {
        for( size_t index = 0; index < app.factions.size(); ++index )
        {
            if( ContainsInsensitive( app.factions[index]->m_name, needle ) )
            {
                return static_cast<int>( index );
            }
        }
        return app.factions.empty() ? -1 : 0;
    };
    auto findRace = [&]( const char* needle ) {
        for( size_t index = 0; index < app.races.size(); ++index )
        {
            if( ContainsInsensitive( app.races[index]->m_name, needle ) )
            {
                return static_cast<int>( index );
            }
        }
        return app.races.empty() ? -1 : 0;
    };

    app.selectedHull = findHull( "raven" );
    app.selectedFaction = findFaction( "caldari" );
    app.selectedRace = findRace( "caldari" );
}

bool LoadSofData( ViewerApp& app, const std::string& path )
{
    app.status = "Loading SOF data: " + path;
    app.sof = nullptr;
    app.sofData = nullptr;
    app.hulls.clear();
    app.factions.clear();
    app.races.clear();
    ClearSceneObjects( app );

    app.sof.CreateInstance();
    if( !app.sof->LoadData( path.c_str() ) )
    {
        app.status = "EveSOF::LoadData failed: " + path;
        return false;
    }

    IRootPtr dataRoot = GetBeResMan()->LoadObject( path.c_str() );
    if( !dataRoot )
    {
        app.status = "SOF data loaded, but catalog object could not be opened: " + path;
        return false;
    }

    app.sofData = BlueCastPtr( dataRoot->GetRawRoot() );
    if( !app.sofData )
    {
        app.status = "Resource is not EveSOFData: " + path;
        return false;
    }

    for( auto& hull : app.sofData->m_hull )
    {
        if( hull && hull->m_buildClass == EveSOFDataHull::BUILDCLASS_SHIP )
        {
            CatalogHull entry;
            entry.data = hull;
            entry.searchText = ToLower( hull->m_name + " " + hull->m_description + " " + std::string( hull->m_category.c_str() ) );
            app.hulls.push_back( std::move( entry ) );
        }
    }
    for( auto& faction : app.sofData->m_faction )
    {
        if( faction ) app.factions.push_back( faction );
    }
    for( auto& race : app.sofData->m_race )
    {
        if( race ) app.races.push_back( race );
    }

    std::sort( app.hulls.begin(), app.hulls.end(), []( const CatalogHull& a, const CatalogHull& b ) {
        return a.data->m_description.empty() ? a.data->m_name < b.data->m_name : a.data->m_description < b.data->m_description;
    } );
    std::sort( app.factions.begin(), app.factions.end(), []( const EveSOFDataFaction* a, const EveSOFDataFaction* b ) {
        return a->m_name < b->m_name;
    } );
    std::sort( app.races.begin(), app.races.end(), []( const EveSOFDataRace* a, const EveSOFDataRace* b ) {
        return a->m_name < b->m_name;
    } );

    SelectDefaults( app );
    const std::string dna = SelectedDna( app );
    if( !dna.empty() )
    {
        BuildShipFromDna( app, dna );
    }
    app.status = "SOF catalog loaded: " + std::to_string( app.hulls.size() ) + " ship hulls, " +
        std::to_string( app.factions.size() ) + " factions, " + std::to_string( app.races.size() ) + " races";
    return true;
}

void DrawCatalogList( ViewerApp& app )
{
    ImGui::InputText( "Hull search", app.hullFilter, std::size( app.hullFilter ) );
    if( ImGui::BeginListBox( "##hulls", ImVec2( -FLT_MIN, 210.0f ) ) )
    {
        for( int index = 0; index < static_cast<int>( app.hulls.size() ); ++index )
        {
            const CatalogHull& hull = app.hulls[index];
            if( !ContainsInsensitive( hull.searchText, app.hullFilter ) )
            {
                continue;
            }
            std::string label = hull.data->m_description.empty() ? hull.data->m_name : hull.data->m_description + "  [" + hull.data->m_name + "]";
            const bool selected = index == app.selectedHull;
            if( ImGui::Selectable( label.c_str(), selected ) )
            {
                app.selectedHull = index;
            }
            if( selected ) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    ImGui::InputText( "Faction search", app.factionFilter, std::size( app.factionFilter ) );
    const char* factionPreview = app.selectedFaction >= 0 ? app.factions[app.selectedFaction]->m_name.c_str() : "None";
    if( ImGui::BeginCombo( "Faction/material", factionPreview ) )
    {
        for( int index = 0; index < static_cast<int>( app.factions.size() ); ++index )
        {
            if( !ContainsInsensitive( app.factions[index]->m_name, app.factionFilter ) ) continue;
            const bool selected = index == app.selectedFaction;
            if( ImGui::Selectable( app.factions[index]->m_name.c_str(), selected ) ) app.selectedFaction = index;
        }
        ImGui::EndCombo();
    }

    ImGui::InputText( "Race search", app.raceFilter, std::size( app.raceFilter ) );
    const char* racePreview = app.selectedRace >= 0 ? app.races[app.selectedRace]->m_name.c_str() : "None";
    if( ImGui::BeginCombo( "Race", racePreview ) )
    {
        for( int index = 0; index < static_cast<int>( app.races.size() ); ++index )
        {
            if( !ContainsInsensitive( app.races[index]->m_name, app.raceFilter ) ) continue;
            const bool selected = index == app.selectedRace;
            if( ImGui::Selectable( app.races[index]->m_name.c_str(), selected ) ) app.selectedRace = index;
        }
        ImGui::EndCombo();
    }

    const std::string selectedDna = SelectedDna( app );
    ImGui::TextWrapped( "Selected DNA: %s", selectedDna.empty() ? "<incomplete>" : selectedDna.c_str() );
    if( ImGui::Button( "Load selected real ship" ) && !selectedDna.empty() )
    {
        BuildShipFromDna( app, selectedDna );
    }
    ImGui::SameLine();
    if( ImGui::Button( "Frame ship (R)" ) ) FrameCurrentShip( app );

    ImGui::InputText( "Direct DNA", app.dnaBuffer, std::size( app.dnaBuffer ) );
    if( ImGui::Button( "Load DNA" ) ) BuildShipFromDna( app, app.dnaBuffer );
}

void DrawUi( ViewerApp& app )
{
    ImGui::SetNextWindowPos( ImVec2( 18.0f, 18.0f ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 470.0f, 850.0f ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "NSAMDR Real Ship Viewer" );

    ImGui::TextWrapped( "%s", app.status.c_str() );
    ImGui::Separator();

    if( ImGui::CollapsingHeader( "Real EVE ship and material", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        const char* sofPreview = app.sofCandidates.empty() ? "No SOF candidates" : app.sofCandidates[app.selectedSofCandidate].c_str();
        if( ImGui::BeginCombo( "SOF data resource", sofPreview ) )
        {
            for( int index = 0; index < static_cast<int>( app.sofCandidates.size() ); ++index )
            {
                const bool selected = index == app.selectedSofCandidate;
                if( ImGui::Selectable( app.sofCandidates[index].c_str(), selected ) ) app.selectedSofCandidate = index;
            }
            ImGui::EndCombo();
        }
        if( ImGui::Button( "Load selected SOF catalog" ) && !app.sofCandidates.empty() )
        {
            LoadSofData( app, app.sofCandidates[app.selectedSofCandidate] );
        }
        DrawCatalogList( app );
    }

    if( ImGui::CollapsingHeader( "NSAMDR material hook", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        static const char* modes[] = {
            "0 - Original material",
            "1 - Full NSAMDR",
            "2 - Stretch damage mask",
            "3 - Stochastic reconstruction",
            "4 - Neural residual",
        };
        ImGui::Combo( "Mode", &app.nsamdrMode, modes, static_cast<int>( std::size( modes ) ) );
        if( app.nsamdrMode > 0 ) app.previousEnabledMode = app.nsamdrMode;
        ImGui::SliderFloat( "Strength", &app.nsamdrStrength, 0.0f, 2.0f, "%.2f" );
        ImGui::TextWrapped(
            "The viewer publishes NSAMDRSettings to Trinity's global variable store. "
            "The real ship/material path is active now; the corresponding EVE effect override must consume this variable before modes 1-4 alter the material." );
    }

    if( ImGui::CollapsingHeader( "Studio lighting", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::SliderFloat( "Key intensity", &app.keyIntensity, 0.0f, 4.0f, "%.2f" );
        ImGui::SliderFloat( "Cool fill", &app.fillIntensity, 0.0f, 2.5f, "%.2f" );
        ImGui::SliderFloat( "Reflection / rim", &app.rimIntensity, 0.0f, 2.0f, "%.2f" );
        ImGui::SliderFloat( "Background", &app.backgroundBrightness, 0.0f, 2.0f, "%.2f" );
        ImGui::SliderAngle( "Light yaw", &app.lightYaw, -180.0f, 180.0f );
        ImGui::SliderAngle( "Light pitch", &app.lightPitch, -89.0f, 89.0f );
        ImGui::SliderFloat( "Exposure hook", &app.exposure, 0.25f, 3.0f, "%.2f" );
        if( ImGui::Button( "Studio preset" ) )
        {
            app.keyIntensity = 1.85f;
            app.fillIntensity = 0.82f;
            app.rimIntensity = 0.42f;
            app.backgroundBrightness = 0.65f;
            app.lightYaw = -0.72f;
            app.lightPitch = -0.58f;
        }
        ImGui::SameLine();
        if( ImGui::Button( "Harsh inspection" ) )
        {
            app.keyIntensity = 2.75f;
            app.fillIntensity = 0.30f;
            app.rimIntensity = 0.18f;
            app.backgroundBrightness = 0.25f;
            app.lightYaw = -0.25f;
            app.lightPitch = -0.35f;
        }
    }

    if( ImGui::CollapsingHeader( "Camera" ) )
    {
        ImGui::SliderFloat( "Orbit sensitivity", &app.orbitSensitivity, 0.001f, 0.02f, "%.4f" );
        ImGui::SliderFloat( "Pan sensitivity", &app.panSensitivity, 0.0002f, 0.008f, "%.4f" );
        ImGui::SliderFloat( "Zoom sensitivity", &app.zoomSensitivity, 0.02f, 0.35f, "%.2f" );
        ImGui::Text( "Right drag: orbit" );
        ImGui::Text( "Middle drag: pan" );
        ImGui::Text( "Mouse wheel: zoom" );
        ImGui::Text( "R: frame/reset   Space: NSAMDR off/on" );
    }

    ImGui::End();
}

void HandleKeyboard( ViewerApp& app )
{
    if( GetAsyncKeyState( VK_ESCAPE ) & 1 ) app.running = false;
    if( GetAsyncKeyState( 'R' ) & 1 ) FrameCurrentShip( app );
    if( GetAsyncKeyState( VK_SPACE ) & 1 )
    {
        app.nsamdrMode = app.nsamdrMode == 0 ? std::max( app.previousEnabledMode, 1 ) : 0;
    }
    for( int mode = 0; mode <= 4; ++mode )
    {
        if( GetAsyncKeyState( '0' + mode ) & 1 ) app.nsamdrMode = mode;
    }
}

LRESULT CALLBACK ViewerWindowProc( HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    if( ImGui::GetCurrentContext() != nullptr && ImGui_ImplWin32_WndProcHandler( hwnd, message, wParam, lParam ) )
    {
        return true;
    }

    ViewerApp* app = g_app;
    switch( message )
    {
    case WM_CLOSE:
        if( app ) app->running = false;
        return 0;
    case WM_DESTROY:
        PostQuitMessage( 0 );
        return 0;
    case WM_RBUTTONDOWN:
        if( app && !ImGui::GetIO().WantCaptureMouse )
        {
            app->rightDragging = true;
            app->previousMouse = { GET_X_LPARAM( lParam ), GET_Y_LPARAM( lParam ) };
            SetCapture( hwnd );
        }
        return 0;
    case WM_MBUTTONDOWN:
        if( app && !ImGui::GetIO().WantCaptureMouse )
        {
            app->middleDragging = true;
            app->previousMouse = { GET_X_LPARAM( lParam ), GET_Y_LPARAM( lParam ) };
            SetCapture( hwnd );
        }
        return 0;
    case WM_RBUTTONUP:
        if( app ) app->rightDragging = false;
        ReleaseCapture();
        return 0;
    case WM_MBUTTONUP:
        if( app ) app->middleDragging = false;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if( app && ( app->rightDragging || app->middleDragging ) )
        {
            const POINT current = { GET_X_LPARAM( lParam ), GET_Y_LPARAM( lParam ) };
            const float dx = static_cast<float>( current.x - app->previousMouse.x );
            const float dy = static_cast<float>( current.y - app->previousMouse.y );
            app->previousMouse = current;
            if( app->rightDragging )
            {
                app->cameraYaw += dx * app->orbitSensitivity;
                app->cameraPitch = std::clamp( app->cameraPitch - dy * app->orbitSensitivity, -1.52f, 1.52f );
            }
            if( app->middleDragging ) PanCamera( *app, dx, dy );
        }
        return 0;
    case WM_MOUSEWHEEL:
        if( app && !ImGui::GetIO().WantCaptureMouse )
        {
            const float notches = static_cast<float>( GET_WHEEL_DELTA_WPARAM( wParam ) ) / static_cast<float>( WHEEL_DELTA );
            app->cameraDistance *= std::exp( -notches * app->zoomSensitivity );
            app->cameraDistance = std::clamp( app->cameraDistance, std::max( 0.05f * app->framedRadius, 0.2f ), std::max( app->framedRadius * 100.0f, 1000.0f ) );
        }
        return 0;
    default:
        break;
    }
    return DefWindowProcW( hwnd, message, wParam, lParam );
}

HWND CreateViewerWindow()
{
    const wchar_t* className = L"NSAMDRRealShipViewerWindow";
    WNDCLASSEXW windowClass = {};
    windowClass.cbSize = sizeof( windowClass );
    windowClass.style = CS_OWNDC;
    windowClass.lpfnWndProc = ViewerWindowProc;
    windowClass.hInstance = GetModuleHandleW( nullptr );
    windowClass.hCursor = LoadCursorW( nullptr, IDC_ARROW );
    windowClass.lpszClassName = className;
    RegisterClassExW( &windowClass );

    RECT rect = { 0, 0, kWindowWidth, kWindowHeight };
    const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    AdjustWindowRect( &rect, style, FALSE );
    return CreateWindowExW(
        0,
        className,
        L"NSAMDR Real EVE Ship Material Viewer - Trinity DX11",
        style,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        windowClass.hInstance,
        nullptr );
}

bool ConfigureResourceCache( ViewerApp& app, const std::wstring& sharedCache, const std::wstring& indexPath )
{
    if( !ReadBinaryTextFile( indexPath, app.resourceIndexContents ) )
    {
        app.status = "Could not read resfileindex.txt";
        return false;
    }

    std::wstring cacheFolder = sharedCache;
    if( !cacheFolder.empty() && cacheFolder.back() != L'\\' && cacheFolder.back() != L'/' ) cacheFolder += L'\\';
    cacheFolder += L"ResFiles";

    BeRemoteFileCache->SetCacheFolder( cacheFolder.c_str() );
    BeRemoteFileCache->AddFileIndex( app.resourceIndexContents );

    BluePaths* paths = static_cast<BluePaths*>( BlueGetBluePaths() );
    if( !paths->IsFileSystemRegistered( "Remote" ) )
    {
        const Be::Result<std::string> result = paths->RegisterFileSystemBeforeLocal( "Remote" );
        if( !BeIsSuccess( result ) )
        {
            app.status = "Could not register Blue remote resource filesystem: " + result.value;
            return false;
        }
    }

    app.sofCandidates = FindSofCandidates( app.resourceIndexContents );
    if( app.sofCandidates.empty() )
    {
        app.status = "SharedCache mounted, but no SOF data candidates were found in the resource index";
        return false;
    }
    app.status = "EVE SharedCache mounted; found " + std::to_string( app.sofCandidates.size() ) + " SOF candidates";
    return true;
}

bool InitializeBlueAndTrinity( std::string& error )
{
    if( !InitializePython( error ) )
    {
        return false;
    }

    BlueModuleStartup();
    if( !BlueInitializePaths( L"." ) )
    {
        error = "BlueInitializePaths failed";
        return false;
    }
    if( !BlueInitializeResourceLoading() )
    {
        error = "BlueInitializeResourceLoading failed";
        return false;
    }
    if( BlueGetBeOS() == nullptr || !BlueGetBeOS()->Startup( 0 ) )
    {
        error = "Blue BeOS startup failed";
        return false;
    }

    BeClasses->RegisterClasses( BlueRegistration::GetClassRegs() );
    InitializeTrinity();
    return true;
}

bool InitializeRenderPipeline( ViewerApp& app )
{
    gTriDev.CreateInstance();
    if( !gTriDev || !gTriDev->CreateSimpleDevice(
            app.hwnd,
            kWindowWidth,
            kWindowHeight,
            TriDevice::WINDOWED,
            Tr2RenderContextEnum::PRESENT_INTERVAL_ONE ) )
    {
        app.status = "TriDevice::CreateSimpleDevice failed";
        return false;
    }

    app.scene = new PreviewScene();
    app.scene->Initialize();
    app.scene->ConfigureLighting(
        app.keyIntensity,
        app.fillIntensity,
        app.rimIntensity,
        app.backgroundBrightness,
        app.lightYaw,
        app.lightPitch );

    app.view.CreateInstance();
    app.projection.CreateInstance();
    app.renderJobs.CreateInstance();
    app.renderJob.CreateInstance();
    app.clearStep.CreateInstance();
    app.viewStep.CreateInstance();
    app.projectionStep.CreateInstance();
    app.sceneStep.CreateInstance();

    app.clearStep->m_color = Color( 0.004f, 0.010f, 0.022f, 1.0f );
    app.clearStep->m_depth = 1.0f;
    app.clearStep->m_stencil = 0;
    app.clearStep->m_isColorCleared = true;
    app.clearStep->m_isDepthCleared = true;
    app.clearStep->m_isStencilCleared = true;
    app.viewStep->SetViewCameraParent( app.view, nullptr );
    app.projectionStep->SetProjection( app.projection );
    app.sceneStep->m_scene = app.scene;

    app.renderJob->Steps().Insert( -1, app.clearStep->GetRawRoot() );
    app.renderJob->Steps().Insert( -1, app.viewStep->GetRawRoot() );
    app.renderJob->Steps().Insert( -1, app.projectionStep->GetRawRoot() );
    app.renderJob->Steps().Insert( -1, app.sceneStep->GetRawRoot() );
    app.renderJobs->m_scheduledRecurring.Insert( -1, app.renderJob->GetRawRoot() );
    gTriDev->SetRenderJobs( app.renderJobs );

    UpdateCamera( app );

    USE_MAIN_THREAD_RENDER_CONTEXT();
    ID3D11Device* device = renderContext.m_d3dDevice11;
    ID3D11DeviceContext* context = renderContext.m_context;
    IDXGISwapChain* swapChain = renderContext.m_swapChain;
    if( device == nullptr || context == nullptr || swapChain == nullptr )
    {
        app.status = "Trinity DX11 native handles are unavailable";
        return false;
    }

    ComPtr<ID3D11Texture2D> backBuffer;
    if( FAILED( swapChain->GetBuffer( 0, IID_PPV_ARGS( &backBuffer ) ) ) ||
        FAILED( device->CreateRenderTargetView( backBuffer.Get(), nullptr, &app.imguiRenderTarget ) ) )
    {
        app.status = "Could not create ImGui backbuffer render target";
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init( app.hwnd );
    ImGui_ImplDX11_Init( device, context );

    app.nsamdrSettings.Register( "NSAMDRSettings", Vector4( 0.0f, 1.0f, 1.0f, 0.0f ) );
    return true;
}

void ShutdownViewer( ViewerApp& app )
{
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    if( ImGui::GetCurrentContext() != nullptr ) ImGui::DestroyContext();

    ClearSceneObjects( app );
    if( gTriDev )
    {
        gTriDev->SetRenderJobs( nullptr );
        gTriDev->InvalidateAndUnregisterForTicks();
        gTriDev = nullptr;
    }

    app.scene = nullptr;
}

int RunViewer( int argc, wchar_t** argv )
{
    ViewerApp app;
    g_app = &app;

    std::string error;
    if( !InitializeBlueAndTrinity( error ) )
    {
        MessageBoxA( nullptr, error.c_str(), "NSAMDR initialisation error", MB_ICONERROR | MB_OK );
        return 10;
    }

    std::wstring sharedCache = GetArgument( argc, argv, L"--shared-cache" );
    std::wstring indexPath = GetArgument( argc, argv, L"--res-index" );
    if( sharedCache.empty() ) sharedCache = GetEnvironmentWide( L"NSAMDR_EVE_SHARED_CACHE" );
    if( indexPath.empty() ) indexPath = GetEnvironmentWide( L"NSAMDR_EVE_RES_INDEX" );
    if( sharedCache.empty() || indexPath.empty() )
    {
        MessageBoxW( nullptr, L"Missing --shared-cache or --res-index. Run through scripts\\build\\run_nsamdr_realship_dx11.bat.", L"NSAMDR resource error", MB_ICONERROR | MB_OK );
        return 11;
    }

    app.hwnd = CreateViewerWindow();
    if( app.hwnd == nullptr )
    {
        MessageBoxW( nullptr, L"Could not create the preview window.", L"NSAMDR window error", MB_ICONERROR | MB_OK );
        return 12;
    }
    ShowWindow( app.hwnd, SW_SHOWDEFAULT );
    UpdateWindow( app.hwnd );

    if( !InitializeRenderPipeline( app ) )
    {
        MessageBoxA( app.hwnd, app.status.c_str(), "NSAMDR rendering error", MB_ICONERROR | MB_OK );
        return 13;
    }

    ConfigureResourceCache( app, sharedCache, indexPath );
    for( int index = 0; index < static_cast<int>( app.sofCandidates.size() ); ++index )
    {
        app.selectedSofCandidate = index;
        if( LoadSofData( app, app.sofCandidates[index] ) )
        {
            break;
        }
    }

    const auto start = std::chrono::steady_clock::now();
    while( app.running )
    {
        MSG message;
        while( PeekMessageW( &message, nullptr, 0, 0, PM_REMOVE ) )
        {
            if( message.message == WM_QUIT ) app.running = false;
            TranslateMessage( &message );
            DispatchMessageW( &message );
        }
        if( !app.running ) break;

        HandleKeyboard( app );
        UpdateCamera( app );
        app.scene->ConfigureLighting(
            app.keyIntensity,
            app.fillIntensity,
            app.rimIntensity,
            app.backgroundBrightness,
            app.lightYaw,
            app.lightPitch );
        app.clearStep->m_color = Color(
            0.006f * app.backgroundBrightness,
            0.014f * app.backgroundBrightness,
            0.030f * app.backgroundBrightness,
            1.0f );
        app.nsamdrSettings = Vector4( static_cast<float>( app.nsamdrMode ), app.nsamdrStrength, app.exposure, 0.0f );

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        DrawUi( app );

        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>( now - start ).count();
        const Be::Time time = static_cast<Be::Time>( seconds * 10000000.0 );
        app.scene->Update( time, time );
        gTriDev->OnTick( time, time, nullptr );

        ImGui::Render();
        USE_MAIN_THREAD_RENDER_CONTEXT();
        ID3D11DeviceContext* context = renderContext.m_context;
        ID3D11RenderTargetView* renderTarget = app.imguiRenderTarget.Get();
        context->OMSetRenderTargets( 1, &renderTarget, nullptr );
        ImGui_ImplDX11_RenderDrawData( ImGui::GetDrawData() );
    }

    ShutdownViewer( app );
    DestroyWindow( app.hwnd );
    g_app = nullptr;
    return 0;
}

} // namespace

extern "C" NSAMDR_VIEWER_API int NSAMDR_RunRealShipViewer( int argc, wchar_t** argv )
{
    return RunViewer( argc, argv );
}

#else

extern "C" NSAMDR_VIEWER_API int NSAMDR_RunRealShipViewer( int, wchar_t** )
{
    return 1;
}

#endif
