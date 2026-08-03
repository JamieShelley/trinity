#pragma once

#include "NSAMDRPreviewTypes.h"

namespace nsamdr
{
float ClampFloat(float value, float minimum, float maximum);
XMFLOAT3 Add3(const XMFLOAT3& a, const XMFLOAT3& b);
XMFLOAT3 Subtract3(const XMFLOAT3& a, const XMFLOAT3& b);
XMFLOAT3 Multiply3(const XMFLOAT3& value, float scalar);
float Dot3(const XMFLOAT3& a, const XMFLOAT3& b);
XMFLOAT3 Cross3(const XMFLOAT3& a, const XMFLOAT3& b);
float Length3(const XMFLOAT3& value);
XMFLOAT3 Normalize3(const XMFLOAT3& value);
const EnvironmentGpu* SelectedEnvironment(const PreviewResources& resources, const PreviewState& state);
std::string GetEnvironmentString(const char* name);
std::wstring ToWidePath(const std::string& path);
std::string ToLowerAscii(std::string value);
std::vector<std::string> SplitDelimited(const std::string& value, char delimiter);
std::string FileLabel(const std::string& path);
std::vector<std::string> GetEnvironmentPaths();
std::string CatalogSelectionAsset(const ShipCatalog& catalog, const CatalogSelection& selection);
std::string CatalogSelectionLabel(const ShipCatalog& catalog, const CatalogSelection& selection);
void RebuildCatalogFilter(ShipCatalog& catalog);
bool LoadShipCatalog(const std::string& path, const std::string& currentQuery, ShipCatalog& catalog);
std::wstring QuoteWindowsArgument(const std::wstring& value);
bool LaunchCachedShip(const std::string& logicalAsset, const std::string& selectionKey, std::string& error);

} // namespace nsamdr
