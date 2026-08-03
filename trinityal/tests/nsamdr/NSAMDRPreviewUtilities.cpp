#include "StdAfx.h"
#include "NSAMDRPreviewUtilities.h"

namespace nsamdr
{
float ClampFloat(float value, float minimum, float maximum)
{
    return std::max(minimum, std::min(maximum, value));
}

XMFLOAT3 Add3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(a.x + b.x, a.y + b.y, a.z + b.z);
}

XMFLOAT3 Subtract3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(a.x - b.x, a.y - b.y, a.z - b.z);
}

XMFLOAT3 Multiply3(const XMFLOAT3& value, float scalar)
{
    return XMFLOAT3(value.x * scalar, value.y * scalar, value.z * scalar);
}

float Dot3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

XMFLOAT3 Cross3(const XMFLOAT3& a, const XMFLOAT3& b)
{
    return XMFLOAT3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

float Length3(const XMFLOAT3& value)
{
    return std::sqrt(std::max(Dot3(value, value), 0.0f));
}

XMFLOAT3 Normalize3(const XMFLOAT3& value)
{
    const float length = Length3(value);
    if (length <= 1.0e-8f)
    {
        return XMFLOAT3(0.0f, 1.0f, 0.0f);
    }
    return Multiply3(value, 1.0f / length);
}

const EnvironmentGpu* SelectedEnvironment(const PreviewResources& resources, const PreviewState& state)
{
    if (resources.environments.empty()) return nullptr;
    const size_t index = std::min<size_t>(state.environmentIndex, resources.environments.size() - 1U);
    return &resources.environments[index];
}

std::string GetEnvironmentString(const char* name)
{
    const DWORD required = GetEnvironmentVariableA(name, nullptr, 0);
    if (required == 0)
    {
        return {};
    }

    std::vector<char> buffer(required, '\0');
    if (GetEnvironmentVariableA(name, buffer.data(), required) == 0)
    {
        return {};
    }
    return std::string(buffer.data());
}

std::wstring ToWidePath(const std::string& path)
{
    if (path.empty())
    {
        return {};
    }

    UINT codePage = CP_UTF8;
    DWORD flags = MB_ERR_INVALID_CHARS;
    int count = MultiByteToWideChar(codePage, flags, path.c_str(), -1, nullptr, 0);
    if (count <= 0)
    {
        codePage = CP_ACP;
        flags = 0;
        count = MultiByteToWideChar(codePage, flags, path.c_str(), -1, nullptr, 0);
    }
    if (count <= 0)
    {
        return {};
    }

    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(codePage, flags, path.c_str(), -1, &result[0], count);
    if (!result.empty() && result.back() == L'\0')
    {
        result.pop_back();
    }
    return result;
}


std::string ToLowerAscii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::vector<std::string> SplitDelimited(const std::string& value, char delimiter)
{
    std::vector<std::string> result;
    std::string part;
    std::istringstream stream(value);
    while (std::getline(stream, part, delimiter))
    {
        result.push_back(part);
    }
    if (!value.empty() && value.back() == delimiter) result.emplace_back();
    return result;
}

std::string FileLabel(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1U);
}

std::vector<std::string> GetEnvironmentPaths()
{
    std::vector<std::string> paths;
    const std::string list = GetEnvironmentString("NSAMDR_ENVIRONMENTS");
    for (const std::string& value : SplitDelimited(list, ';'))
    {
        if (!value.empty() && std::find(paths.begin(), paths.end(), value) == paths.end()) paths.push_back(value);
    }
    const std::string primary = GetEnvironmentString("NSAMDR_ENVIRONMENT");
    if (!primary.empty() && std::find(paths.begin(), paths.end(), primary) == paths.end()) paths.insert(paths.begin(), primary);
    return paths;
}

std::string CatalogSelectionAsset(const ShipCatalog& catalog, const CatalogSelection& selection)
{
    if (selection.entryIndex >= catalog.entries.size()) return {};
    const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
    if (selection.variantIndex >= 0 && static_cast<size_t>(selection.variantIndex) < entry.variants.size())
    {
        return entry.variants[static_cast<size_t>(selection.variantIndex)];
    }
    return entry.preferredAsset;
}

std::string CatalogSelectionLabel(const ShipCatalog& catalog, const CatalogSelection& selection)
{
    if (selection.entryIndex >= catalog.entries.size()) return {};
    const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
    if (selection.variantIndex < 0) return entry.displayName;
    const std::string asset = CatalogSelectionAsset(catalog, selection);
    const size_t separator = asset.find_last_of("/\\");
    return entry.displayName + " - " + (separator == std::string::npos ? asset : asset.substr(separator + 1));
}

void RebuildCatalogFilter(ShipCatalog& catalog)
{
    std::string previousAsset;
    if (catalog.selectedFilteredIndex >= 0 &&
        static_cast<size_t>(catalog.selectedFilteredIndex) < catalog.filtered.size())
    {
        previousAsset = CatalogSelectionAsset(
            catalog,
            catalog.filtered[static_cast<size_t>(catalog.selectedFilteredIndex)]);
    }

    catalog.filtered.clear();
    const std::string search = ToLowerAscii(std::string(catalog.search.data()));
    for (size_t entryIndex = 0; entryIndex < catalog.entries.size(); ++entryIndex)
    {
        const ShipCatalogEntry& entry = catalog.entries[entryIndex];
        std::string searchable = entry.displayName + " " + entry.groupName + " " +
            entry.factionName + " " + entry.typeId + " " + entry.canonicalKey + " " + entry.preferredAsset;
        for (const std::string& variant : entry.variants) searchable += " " + variant;
        if (!search.empty() && ToLowerAscii(searchable).find(search) == std::string::npos) continue;

        if (catalog.showRawVariants && !entry.variants.empty())
        {
            for (size_t variantIndex = 0; variantIndex < entry.variants.size(); ++variantIndex)
            {
                catalog.filtered.push_back({entryIndex, static_cast<int>(variantIndex)});
            }
        }
        else
        {
            catalog.filtered.push_back({entryIndex, -1});
        }
    }

    catalog.selectedFilteredIndex = catalog.filtered.empty() ? -1 : 0;
    const std::string wanted = !previousAsset.empty() ? previousAsset : catalog.currentQuery;
    if (!wanted.empty())
    {
        const std::string loweredWanted = ToLowerAscii(wanted);
        for (size_t filteredIndex = 0; filteredIndex < catalog.filtered.size(); ++filteredIndex)
        {
            const CatalogSelection& selection = catalog.filtered[filteredIndex];
            const ShipCatalogEntry& entry = catalog.entries[selection.entryIndex];
            const std::string asset = CatalogSelectionAsset(catalog, selection);
            bool matches = ToLowerAscii(asset) == loweredWanted ||
                ToLowerAscii(entry.canonicalKey) == loweredWanted;
            if (!matches && selection.variantIndex < 0)
            {
                matches = ToLowerAscii(entry.preferredAsset) == loweredWanted ||
                    std::any_of(entry.variants.begin(), entry.variants.end(), [&](const std::string& variant) {
                        return ToLowerAscii(variant) == loweredWanted;
                    });
            }
            if (matches)
            {
                catalog.selectedFilteredIndex = static_cast<int>(filteredIndex);
                break;
            }
        }
    }
}

bool LoadShipCatalog(const std::string& path, const std::string& currentQuery, ShipCatalog& catalog)
{
    catalog = ShipCatalog{};
    catalog.currentQuery = currentQuery;
    if (path.empty())
    {
        catalog.status = "Ship catalog is unavailable. Launch through the real EVE asset test script.";
        return false;
    }
    std::ifstream input(path);
    if (!input)
    {
        catalog.status = "Could not open ship catalog: " + path;
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = SplitDelimited(line, '\t');
        ShipCatalogEntry entry;
        if (fields.size() >= 7)
        {
            entry.displayName = fields[0];
            entry.groupName = fields[1];
            entry.factionName = fields[2];
            entry.typeId = fields[3] == "0" ? "" : fields[3];
            entry.canonicalKey = fields[4];
            entry.preferredAsset = fields[5];
            entry.variants = SplitDelimited(fields[6], '|');
            entry.variants.erase(
                std::remove_if(entry.variants.begin(), entry.variants.end(), [](const std::string& value) { return value.empty(); }),
                entry.variants.end());
        }
        else
        {
            // Legacy v1 catalog: one raw resource path per line.
            entry.displayName = line;
            entry.groupName = "Unmapped ship asset";
            entry.canonicalKey = line;
            entry.preferredAsset = line;
            entry.variants.push_back(line);
        }
        if (entry.displayName.empty()) entry.displayName = entry.preferredAsset;
        if (entry.preferredAsset.empty() && !entry.variants.empty()) entry.preferredAsset = entry.variants.front();
        if (entry.preferredAsset.empty()) continue;
        if (std::find(entry.variants.begin(), entry.variants.end(), entry.preferredAsset) == entry.variants.end())
        {
            entry.variants.insert(entry.variants.begin(), entry.preferredAsset);
        }
        catalog.entries.push_back(std::move(entry));
    }

    RebuildCatalogFilter(catalog);
    catalog.status = catalog.entries.empty()
        ? "No ship GR2 resources were found in the EVE cache index."
        : "Select a named ship. The highest-detail preferred asset is used unless raw variants are shown.";
    return !catalog.entries.empty();
}

std::wstring QuoteWindowsArgument(const std::wstring& value)
{
    std::wstring quoted = L"\"";
    for (wchar_t character : value)
    {
        if (character == L'\"') quoted += L'\\';
        quoted += character;
    }
    quoted += L"\"";
    return quoted;
}

bool LaunchCachedShip(const std::string& logicalAsset, const std::string& selectionKey, std::string& error)
{
    const std::string python = GetEnvironmentString("NSAMDR_PYTHON_EXE");
    const std::string tool = GetEnvironmentString("NSAMDR_EVE_TOOL");
    const std::string repoRoot = GetEnvironmentString("NSAMDR_EVE_REPO_ROOT");
    const std::string cacheRoot = GetEnvironmentString("NSAMDR_EVE_CACHE");
    const std::string launcher = GetEnvironmentString("NSAMDR_EVE_LAUNCHER");
    if (python.empty() || tool.empty() || repoRoot.empty() || cacheRoot.empty() || launcher.empty())
    {
        error = "The current viewer was not launched with the EVE cache selection environment.";
        return false;
    }

    const std::wstring command =
        QuoteWindowsArgument(ToWidePath(python)) + L" " +
        QuoteWindowsArgument(ToWidePath(tool)) + L" prepare-run --repo-root " +
        QuoteWindowsArgument(ToWidePath(repoRoot)) + L" --shared-cache " +
        QuoteWindowsArgument(ToWidePath(cacheRoot)) + L" --query " +
        QuoteWindowsArgument(ToWidePath(logicalAsset)) + L" --selection-key " +
        QuoteWindowsArgument(ToWidePath(selectionKey)) + L" --launcher " +
        QuoteWindowsArgument(ToWidePath(launcher));

    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const std::wstring workingDirectory = ToWidePath(repoRoot);
    const BOOL launched = CreateProcessW(
        nullptr,
        mutableCommand.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NEW_CONSOLE,
        nullptr,
        workingDirectory.empty() ? nullptr : workingDirectory.c_str(),
        &startup,
        &process);
    if (!launched)
    {
        error = "Could not launch the selected-ship converter. Windows error " + std::to_string(GetLastError()) + ".";
        return false;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return true;
}

} // namespace nsamdr
