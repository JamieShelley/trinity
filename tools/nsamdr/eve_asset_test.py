from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_RAVEN = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"


@dataclass(frozen=True)
class ResourceRow:
    logical: str
    hashed: str
    index_file: str


@dataclass(frozen=True)
class ShipCatalogEntry:
    display_name: str
    type_id: int | None
    group_name: str
    faction_name: str
    canonical_key: str
    preferred_asset: str
    variants: tuple[str, ...]


SDE_LATEST_URL = "https://developers.eveonline.com/static-data/eve-online-static-data-latest-jsonl.zip"
SDE_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _add_candidate(values: list[Path], value: str | Path | None) -> None:
    if value is None:
        return
    text = str(value).strip().strip('"')
    if text:
        values.append(Path(text))


def _registry_cache_roots() -> Iterable[Path]:
    """Yield legacy/current CCP cache paths from the Windows registry when present."""
    if os.name != "nt":
        return
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return

    key_paths = [
        r"Software\CCP\EVEONLINE",
        r"Software\WOW6432Node\CCP\EVEONLINE",
        r"Software\CCP\EVE",
    ]
    hives = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
    for hive in hives:
        for key_path in key_paths:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    index = 0
                    while True:
                        try:
                            name, value, kind = winreg.EnumValue(key, index)
                        except OSError:
                            break
                        index += 1
                        if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
                            continue
                        if not isinstance(value, str) or not value.strip():
                            continue
                        lowered = f"{name} {value}".lower()
                        if any(token in lowered for token in ("cache", "eve", "game")):
                            yield Path(os.path.expandvars(value.strip().strip('"')))
            except OSError:
                continue



def _read_registry_string(key, name: str):
    try:
        import winreg  # type: ignore[import-not-found]
        value, kind = winreg.QueryValueEx(key, name)
    except (ImportError, OSError):
        return None
    if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) or not isinstance(value, str):
        return None
    value = os.path.expandvars(value.strip().strip('"'))
    return value or None


def _installed_program_roots() -> Iterable[Path]:
    """Find EVE/CCP launcher locations through Windows Installed Apps registry data."""
    if os.name != "nt":
        return
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return

    uninstall_roots = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    ]
    hives = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
    views = [0, getattr(winreg, "KEY_WOW64_64KEY", 0), getattr(winreg, "KEY_WOW64_32KEY", 0)]
    yielded: set[str] = set()

    def emit(raw: str | None):
        if not raw:
            return
        value = raw.strip().strip('"')
        # DisplayIcon and uninstall strings may include arguments or a trailing ,0.
        match = re.match(r'^"([^"]+)"', value)
        if match:
            value = match.group(1)
        else:
            value = value.split(",", 1)[0]
            if ".exe " in value.lower():
                value = value[: value.lower().find(".exe ") + 4]
        path = Path(os.path.expandvars(value))
        if path.suffix.lower() == ".exe":
            path = path.parent
        key = os.path.normcase(os.path.normpath(str(path)))
        if key and key not in yielded:
            yielded.add(key)
            return path
        return None

    for hive in hives:
        for root_path in uninstall_roots:
            for view in views:
                try:
                    with winreg.OpenKey(hive, root_path, 0, winreg.KEY_READ | view) as root:
                        count = winreg.QueryInfoKey(root)[0]
                        for index in range(count):
                            try:
                                sub_name = winreg.EnumKey(root, index)
                                with winreg.OpenKey(root, sub_name) as sub:
                                    display_name = (_read_registry_string(sub, "DisplayName") or "").lower()
                                    publisher = (_read_registry_string(sub, "Publisher") or "").lower()
                                    combined = f"{display_name} {publisher} {sub_name.lower()}"
                                    if not (
                                        "eve online" in combined
                                        or "eve launcher" in combined
                                        or ("ccp" in combined and "eve" in combined)
                                    ):
                                        continue
                                    for value_name in ("InstallLocation", "InstallSource", "DisplayIcon", "UninstallString"):
                                        candidate = emit(_read_registry_string(sub, value_name))
                                        if candidate:
                                            yield candidate
                            except OSError:
                                continue
                except OSError:
                    continue

    app_path_roots = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\eve-online.exe",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\evelauncher.exe",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\eve.exe",
    ]
    for hive in hives:
        for key_path in app_path_roots:
            for view in views:
                try:
                    with winreg.OpenKey(hive, key_path, 0, winreg.KEY_READ | view) as key:
                        for name in ("", "Path"):
                            candidate = emit(_read_registry_string(key, name))
                            if candidate:
                                yield candidate
                except OSError:
                    continue


def _launcher_config_cache_roots() -> Iterable[Path]:
    """Find SharedCache paths embedded in EVE launcher JSON/YAML/config files."""
    roots: list[Path] = []
    for base in (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMDATA")):
        if not base:
            continue
        base_path = Path(base)
        roots.extend([
            base_path / "eve-online",
            base_path / "EVE Online",
            base_path / "CCP" / "EVE",
            base_path / "CCP" / "EVE Online",
        ])

    # Require either an explicit SharedCache component or nearby cache/game key text.
    quoted_path = re.compile(r'["\']([A-Za-z]:[\\/][^"\'\r\n]+)["\']')
    shared_path = re.compile(r'([A-Za-z]:[\\/][^\r\n"\']*SharedCache[^\r\n"\']*)', re.IGNORECASE)
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for suffix in ("*.json", "*.yaml", "*.yml", "*.conf", "*.config", "*.ini"):
                files.extend(path for path in root.rglob(suffix) if path.is_file())
        except OSError:
            continue
    files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)

    for config_path in files[:120]:
        try:
            text = config_path.read_text(encoding="utf-8", errors="replace")[:4 * 1024 * 1024]
        except OSError:
            continue
        for match in shared_path.finditer(text):
            yield Path(match.group(1).strip().rstrip(" .,)];}"))
        lowered = text.lower()
        if any(token in lowered for token in ("sharedcache", "shared_cache", "gamefiles", "game directory", "cachepath")):
            for match in quoted_path.finditer(text):
                value = match.group(1).replace("\\\\", "\\")
                if any(token in value.lower() for token in ("eve", "sharedcache", "resfiles")):
                    yield Path(value)


def _launcher_log_cache_roots() -> Iterable[Path]:
    """Extract SharedCache locations recorded by recent EVE launcher logs."""
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA")
    local = os.environ.get("LOCALAPPDATA")
    log_roots: list[Path] = []
    for base in (appdata, local):
        if not base:
            continue
        base_path = Path(base)
        log_roots.extend([
            base_path / "eve-online" / "logs",
            base_path / "EVE Online" / "logs",
            base_path / "CCP" / "EVE" / "launcher" / "logs",
            base_path / "CCP" / "EVE" / "logs",
        ])

    # Match Windows paths around launcher messages such as:
    # Shared cache in “S:\games\EVE\SharedCache”
    path_pattern = re.compile(
        r"(?:shared[ -]?cache|game files|cache directory)[^\r\n]{0,180}?"
        r"[\"'“‘]?([A-Za-z]:[\\/][^\"'”’\r\n]+)",
        re.IGNORECASE,
    )
    files: list[Path] = []
    for root in log_roots:
        if not root.is_dir():
            continue
        try:
            files.extend(path for path in root.rglob("*.log") if path.is_file())
            files.extend(path for path in root.rglob("*.txt") if path.is_file())
        except OSError:
            continue
    files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)

    for log_path in files[:30]:
        try:
            # Launcher logs are small enough in normal use; cap reads to 4 MiB.
            with log_path.open("rb") as handle:
                data = handle.read(4 * 1024 * 1024)
            text = data.decode("utf-8", errors="replace")
        except OSError:
            continue
        for match in path_pattern.finditer(text):
            raw = match.group(1).strip().rstrip(" .,)];}")
            roots.append(Path(raw))
    yield from roots



def _saved_cache_roots() -> Iterable[Path]:
    """Reuse a previously verified cache location without requiring user input."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        saved = Path(local) / "NSAMDR" / "eve_shared_cache.txt"
        try:
            value = saved.read_text(encoding="utf-8").strip()
            if value:
                yield Path(value)
        except OSError:
            pass

    try:
        repo_root = Path(__file__).resolve().parents[2]
        manifests = sorted(
            (repo_root / "artifacts" / "nsamdr" / "eve_assets").glob("*/asset_manifest.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        for manifest in manifests[:20]:
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                value = data.get("cacheRoot")
                if isinstance(value, str) and value.strip():
                    yield Path(value)
            except (OSError, ValueError, TypeError):
                continue
    except (OSError, IndexError):
        pass


def _save_cache_root(root: Path) -> None:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return
    try:
        path = Path(local) / "NSAMDR" / "eve_shared_cache.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(root) + "\n", encoding="utf-8")
    except OSError:
        pass


def candidate_cache_roots() -> Iterable[Path]:
    seen: set[str] = set()
    values: list[Path] = []

    _add_candidate(values, os.environ.get("EVE_SHARED_CACHE"))
    for value in _saved_cache_roots():
        values.append(value)

    home = Path.home()
    local = os.environ.get("LOCALAPPDATA")
    program_data = os.environ.get("PROGRAMDATA")
    public = os.environ.get("PUBLIC")

    if local:
        values.extend([
            Path(local) / "CCP" / "EVE" / "SharedCache",
            Path(local) / "EVE Online" / "SharedCache",
        ])
    if program_data:
        values.extend([
            Path(program_data) / "CCP" / "EVE" / "SharedCache",
            Path(program_data) / "CCP" / "EVE",
        ])
    if public:
        values.append(Path(public) / "Documents" / "EVE" / "SharedCache")

    values.extend([
        home / "Documents" / "EVE" / "SharedCache",
        home / "AppData" / "Local" / "CCP" / "EVE" / "SharedCache",
    ])

    for value in _registry_cache_roots() or ():
        values.append(value)
    for value in _installed_program_roots() or ():
        values.append(value)
    for value in _launcher_config_cache_roots():
        values.append(value)
    for value in _launcher_log_cache_roots():
        values.append(value)

    for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        values.extend([
            Path(f"{drive}:/EVE/SharedCache"),
            Path(f"{drive}:/EVE"),
            Path(f"{drive}:/CCP/EVE/SharedCache"),
            Path(f"{drive}:/CCP/EVE"),
            Path(f"{drive}:/Games/EVE/SharedCache"),
            Path(f"{drive}:/Games/EVE"),
            Path(f"{drive}:/Games/EVE Online/SharedCache"),
            Path(f"{drive}:/Games/EVE Online"),
            Path(f"{drive}:/SteamLibrary/steamapps/common/EVE Online/SharedCache"),
            Path(f"{drive}:/SteamLibrary/steamapps/common/EVE Online"),
            Path(f"{drive}:/Program Files/CCP/EVE/SharedCache"),
            Path(f"{drive}:/Program Files (x86)/CCP/EVE/SharedCache"),
        ])

    for value in values:
        key = os.path.normcase(os.path.normpath(str(value)))
        if key not in seen:
            seen.add(key)
            yield value


def _layout_variants(candidate: Path) -> Iterable[Path]:
    """Try a selected folder, its parents, and common SharedCache children."""
    seen: set[str] = set()
    current = candidate.expanduser()
    bases = [current]
    try:
        bases.extend(list(current.parents)[:6])
    except (OSError, RuntimeError):
        pass

    for base in bases:
        variants = [base]
        if base.name.lower() != "sharedcache":
            variants.extend([
                base / "SharedCache",
                base / "EVE" / "SharedCache",
                base / "EVE Online" / "SharedCache",
                base / "game" / "SharedCache",
            ])
        # Installed-app roots sometimes contain a versioned launcher folder with
        # the game cache one or two levels below it. Keep this bounded and never
        # recursively scan a drive root.
        if base.is_dir() and len(base.parts) > 2:
            for pattern in ("*/SharedCache", "*/*/SharedCache"):
                try:
                    variants.extend(base.glob(pattern))
                except OSError:
                    pass
        for value in variants:
            key = os.path.normcase(os.path.normpath(str(value)))
            if key not in seen:
                seen.add(key)
                yield value


def _inspect_layout(root: Path) -> tuple[Path, list[Path], Path] | None:
    try:
        resolved = root.resolve() if root.exists() else root
    except OSError:
        resolved = root
    resfiles = resolved / "ResFiles"
    tq = resolved / "tq"
    indexes = [
        tq / "resfileindex.txt",
        tq / "resfileindex_prefetch.txt",
        tq / "EVE.app" / "Contents" / "Resources" / "build" / "resfileindex.txt",
        tq / "EVE.app" / "Contents" / "Resources" / "build" / "resfileindex_prefetch.txt",
    ]
    existing = [path for path in indexes if path.is_file()]
    if resfiles.is_dir() and existing:
        return resolved, existing, resfiles
    return None


def _prompt_for_cache() -> Path | None:
    """Open a folder picker as a last-resort interactive fallback."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Select EVE game files folder (contains ResFiles and tq)",
            mustexist=True,
        )
        root.destroy()
    except Exception:
        return None
    return Path(selected) if selected else None


def resolve_layout(raw: str | None, *, allow_prompt: bool = False) -> tuple[Path, list[Path], Path]:
    candidates: list[Path] = []
    if raw and raw.strip():
        candidates.append(Path(raw.strip().strip('"')))
    candidates.extend(candidate_cache_roots())

    checked: list[str] = []
    seen_checked: set[str] = set()

    def try_candidate(candidate: Path) -> tuple[Path, list[Path], Path] | None:
        for root in _layout_variants(candidate):
            key = os.path.normcase(os.path.normpath(str(root)))
            if key in seen_checked:
                continue
            seen_checked.add(key)
            checked.append(str(root))
            result = _inspect_layout(root)
            if result:
                return result
        return None

    for candidate in candidates:
        result = try_candidate(candidate)
        if result:
            _save_cache_root(result[0])
            return result

    if allow_prompt and os.name == "nt":
        eprint("Automatic detection did not find the EVE game files folder.")
        eprint("Opening a folder picker. Select the folder that contains ResFiles and tq.")
        selected = _prompt_for_cache()
        if selected:
            result = try_candidate(selected)
            if result:
                _save_cache_root(result[0])
                return result
            eprint(f"Selected folder was not an EVE game-files root: {selected}")

    eprint("ERROR: Automatic EVE installation discovery did not locate the SharedCache indexes.")
    eprint("The tool searched Windows Installed Apps registry entries, App Paths, launcher configuration/logs,")
    eprint("previous NSAMDR manifests, saved verified locations, and common EVE/Steam install folders.")
    eprint("Expected an installation folder containing ResFiles and tq\\resfileindex.txt.")
    if raw:
        eprint(f"Requested path: {raw}")
    if checked:
        eprint("Last locations checked:")
        for item in checked[-12:]:
            eprint(f"  {item}")
    raise SystemExit(10)


def read_rows(indexes: list[Path]) -> list[ResourceRow]:
    rows: dict[str, ResourceRow] = {}
    for index in indexes:
        with index.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for values in csv.reader(handle):
                if len(values) < 2:
                    continue
                logical = values[0].strip().replace("\\", "/")
                hashed = values[1].strip().replace("\\", "/")
                if not logical or not hashed:
                    continue
                rows[logical.lower()] = ResourceRow(logical, hashed, str(index))
    return list(rows.values())


def select_model(rows: list[ResourceRow], query: str) -> ResourceRow:
    query = (query or DEFAULT_RAVEN).strip().replace("\\", "/")
    lower = query.lower()
    exact = [row for row in rows if row.logical.lower() == lower]
    if exact:
        return exact[0]

    terms = [term for term in lower.replace("res:/", "").split() if term]
    matches = [
        row for row in rows
        if row.logical.lower().endswith(".gr2")
        and all(term in row.logical.lower() for term in terms)
    ]
    if not matches and "/" in lower:
        basename = lower.rsplit("/", 1)[-1]
        matches = [row for row in rows if row.logical.lower().endswith(basename)]
    if not matches and "cb1" in lower:
        matches = [
            row for row in rows
            if row.logical.lower().endswith(".gr2")
            and "/model/ship/caldari/" in row.logical.lower()
            and "/cb1/" in row.logical.lower()
        ]
    if not matches:
        raise RuntimeError(f"No GR2 resource matched query: {query}")

    def score(row: ResourceRow) -> tuple[int, int, int, str]:
        value = row.logical.lower()
        return (
            0 if value.endswith("cb1_t1.gr2") else 1,
            0 if "/battleship/" in value else 1,
            len(value),
            value,
        )

    return sorted(matches, key=score)[0]


def related_textures(rows: list[ResourceRow], model: ResourceRow) -> dict[str, ResourceRow]:
    logical = model.logical.replace("\\", "/")
    directory = logical.rsplit("/", 1)[0].lower() + "/"
    stem = Path(logical.rsplit("/", 1)[-1]).stem.lower()
    in_folder = [row for row in rows if row.logical.lower().startswith(directory)]

    result: dict[str, ResourceRow] = {}
    suffixes = {"albedo": "_d.dds", "normal": "_n.dds", "pgs": "_pgs.dds"}
    for kind, suffix in suffixes.items():
        exact_name = f"{stem}{suffix}"
        exact = [row for row in in_folder if row.logical.lower().endswith("/" + exact_name)]
        if exact:
            result[kind] = exact[0]
            continue
        candidates = [row for row in in_folder if row.logical.lower().endswith(suffix)]
        if candidates:
            candidates.sort(key=lambda row: (0 if stem in row.logical.lower() else 1, len(row.logical)))
            result[kind] = candidates[0]
    return result



ENVIRONMENT_SCENES = {
    "amarr": "a03",
    "caldari": "c02",
    "gallente": "g03",
    "minmatar": "m02",
}


def _environment_code_for_model(model: ResourceRow) -> str:
    logical = model.logical.lower()
    for race, code in ENVIRONMENT_SCENES.items():
        if f"/ship/{race}/" in logical:
            return code
    return "c02"


def _resource_references_from_bytes(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="ignore")
    references = re.findall(
        r"res:/[^\s\"'<>]+?\.(?:dds|png|tga|jpg|jpeg)",
        text,
        flags=re.IGNORECASE,
    )
    return [value.rstrip("),]}>").replace("\\", "/") for value in references]


def select_environment_source(
    rows: list[ResourceRow],
    model: ResourceRow,
    resfiles: Path,
) -> tuple[ResourceRow | None, ResourceRow | None]:
    by_logical = {row.logical.lower(): row for row in rows}
    code = _environment_code_for_model(model)
    preferred_scene_names = [
        f"res:/dx9/scene/universe/{code}_cube.red",
        f"res:/graphics/scene/universe/{code}_cube.red",
    ]
    scene: ResourceRow | None = None
    for name in preferred_scene_names:
        row = by_logical.get(name)
        if row and (resfiles / Path(row.hashed)).is_file():
            scene = row
            break
    if scene is None:
        scene_candidates = [
            row for row in rows
            if row.logical.lower().endswith(".red")
            and "/scene/universe/" in row.logical.lower()
            and code in Path(row.logical).stem.lower()
            and (resfiles / Path(row.hashed)).is_file()
        ]
        if scene_candidates:
            scene = min(scene_candidates, key=lambda row: len(row.logical))

    referenced: list[ResourceRow] = []
    if scene is not None:
        try:
            data = (resfiles / Path(scene.hashed)).read_bytes()
            for reference in _resource_references_from_bytes(data):
                row = by_logical.get(reference.lower())
                if row and row.logical.lower().endswith(".dds"):
                    referenced.append(row)
        except OSError:
            pass

    candidates: dict[str, ResourceRow] = {row.logical.lower(): row for row in referenced}
    for row in rows:
        value = row.logical.lower()
        if not value.endswith(".dds") or "/scene/universe/" not in value:
            continue
        if code in value or any(token in value for token in ("nebula", "cube", "background", "env")):
            candidates[value] = row

    def score(row: ResourceRow) -> tuple[int, int, int, int, str]:
        value = row.logical.lower()
        exists = (resfiles / Path(row.hashed)).is_file()
        return (
            0 if exists else 1,
            0 if row in referenced else 1,
            0 if code in value else 1,
            0 if any(token in value for token in ("nebula", "cube", "background")) else 1,
            value,
        )

    texture = min(candidates.values(), key=score) if candidates else None
    if texture is not None and not (resfiles / Path(texture.hashed)).is_file():
        texture = None
    return scene, texture


def _safe_text(value: object, fallback: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("en", "en-us", "en_US"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return fallback


def _sde_cache_directory(repo_root: Path) -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "NSAMDR" / "sde"
    return repo_root / "artifacts" / "nsamdr" / "sde"


def _download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NSAMDR-EVE-asset-inspector/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            next_report = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if received >= next_report:
                    if total:
                        print(f"Downloading official EVE SDE: {received / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB", flush=True)
                    else:
                        print(f"Downloading official EVE SDE: {received / (1024 * 1024):.1f} MiB", flush=True)
                    next_report = received + 32 * 1024 * 1024
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _ensure_sde_archive(repo_root: Path) -> Path:
    cache_dir = _sde_cache_directory(repo_root)
    archive = cache_dir / "eve-online-static-data-latest-jsonl.zip"
    has_cached_archive = archive.is_file() and archive.stat().st_size > 1024 * 1024
    if has_cached_archive:
        age = time.time() - archive.stat().st_mtime
        if age <= SDE_CACHE_MAX_AGE_SECONDS:
            return archive
    print("Fetching the official EVE Static Data Export for ship names (cached for seven days)...", flush=True)
    try:
        _download_with_progress(SDE_LATEST_URL, archive)
    except (OSError, urllib.error.URLError) as exc:
        if has_cached_archive:
            print(f"WARNING: Could not refresh the SDE; using the existing cached copy: {exc}", flush=True)
            return archive
        raise
    return archive


def _find_zip_member(archive: zipfile.ZipFile, basename: str) -> str:
    wanted = basename.lower()
    matches = [name for name in archive.namelist() if Path(name).name.lower() == wanted]
    if not matches:
        raise RuntimeError(f"Official SDE archive did not contain {basename}")
    return min(matches, key=len)


def _read_jsonl_member(archive: zipfile.ZipFile, basename: str) -> Iterable[dict]:
    member = _find_zip_member(archive, basename)
    with archive.open(member, "r") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            value = json.loads(raw.decode("utf-8"))
            if isinstance(value, dict):
                yield value


def _normalise_resource_path(value: str) -> str:
    return value.strip().replace("\\", "/").lower()


def _model_family_key(logical: str) -> str:
    stem = Path(logical.rsplit("/", 1)[-1]).stem.lower()
    stem = re.sub(r"(?:[_-](?:lod|l|level|detail)[_-]?\d+)$", "", stem)
    stem = re.sub(r"(?:[_-](?:low|medium|med|high|proxy))$", "", stem)
    return stem


def _asset_quality_score(logical: str, exact_stem: str = "") -> tuple[int, int, int, int, str]:
    value = logical.lower()
    stem = Path(value.rsplit("/", 1)[-1]).stem
    is_lod = bool(re.search(r"(?:[_-](?:lod|l|level|detail)[_-]?\d+|[_-](?:low|medium|med|proxy))$", stem))
    high_hint = bool(re.search(r"(?:[_-](?:high|lod0|l0|detail0))$", stem))
    return (
        0 if exact_stem and stem == exact_stem else 1,
        0 if not is_lod else 1,
        0 if high_hint else 1,
        len(value),
        value,
    )


def _candidate_assets_for_graphic(
    ship_rows: list[ResourceRow],
    by_logical: dict[str, ResourceRow],
    graphic: dict,
) -> list[ResourceRow]:
    graphic_file = _normalise_resource_path(str(graphic.get("graphicFile") or ""))
    sof_hull = str(graphic.get("sofHullName") or "").strip().lower()
    exact_paths: list[str] = []
    if graphic_file:
        base, _ = os.path.splitext(graphic_file)
        exact_paths.extend([base + ".gr2", base + "_t1.gr2", base + "_t2.gr2"])
        if sof_hull and "/" in base:
            exact_paths.insert(0, base.rsplit("/", 1)[0] + "/" + sof_hull + ".gr2")
    for exact in exact_paths:
        row = by_logical.get(exact)
        if row:
            return [row]

    candidates: list[ResourceRow] = []
    if graphic_file and "/" in graphic_file:
        directory = graphic_file.rsplit("/", 1)[0] + "/"
        graphic_stem = Path(graphic_file.rsplit("/", 1)[-1]).stem.lower()
        candidates.extend(
            row for row in ship_rows
            if row.logical.lower().startswith(directory)
            and (
                Path(row.logical.rsplit("/", 1)[-1]).stem.lower().startswith(graphic_stem)
                or _model_family_key(row.logical) == _model_family_key(graphic_file)
            )
        )
    if sof_hull:
        hull_family = _model_family_key(sof_hull)
        candidates.extend(
            row for row in ship_rows
            if _model_family_key(row.logical) == hull_family
            or f"/{hull_family}/" in row.logical.lower()
        )
    unique = {row.logical.lower(): row for row in candidates}
    return list(unique.values())


def _expand_asset_variants(ship_rows: list[ResourceRow], preferred: ResourceRow) -> tuple[str, ...]:
    family = _model_family_key(preferred.logical)
    directory = preferred.logical.lower().rsplit("/", 1)[0] + "/"
    variants = sorted({
        row.logical.replace("\\", "/")
        for row in ship_rows
        if row.logical.lower().startswith(directory)
        and _model_family_key(row.logical) == family
    }, key=lambda value: _asset_quality_score(value, Path(preferred.logical).stem.lower()))
    if preferred.logical not in variants:
        variants.insert(0, preferred.logical)
    return tuple(variants)


def _infer_fallback_display(logical: str) -> tuple[str, str, str]:
    parts = logical.replace("\\", "/").split("/")
    lower_parts = [part.lower() for part in parts]
    race = "Unknown"
    ship_class = "Ship"
    try:
        ship_index = lower_parts.index("ship")
        if ship_index + 1 < len(parts):
            race = parts[ship_index + 1].replace("_", " ").title()
        if ship_index + 2 < len(parts):
            ship_class = parts[ship_index + 2].replace("_", " ").title()
    except ValueError:
        pass
    code = _model_family_key(logical)
    return f"Unmapped {race} {ship_class} ({code})", ship_class, race


def _build_sde_ship_catalog(rows: list[ResourceRow], repo_root: Path) -> list[ShipCatalogEntry]:
    ship_rows = [
        row for row in rows
        if row.logical.lower().endswith(".gr2") and "/model/ship/" in row.logical.lower()
    ]
    by_logical = {row.logical.lower(): row for row in ship_rows}
    archive_path = _ensure_sde_archive(repo_root)
    with zipfile.ZipFile(archive_path, "r") as archive:
        groups: dict[int, tuple[int, str]] = {}
        for value in _read_jsonl_member(archive, "groups.jsonl"):
            group_id = value.get("_key")
            if isinstance(group_id, int):
                groups[group_id] = (
                    int(value.get("categoryID") or -1),
                    _safe_text(value.get("name"), f"Group {group_id}"),
                )

        graphics: dict[int, dict] = {}
        for value in _read_jsonl_member(archive, "graphics.jsonl"):
            graphic_id = value.get("_key")
            if isinstance(graphic_id, int):
                graphics[graphic_id] = value

        entries: list[ShipCatalogEntry] = []
        claimed_assets: set[str] = set()
        for value in _read_jsonl_member(archive, "types.jsonl"):
            type_id = value.get("_key")
            group_id = value.get("groupID")
            graphic_id = value.get("graphicID")
            if not isinstance(type_id, int) or not isinstance(group_id, int) or not isinstance(graphic_id, int):
                continue
            category_id, group_name = groups.get(group_id, (-1, ""))
            if category_id != 6 or value.get("published") is False:
                continue
            graphic = graphics.get(graphic_id)
            if not graphic:
                continue
            candidates = _candidate_assets_for_graphic(ship_rows, by_logical, graphic)
            if not candidates:
                continue
            sof_hull = str(graphic.get("sofHullName") or "").strip().lower()
            graphic_stem = _model_family_key(sof_hull) if sof_hull else Path(str(graphic.get("graphicFile") or "")).stem.lower()
            preferred = min(candidates, key=lambda row: _asset_quality_score(row.logical, graphic_stem))
            variants = _expand_asset_variants(ship_rows, preferred)
            claimed_assets.update(asset.lower() for asset in variants)
            display_name = _safe_text(value.get("name"), f"Type {type_id}")
            faction_source = str(graphic.get("sofRaceName") or graphic.get("sofFactionName") or "").strip().replace("_", " ")
            faction_name = faction_source.title()
            if faction_name.lower().endswith("base") and len(faction_name) > 4:
                faction_name = faction_name[:-4].rstrip().title()
            entries.append(ShipCatalogEntry(
                display_name=display_name,
                type_id=type_id,
                group_name=group_name or "Ship",
                faction_name=faction_name,
                canonical_key=f"type:{type_id}",
                preferred_asset=preferred.logical.replace("\\", "/"),
                variants=variants,
            ))

    # Preserve access to cache-only/test hulls that are absent from the public SDE.
    fallback_groups: dict[tuple[str, str], list[ResourceRow]] = {}
    for row in ship_rows:
        if row.logical.lower() in claimed_assets:
            continue
        directory = row.logical.lower().rsplit("/", 1)[0]
        fallback_groups.setdefault((directory, _model_family_key(row.logical)), []).append(row)
    for (_, family), family_rows in fallback_groups.items():
        preferred = min(family_rows, key=lambda row: _asset_quality_score(row.logical))
        display_name, group_name, faction_name = _infer_fallback_display(preferred.logical)
        variants = tuple(sorted(
            {row.logical.replace("\\", "/") for row in family_rows},
            key=_asset_quality_score,
        ))
        entries.append(ShipCatalogEntry(
            display_name=display_name,
            type_id=None,
            group_name=group_name,
            faction_name=faction_name,
            canonical_key=f"asset:{family}:{preferred.logical.lower().rsplit('/', 1)[0]}",
            preferred_asset=preferred.logical.replace("\\", "/"),
            variants=variants,
        ))

    entries.sort(key=lambda entry: (
        entry.display_name.lower(),
        entry.group_name.lower(),
        entry.preferred_asset.lower(),
    ))
    return entries


def _build_fallback_ship_catalog(rows: list[ResourceRow]) -> list[ShipCatalogEntry]:
    ship_rows = [
        row for row in rows
        if row.logical.lower().endswith(".gr2") and "/model/ship/" in row.logical.lower()
    ]
    grouped: dict[tuple[str, str], list[ResourceRow]] = {}
    for row in ship_rows:
        directory = row.logical.lower().rsplit("/", 1)[0]
        grouped.setdefault((directory, _model_family_key(row.logical)), []).append(row)
    entries: list[ShipCatalogEntry] = []
    for (_, family), family_rows in grouped.items():
        preferred = min(family_rows, key=lambda row: _asset_quality_score(row.logical))
        display_name, group_name, faction_name = _infer_fallback_display(preferred.logical)
        variants = tuple(sorted(
            {row.logical.replace("\\", "/") for row in family_rows},
            key=_asset_quality_score,
        ))
        entries.append(ShipCatalogEntry(
            display_name=display_name,
            type_id=None,
            group_name=group_name,
            faction_name=faction_name,
            canonical_key=f"asset:{family}:{preferred.logical.lower().rsplit('/', 1)[0]}",
            preferred_asset=preferred.logical.replace("\\", "/"),
            variants=variants,
        ))
    return sorted(entries, key=lambda entry: entry.display_name.lower())


def _tsv_clean(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def write_ship_catalog(rows: list[ResourceRow], output_path: Path, repo_root: Path) -> Path:
    source = "official SDE"
    try:
        entries = _build_sde_ship_catalog(rows, repo_root)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
        source = f"cache-path fallback ({exc})"
        print(f"WARNING: Real-name SDE mapping unavailable: {exc}", flush=True)
        entries = _build_fallback_ship_catalog(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# NSAMDR_SHIP_CATALOG_V2\tdisplay_name\tgroup\tfaction\ttype_id\tcanonical_key\tpreferred_asset\tvariants"]
    for entry in entries:
        lines.append("\t".join([
            _tsv_clean(entry.display_name),
            _tsv_clean(entry.group_name),
            _tsv_clean(entry.faction_name),
            str(entry.type_id or 0),
            _tsv_clean(entry.canonical_key),
            _tsv_clean(entry.preferred_asset),
            "|".join(_tsv_clean(value) for value in entry.variants),
        ]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ship catalog: {output_path} ({len(entries)} grouped ships; names from {source})", flush=True)
    return output_path


def source_path(resfiles: Path, row: ResourceRow) -> Path:
    return resfiles / Path(row.hashed)


def copy_resource(resfiles: Path, row: ResourceRow, output_dir: Path) -> Path:
    source = source_path(resfiles, row)
    if not source.is_file():
        raise RuntimeError(
            f"Indexed EVE resource is not present locally: {row.logical}\n"
            f"Expected cache file: {source}\n"
            "In the EVE Launcher, verify the Shared Cache and enable full resource download, then retry."
        )
    logical_name = row.logical.rsplit("/", 1)[-1]
    destination = output_dir / logical_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def find_command(names: list[str]) -> str:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(f"Required command not found: {' or '.join(names)}")


def ensure_converter(converter_dir: Path) -> tuple[str, Path]:
    node = find_command(["node.exe", "node"])
    npm = find_command(["npm.cmd", "npm.exe", "npm"])
    script = converter_dir / "convert_eve_asset.mjs"
    package = converter_dir / "package.json"
    installed = converter_dir / "node_modules" / "@carbonenginejs" / "runtime-resource"
    if not script.is_file() or not package.is_file():
        raise RuntimeError(f"Missing NSAMDR converter source under {converter_dir}")
    if not installed.is_dir():
        print("Installing the open-source CarbonEngineJS GR2/DDS reader (one-time setup)...", flush=True)
        result = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=converter_dir,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "npm could not install @carbonenginejs/runtime-resource. "
                "Check Node.js 18+ and npm connectivity, then rerun."
            )
    return node, script


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command[0]}")


def convert_gr2(repo_root: Path, input_path: Path, output_path: Path) -> Path:
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    summary = output_path.with_suffix(".conversion.json")
    run_checked([node, str(script), "gr2-to-obj", str(input_path), str(output_path), str(summary)], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"GR2 converter did not create {output_path}")
    return output_path


def convert_dds(repo_root: Path, input_path: Path, output_path: Path) -> Path:
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    run_checked([node, str(script), "dds-to-png", str(input_path), str(output_path)], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"DDS converter did not create {output_path}")
    return output_path


def convert_environment_dds(repo_root: Path, input_path: Path, output_path: Path) -> Path:
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    run_checked([node, str(script), "dds-to-environment-png", str(input_path), str(output_path)], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"Environment DDS converter did not create {output_path}")
    return output_path


def prepare_asset(
    repo_root: Path,
    cache_arg: str,
    query: str,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, Path, Path, Path]:
    cache_root, indexes, resfiles = resolve_layout(cache_arg, allow_prompt=False)
    print(f"EVE SharedCache: {cache_root}", flush=True)
    print("Reading EVE resource indexes...", flush=True)
    rows = read_rows(indexes)
    print(f"Indexed resources: {len(rows)}", flush=True)

    catalog_path = write_ship_catalog(
        rows,
        repo_root / "artifacts" / "nsamdr" / "eve_assets" / "ship_catalog.tsv",
        repo_root,
    )

    model = select_model(rows, query)
    textures = related_textures(rows, model)
    environment_scene, environment_texture = select_environment_source(rows, model, resfiles)
    print(f"Selected model: {model.logical}", flush=True)
    for kind, row in textures.items():
        print(f"Selected {kind}: {row.logical}", flush=True)
    if environment_scene:
        print(f"Selected EVE environment scene: {environment_scene.logical}", flush=True)
    if environment_texture:
        print(f"Selected EVE environment texture: {environment_texture.logical}", flush=True)
    else:
        print("No local EVE nebula texture was available; the viewer will use its procedural fallback.", flush=True)

    asset_name = Path(model.logical.rsplit("/", 1)[-1]).stem
    output_dir = repo_root / "artifacts" / "nsamdr" / "eve_assets" / asset_name
    gr2_path = copy_resource(resfiles, model, output_dir)
    copied_environment_scene = copy_resource(resfiles, environment_scene, output_dir) if environment_scene else None
    copied_environment_texture = copy_resource(resfiles, environment_texture, output_dir) if environment_texture else None
    copied: dict[str, Path] = {}
    for kind, row in textures.items():
        copied[kind] = copy_resource(resfiles, row, output_dir)

    obj_path = convert_gr2(repo_root, gr2_path, output_dir / f"{asset_name}.obj")
    converted_textures: dict[str, Path] = {}
    for kind in ("albedo", "normal", "pgs"):
        if kind in copied:
            suffix = {"albedo": "_d.png", "normal": "_n.png", "pgs": "_pgs.png"}[kind]
            converted_textures[kind] = convert_dds(repo_root, copied[kind], output_dir / f"{asset_name}{suffix}")
    environment_png = None
    if copied_environment_texture:
        try:
            environment_png = convert_environment_dds(
                repo_root,
                copied_environment_texture,
                output_dir / f"{asset_name}_environment.png",
            )
        except RuntimeError as exc:
            print(f"WARNING: EVE environment conversion failed; using procedural fallback: {exc}", flush=True)

    manifest = {
        "cacheRoot": str(cache_root),
        "catalog": str(catalog_path),
        "model": {**asdict(model), "local": str(gr2_path)},
        "textures": {
            kind: {
                **asdict(textures[kind]),
                "local": str(path),
                "converted": str(converted_textures[kind]) if kind in converted_textures else None,
            }
            for kind, path in copied.items()
        },
        "obj": str(obj_path),
        "albedoPng": str(converted_textures.get("albedo")) if converted_textures.get("albedo") else None,
        "normalPng": str(converted_textures.get("normal")) if converted_textures.get("normal") else None,
        "pgsPng": str(converted_textures.get("pgs")) if converted_textures.get("pgs") else None,
        "environment": {
            "scene": {**asdict(environment_scene), "local": str(copied_environment_scene)} if environment_scene else None,
            "texture": {**asdict(environment_texture), "local": str(copied_environment_texture)} if environment_texture else None,
            "converted": str(environment_png) if environment_png else None,
        },
        "environmentPng": str(environment_png) if environment_png else None,
    }
    manifest_path = output_dir / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Asset manifest: {manifest_path}", flush=True)
    return (
        obj_path,
        converted_textures.get("albedo"),
        converted_textures.get("normal"),
        converted_textures.get("pgs"),
        environment_png,
        manifest_path,
        catalog_path,
        cache_root,
    )


def launch_preview(
    repo_root: Path,
    launcher: Path,
    obj_path: Path,
    albedo: Path | None,
    normal: Path | None,
    pgs: Path | None,
    environment: Path | None,
    manifest: Path,
    catalog: Path,
    cache_root: Path,
    current_query: str,
) -> int:
    if not launcher.is_file():
        raise RuntimeError(f"Missing preview launcher: {launcher}")
    command = [
        os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call",
        str(launcher), str(obj_path), str(albedo or ""), str(normal or ""), str(pgs or ""), str(environment or ""),
    ]
    env = os.environ.copy()
    env.update({
        "NSAMDR_EVE_MANIFEST": str(manifest),
        "NSAMDR_EVE_CATALOG": str(catalog),
        "NSAMDR_EVE_CACHE": str(cache_root),
        "NSAMDR_EVE_REPO_ROOT": str(repo_root),
        "NSAMDR_EVE_TOOL": str(Path(__file__).resolve()),
        "NSAMDR_EVE_LAUNCHER": str(launcher),
        "NSAMDR_EVE_QUERY": current_query,
        "NSAMDR_ENVIRONMENT": str(environment or ""),
        "NSAMDR_PYTHON_EXE": sys.executable,
    })
    print("Launching the Granny-free Trinity NSAMDR viewer...", flush=True)
    return subprocess.run(command, cwd=repo_root, env=env, check=False).returncode


def command_list(args: argparse.Namespace) -> int:
    _, indexes, _ = resolve_layout(args.shared_cache)
    rows = read_rows(indexes)
    query = args.query.lower()
    matches = [row for row in rows if query in row.logical.lower()]
    matches.sort(key=lambda row: row.logical.lower())
    for row in matches[: args.limit]:
        print(f"{row.logical},{row.hashed}")
    print(f"Matches shown: {min(len(matches), args.limit)} of {len(matches)}", file=sys.stderr)
    return 0


def command_prepare_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    obj_path, albedo, normal, pgs, environment, manifest, catalog, cache_root = prepare_asset(
        repo_root, args.shared_cache, args.query
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    selected_query = str(manifest_data.get("model", {}).get("logical") or args.query)
    selected_catalog_key = str(args.selection_key or selected_query)
    return launch_preview(
        repo_root, Path(args.launcher).resolve(), obj_path, albedo, normal, pgs, environment,
        manifest, catalog, cache_root, selected_catalog_key
    )


def command_convert_gr2(args: argparse.Namespace) -> int:
    convert_gr2(Path(args.repo_root).resolve(), Path(args.input).resolve(), Path(args.output).resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract real EVE SharedCache assets for the Granny-free NSAMDR viewer")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List indexed EVE resources matching a substring")
    list_parser.add_argument("--shared-cache", default="")
    list_parser.add_argument("--query", default="cb1")
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.set_defaults(func=command_list)

    prepare = sub.add_parser("prepare-run", help="Extract, convert and launch a real EVE asset")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--shared-cache", default="")
    prepare.add_argument("--query", default=DEFAULT_RAVEN)
    prepare.add_argument("--selection-key", default="", help="Catalog identity to reselect after switching ships")
    prepare.add_argument("--launcher", required=True)
    prepare.set_defaults(func=command_prepare_run)

    convert = sub.add_parser("convert-gr2", help="Convert a local GR2 file to OBJ using CarbonEngineJS")
    convert.add_argument("--repo-root", required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.set_defaults(func=command_convert_gr2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        eprint("Cancelled.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        eprint(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
