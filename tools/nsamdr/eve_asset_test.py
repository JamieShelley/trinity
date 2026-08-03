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


def select_environment_sources(
    rows: list[ResourceRow],
    model: ResourceRow,
    resfiles: Path,
) -> tuple[ResourceRow | None, list[ResourceRow]]:
    """Return every locally available universe cubemap, with the ship-race map first."""
    scene, primary = select_environment_source(rows, model, resfiles)
    code = _environment_code_for_model(model)
    candidates: dict[str, ResourceRow] = {}
    for row in rows:
        logical = row.logical.lower().replace("\\", "/")
        if not logical.endswith(".dds") or "/scene/universe/" not in logical:
            continue
        if not any(token in logical for token in ("cube", "nebula", "background", "environment", "_env")):
            continue
        if not (resfiles / Path(row.hashed)).is_file():
            continue
        candidates[logical] = row

    ordered = list(candidates.values())
    ordered.sort(key=lambda row: (
        0 if primary and row.logical.lower() == primary.logical.lower() else 1,
        0 if code in row.logical.lower() else 1,
        0 if "_cube.dds" in row.logical.lower() else 1,
        row.logical.lower(),
    ))
    return scene, ordered


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


CONVERTER_MODULE_PROBE = (
    "await import('@carbonenginejs/format-gr2'); "
    "await import('@carbonenginejs/runtime-resource/formats/dds'); "
    "await import('black-reader'); "
    "await import('black-reader/black-classes.js'); "
    "await import('black-reader/black-readers.js');"
)


def probe_converter_modules(node: str, converter_dir: Path, *, show_failure: bool = False) -> bool:
    result = subprocess.run(
        [node, "--input-type=module", "--eval", CONVERTER_MODULE_PROBE],
        cwd=converter_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0 and show_failure:
        diagnostic = (result.stderr or result.stdout or "").strip()
        if diagnostic:
            print("NSAMDR converter module-resolution probe failed:", flush=True)
            print("\n".join(diagnostic.splitlines()[-20:]), flush=True)
    return result.returncode == 0


def ensure_converter(converter_dir: Path) -> tuple[str, Path]:
    node = find_command(["node.exe", "node"])
    script = converter_dir / "convert_eve_asset.mjs"
    package = converter_dir / "package.json"
    if not script.is_file() or not package.is_file():
        raise RuntimeError(f"Missing NSAMDR converter source under {converter_dir}")

    # npm is free to hoist or nest transitive packages. Directory-layout checks
    # therefore produce false failures even when Node can resolve the two public
    # entry points used by the converter. Test the imports themselves instead.
    if not probe_converter_modules(node, converter_dir):
        npm = find_command(["npm.cmd", "npm.exe", "npm"])
        print("Installing the open-source CarbonEngineJS readers from public GitHub source archives (one-time setup)...", flush=True)
        lock_file = converter_dir / "package-lock.json"
        if lock_file.exists():
            lock_file.unlink()
        node_modules = converter_dir / "node_modules"
        if node_modules.exists():
            shutil.rmtree(node_modules)
        result = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", "--omit=dev", "--package-lock=false"],
            cwd=converter_dir,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "npm could not install the open-source GR2, DDS and EVE Black readers from their public GitHub source archives. "
                "Check Node.js 18+, HTTPS access to github.com, and npm connectivity, then rerun."
            )

    if not probe_converter_modules(node, converter_dir, show_failure=True):
        raise RuntimeError(
            "Converter dependency installation finished, but Node could not import the GR2, DDS or EVE Black reader entry points. "
            "The module-resolution error printed above identifies the unresolved package."
        )
    return node, script


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout or "").strip()
        if diagnostic:
            diagnostic = "\n".join(diagnostic.splitlines()[-12:])
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {command[0]}\n{diagnostic}"
            )
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
    if output_path.is_file() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return output_path
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    run_checked([node, str(script), "dds-to-png", str(input_path), str(output_path)], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"DDS converter did not create {output_path}")
    return output_path


def convert_environment_dds(repo_root: Path, input_path: Path, output_path: Path) -> Path:
    if output_path.is_file() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return output_path
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    run_checked([node, str(script), "dds-to-environment-png", str(input_path), str(output_path)], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"Environment DDS converter did not create {output_path}")
    return output_path



SOF_DATA_PATH = "res:/dx9/model/spaceobjectfactory/data.black"
AREA_TYPE_FALLBACK_COLORS: dict[str, tuple[float, float, float]] = {
    "primary": (0.34, 0.38, 0.42),
    "glass": (0.035, 0.10, 0.16),
    "sails": (0.42, 0.42, 0.40),
    "reactor": (0.10, 0.20, 0.26),
    "darkhull": (0.025, 0.032, 0.040),
    "wreck": (0.20, 0.16, 0.12),
    "rock": (0.18, 0.17, 0.15),
    "monument": (0.35, 0.35, 0.34),
    "ornament": (0.50, 0.46, 0.35),
    "simpleprimary": (0.34, 0.38, 0.42),
    "turret": (0.12, 0.14, 0.16),
}
COLOR_TYPE_NAMES = [
    "primary", "secondary", "tertiary", "black", "white", "yellow", "orange", "red",
    "blue", "green", "cyan", "fire", "hull", "glass", "reactor", "darkhull", "booster",
    "killmark", "primarylight", "secondarylight", "tertiarylight", "whitelight",
]


def _race_from_ship_model(logical_path: str) -> str:
    """Derive the SOF race from the selected hull's canonical model path.

    EVE ship geometry is stored beneath ``model/ship/<race>/...``. This is the
    hull race and remains valid for faction variants which reuse that hull.
    """
    normalized = logical_path.strip().replace("\\", "/").lower()
    match = re.search(r"(?:^|/)model/ship/([^/]+)/", normalized)
    return match.group(1).strip() if match else ""


def _base_sof_hull_from_model(logical_path: str) -> str:
    """Resolve the base SOF hull id from a direct model resource path.

    A model such as ``.../cb1/cb1_t1.gr2`` belongs to SOF hull ``cb1``.  The
    filename is a geometry variant, not the SOF hull identifier.
    """
    normalized = logical_path.strip().replace("\\", "/").lower()
    path_value = Path(normalized)
    directory_name = path_value.parent.name.strip()
    stem = path_value.stem.strip()
    if directory_name and (
        stem == directory_name
        or stem.startswith(directory_name + "_")
        or stem.startswith(directory_name + "-")
    ):
        return directory_name
    return _model_family_key(normalized)


def _is_base_sof_faction(faction: str, race: str) -> bool:
    normalized_faction = _normal_key(faction)
    normalized_race = _normal_key(race)
    return bool(normalized_race) and normalized_faction in {normalized_race, normalized_race + "base"}


def _resolve_sof_identity(
    rows: list[ResourceRow],
    repo_root: Path,
    model: ResourceRow,
    selection_key: str,
) -> dict[str, str | int | bool | None]:
    """Resolve SOF identity without guessing between ships that share a hull.

    The standalone default supplies a direct GR2 path and no type key.  Many
    SDE graphic records can reference that same geometry; choosing the first
    match silently changes the ship skin/faction.  Direct paths therefore use
    the deterministic base hull identity.  An explicit ``type:<id>`` selection
    from the viewer is the only path allowed to select a faction/skin variant.
    """
    model_race = _race_from_ship_model(model.logical)
    base_hull = _base_sof_hull_from_model(model.logical)
    match = re.fullmatch(r"type:(\d+)", (selection_key or "").strip().lower())
    wanted_type = int(match.group(1)) if match else None

    if wanted_type is None:
        faction = f"{model_race}base" if model_race else ""
        return {
            "typeID": None,
            "hull": base_hull,
            "faction": faction,
            "race": model_race,
            "raceSource": "modelPath" if model_race else "unresolved",
            "identitySource": "direct-model-base",
            "preferFactionTextures": False,
        }

    archive_path = _ensure_sde_archive(repo_root)
    selected_graphic: dict | None = None
    with zipfile.ZipFile(archive_path, "r") as archive:
        graphics = {
            value.get("_key"): value
            for value in _read_jsonl_member(archive, "graphics.jsonl")
            if isinstance(value.get("_key"), int)
        }
        for value in _read_jsonl_member(archive, "types.jsonl"):
            if value.get("_key") == wanted_type:
                selected_graphic = graphics.get(value.get("graphicID"))
                break

    if selected_graphic is None:
        faction = f"{model_race}base" if model_race else ""
        return {
            "typeID": wanted_type,
            "hull": base_hull,
            "faction": faction,
            "race": model_race,
            "raceSource": "modelPath" if model_race else "unresolved",
            "identitySource": "missing-sde-type-fallback",
            "preferFactionTextures": False,
        }

    explicit_race = str(selected_graphic.get("sofRaceName") or "").strip()
    race = explicit_race or model_race
    faction = str(selected_graphic.get("sofFactionName") or (f"{race}base" if race else "")).strip()
    return {
        "typeID": wanted_type,
        "hull": str(selected_graphic.get("sofHullName") or base_hull).strip(),
        "faction": faction,
        "race": race,
        "raceSource": "sde" if explicit_race else ("modelPath" if model_race else "unresolved"),
        "identitySource": "explicit-sde-type",
        # Variant factions may intentionally replace the base hull maps. Base
        # factions retain the exact selected source textures first.
        "preferFactionTextures": not _is_base_sof_faction(faction, race),
    }


def convert_sof(
    repo_root: Path,
    data_black: Path,
    output_path: Path,
    hull: str,
    faction: str,
    race: str,
    model_path: str,
) -> Path:
    converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
    node, script = ensure_converter(converter_dir)
    run_checked([
        node, str(script), "sof-to-json", str(data_black), str(output_path),
        hull, faction, race, model_path,
    ], converter_dir)
    if not output_path.is_file():
        raise RuntimeError(f"SOF converter did not create {output_path}")
    return output_path


def _resource_row(rows_by_logical: dict[str, ResourceRow], logical: str) -> ResourceRow | None:
    return rows_by_logical.get(logical.strip().replace("\\", "/").lower())


def _faction_texture_path(logical: str, insert: str) -> str:
    logical = logical.strip().replace("\\", "/")
    insert = insert.strip().strip("/\\")
    if not insert or "/" not in logical:
        return logical
    directory, filename = logical.rsplit("/", 1)
    separator = filename.find("_")
    if separator < 0:
        faction_filename = f"{filename}_{insert}"
    else:
        faction_filename = f"{filename[:separator]}_{insert}{filename[separator:]}"
    return f"{directory}/{insert}/{faction_filename}"


def _resolve_sof_texture(
    rows_by_logical: dict[str, ResourceRow],
    resfiles: Path,
    logical: str,
    res_path_insert: str,
    prefer_faction_texture: bool = False,
) -> ResourceRow | None:
    """Resolve one SOF texture while preserving the selected source identity.

    Base/direct previews must not silently replace the authored hull texture
    with the first available faction insertion. Explicit non-base SDE type
    selections may prefer that faction replacement.
    """
    original = _resource_row(rows_by_logical, logical)
    modified = _resource_row(rows_by_logical, _faction_texture_path(logical, res_path_insert))
    ordered = (modified, original) if prefer_faction_texture else (original, modified)
    for candidate in ordered:
        if candidate is not None and source_path(resfiles, candidate).is_file():
            return candidate
    return ordered[0] or ordered[1]


def _numeric_vector(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            numbers = [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        while len(numbers) < 4:
            numbers.append(1.0)
        return tuple(numbers[:4])  # type: ignore[return-value]
    if isinstance(value, dict):
        ordered = []
        for keys in (("r", "g", "b", "a"), ("x", "y", "z", "w")):
            if all(key in value for key in keys[:3]):
                for key in keys:
                    ordered.append(value.get(key, 1.0))
                return _numeric_vector(ordered)
    return None


def _normal_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_named_value(value: object, wanted_names: Iterable[str]) -> object | None:
    wanted = {_normal_key(name) for name in wanted_names}
    if isinstance(value, dict):
        for key, child in value.items():
            if _normal_key(str(key)) in wanted:
                return child
        for child in value.values():
            result = _find_named_value(child, wanted)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_named_value(child, wanted)
            if result is not None:
                return result
    return None


def _faction_color(sof: dict, names: Iterable[str]) -> tuple[float, float, float, float] | None:
    value = _find_named_value(sof.get("colors", {}), names)
    vector = _numeric_vector(value)
    if vector is None:
        return None
    return tuple(max(0.0, min(4.0, component)) for component in vector)  # type: ignore[return-value]


def _material_for_area(sof: dict, area_type: str) -> dict:
    materials = sof.get("areaMaterials", {})
    if not isinstance(materials, dict):
        return {}
    for key, value in materials.items():
        if _normal_key(str(key)) == _normal_key(area_type) and isinstance(value, dict):
            return value
    return {}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _numeric_scalar(value: object) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else None
    vector = _numeric_vector(value)
    return vector[0] if vector is not None else None


def _area_base_visuals(
    sof: dict,
    area_type: str,
    race_name: str,
) -> tuple[tuple[float, float, float], float, float, float, tuple[float, float, float]]:
    material = _material_for_area(sof, area_type)
    if not material and area_type != "primary":
        material = _material_for_area(sof, "primary")
    color_type = _find_named_value(material, ("colorType", "glowColorType"))
    color_name = ""
    if isinstance(color_type, str):
        color_name = color_type.replace("TYPE_", "").lower()
    elif isinstance(color_type, (int, float)) and 0 <= int(color_type) < len(COLOR_TYPE_NAMES):
        color_name = COLOR_TYPE_NAMES[int(color_type)]

    area_color_names = {
        "primary": ("primary", "hull"),
        "simpleprimary": ("primary", "hull"),
        "glass": ("glass", "cyan", "blue"),
        "reactor": ("reactor", "fire", "orange"),
        "darkhull": ("darkhull", "black"),
        "ornament": ("secondary", "white", "primary"),
        "sails": ("secondary", "primary"),
        "turret": ("darkhull", "black"),
    }.get(area_type, (area_type, "primary", "hull"))
    tint = _faction_color(sof, area_color_names)

    race_defaults = {
        "amarr": (0.48, 0.38, 0.22),
        "caldari": (0.23, 0.31, 0.37),
        "gallente": (0.20, 0.34, 0.28),
        "minmatar": (0.38, 0.23, 0.15),
    }
    if tint is None:
        base = AREA_TYPE_FALLBACK_COLORS.get(area_type, race_defaults.get(race_name.lower(), (0.34, 0.38, 0.42)))
        if area_type in ("primary", "simpleprimary"):
            base = race_defaults.get(race_name.lower(), base)
        tint_rgb = base
    else:
        tint_rgb = tint[:3]

    default_roughness = 0.48
    default_specular = 0.72
    if area_type == "glass":
        default_roughness, default_specular = 0.10, 1.25
    elif area_type == "darkhull":
        default_roughness, default_specular = 0.62, 0.55
    elif area_type == "ornament":
        default_roughness, default_specular = 0.28, 1.10
    elif area_type == "reactor":
        default_roughness, default_specular = 0.42, 0.85

    glow_names = tuple(name for name in (color_name, "primary_light", "white_light", "primary") if name)
    glow = _faction_color(sof, glow_names)
    if glow is not None:
        glow_rgb = glow[:3]
    elif area_type == "reactor":
        glow_rgb = (1.0, 0.26, 0.04)
    elif area_type == "glass":
        glow_rgb = (0.12, 0.48, 0.85)
    else:
        glow_rgb = (0.34, 0.58, 0.95)
    return tint_rgb, 1.0, default_roughness, default_specular, glow_rgb


def _material_names_for_area(sof: dict, area_type: str) -> list[str]:
    material = _material_for_area(sof, area_type)
    if not material and area_type != "primary":
        material = _material_for_area(sof, "primary")
    names: list[str] = []
    for index in range(1, 5):
        value = _find_named_value(material, (f"material{index}",))
        names.append(value.strip() if isinstance(value, str) else "")
    return names


def _lookup_material_parameters(sof: dict, material_name: str) -> dict:
    library = sof.get("materialLibrary", {})
    if not material_name or not isinstance(library, dict):
        return {}
    wanted = material_name.lower()
    for key, value in library.items():
        if str(key).lower() == wanted and isinstance(value, dict):
            return value
    return {}


def _slot_fallback_colors(
    sof: dict,
    area_type: str,
    race_name: str,
    area_tint: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    def color(names: tuple[str, ...], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        value = _faction_color(sof, names)
        return value[:3] if value is not None else fallback

    primary = color(("primary", "hull"), area_tint)
    secondary = color(("secondary", "white"), tuple(_clamp(component * 1.18, 0.0, 2.0) for component in area_tint))
    tertiary = color(("tertiary", "darkhull"), tuple(_clamp(component * 0.62, 0.0, 2.0) for component in area_tint))
    dark = color(("darkhull", "black"), (0.035, 0.045, 0.055))
    if area_type == "darkhull":
        return [dark, tuple(component * 0.65 for component in dark), primary, tertiary]
    if area_type == "glass":
        glass = color(("glass", "cyan", "blue"), (0.08, 0.28, 0.42))
        return [glass, primary, secondary, dark]
    if area_type == "reactor":
        reactor = color(("reactor", "fire", "orange"), (0.62, 0.16, 0.035))
        return [reactor, dark, primary, secondary]
    if area_type == "ornament":
        return [secondary, primary, tertiary, dark]
    return [primary, secondary, tertiary, dark]


def _parameter_vector(parameters: dict, names: Iterable[str]) -> tuple[float, float, float, float] | None:
    return _numeric_vector(_find_named_value(parameters, names))


def _parameter_scalar(parameters: dict, names: Iterable[str]) -> float | None:
    return _numeric_scalar(_find_named_value(parameters, names))


def _material_slot_visuals(
    material_name: str,
    parameters: dict,
    fallback_color: tuple[float, float, float],
    area_type: str,
    default_roughness: float,
) -> dict[str, object]:
    base_vector = _parameter_vector(parameters, (
        "DiffuseColor", "AlbedoColor", "BaseColor", "MaterialColor", "GeneralColor", "Color",
    ))
    if base_vector is not None and max(base_vector[:3]) > 1.0e-5:
        base_color = tuple(_clamp(component, 0.0, 4.0) for component in base_vector[:3])
    else:
        base_color = fallback_color

    roughness = default_roughness
    rough_value = _parameter_scalar(parameters, (
        "Roughness", "RoughnessFactors", "MaterialRoughness", "DiffuseRoughness",
    ))
    if rough_value is not None:
        roughness = _clamp(rough_value, 0.035, 0.98)
    else:
        gloss = _parameter_scalar(parameters, ("Gloss", "Glossiness", "GlossFactors", "SpecularPower"))
        if gloss is not None:
            roughness = _clamp(1.0 - gloss, 0.035, 0.98) if gloss <= 1.0 else _clamp((2.0 / (gloss + 2.0)) ** 0.5, 0.035, 0.98)

    lower_name = material_name.lower()
    metallic_hint = any(token in lower_name for token in ("metal", "steel", "chrome", "silver", "gold", "copper", "brass"))
    metalness = _parameter_scalar(parameters, ("Metallic", "Metalness", "MetallicFactor"))
    if metalness is None:
        metalness = 0.88 if metallic_hint else (0.35 if area_type in ("ornament", "reactor") else 0.0)
    metalness = _clamp(metalness)

    f0_vector = _parameter_vector(parameters, ("FresnelColor", "SpecularColor", "ReflectanceColor"))
    if f0_vector is not None:
        f0 = tuple(_clamp(component, 0.018, 1.0) for component in f0_vector[:3])
    else:
        f0_scalar = _parameter_scalar(parameters, ("Reflectance", "Specular", "SpecularFactor", "FresnelFactors"))
        dielectric = _clamp(0.04 if f0_scalar is None else f0_scalar, 0.018, 0.24)
        f0 = tuple((1.0 - metalness) * dielectric + metalness * _clamp(component, 0.02, 1.0) for component in base_color)

    if area_type == "glass":
        roughness = min(roughness, 0.16)
        f0 = tuple(max(component, 0.075) for component in f0)
    if metallic_hint:
        roughness = min(roughness, 0.38)
    return {
        "name": material_name,
        "color": base_color,
        "f0": f0,
        "roughness": roughness,
        "gloss": _clamp(1.0 - roughness, 0.0, 1.0),
    }


def _area_visual_parameters(sof: dict, area_type: str, race_name: str) -> tuple[tuple[float, float, float], float, float, float]:
    """Legacy V1 fallback parameters retained for old manifests."""
    tint, detail_scale, roughness, specular, _ = _area_base_visuals(sof, area_type, race_name)
    return tint, detail_scale, roughness, specular


def _area_material_slots(sof: dict, area_type: str, race_name: str) -> tuple[list[dict[str, object]], tuple[float, float, float], tuple[float, float, float], float]:
    tint, detail_scale, default_roughness, _, glow = _area_base_visuals(sof, area_type, race_name)
    names = _material_names_for_area(sof, area_type)
    fallbacks = _slot_fallback_colors(sof, area_type, race_name, tint)
    slots = [
        _material_slot_visuals(name, _lookup_material_parameters(sof, name), fallbacks[index], area_type, default_roughness)
        for index, name in enumerate(names)
    ]
    while len(slots) < 4:
        index = len(slots)
        slots.append(_material_slot_visuals("", {}, fallbacks[index], area_type, default_roughness))
    return slots[:4], tint, glow, detail_scale


def _exact_parameter(parameters: object, name: str) -> object | None:
    if not isinstance(parameters, dict):
        return None
    wanted = _normal_key(name)
    for key, value in parameters.items():
        if _normal_key(str(key)) == wanted:
            return value
    return None


def _gloss_to_roughness(gloss: float, fallback: float) -> float:
    if gloss != gloss:
        return fallback
    if gloss <= 1.0:
        return _clamp(1.0 - gloss, 0.035, 0.98)
    return _clamp((2.0 / (gloss + 2.0)) ** 0.5, 0.035, 0.98)


def _resolved_area_material_slots(
    sof: dict,
    area: dict,
    area_type: str,
    race_name: str,
) -> tuple[list[dict[str, object]], tuple[float, float, float], tuple[float, float, float], float, float, bool, int]:
    fallback_slots, tint, fallback_glow, detail_scale = _area_material_slots(sof, area_type, race_name)
    parameters = area.get("resolvedParameters") if isinstance(area.get("resolvedParameters"), dict) else {}
    names = area.get("materialNames") if isinstance(area.get("materialNames"), list) else []
    prefixes = area.get("materialPrefixes") if isinstance(area.get("materialPrefixes"), list) else []
    while len(prefixes) < 4:
        prefixes.append(f"Mtl{len(prefixes) + 1}")

    slots: list[dict[str, object]] = []
    unresolved = 0
    for index in range(4):
        prefix = str(prefixes[index] or f"Mtl{index + 1}")
        fallback = fallback_slots[index]
        color_value = _numeric_vector(_exact_parameter(parameters, f"{prefix}DiffuseColor"))
        f0_value = _numeric_vector(_exact_parameter(parameters, f"{prefix}FresnelColor"))
        gloss_value = _numeric_scalar(_exact_parameter(parameters, f"{prefix}Gloss"))
        if color_value is None:
            unresolved += 1
        if f0_value is None:
            unresolved += 1
        if gloss_value is None:
            unresolved += 1
        slots.append({
            "name": str(names[index]) if index < len(names) else str(fallback.get("name", "")),
            "color": tuple(_clamp(value, 0.0, 4.0) for value in color_value[:3]) if color_value else fallback["color"],
            "f0": tuple(_clamp(value, 0.0, 1.0) for value in f0_value[:3]) if f0_value else fallback["f0"],
            "roughness": _gloss_to_roughness(gloss_value, float(fallback["roughness"])) if gloss_value is not None else fallback["roughness"],
            "gloss": _clamp(gloss_value, 0.0, 256.0) if gloss_value is not None else float(fallback.get("gloss", 1.0 - float(fallback["roughness"]))),
        })

    glow_value = _numeric_vector(_exact_parameter(parameters, "GeneralGlowColor"))
    glow = tuple(_clamp(value, 0.0, 8.0) for value in glow_value[:3]) if glow_value else fallback_glow
    # The explicit slot checks above already account for unresolved visible
    # parameters. Do not count the converter's unresolved list a second time.
    general_data = _numeric_vector(_exact_parameter(parameters, "GeneralData"))
    general_data_x = float(general_data[0]) if general_data is not None else 1.0
    return slots, tint, glow, detail_scale, general_data_x, unresolved == 0, unresolved

def _texture_usage(textures: dict, candidates: Iterable[str]) -> str:
    wanted = {_normal_key(value) for value in candidates}
    for key, value in textures.items():
        if _normal_key(str(key)) in wanted and isinstance(value, str):
            return value
    for key, value in textures.items():
        normalized = _normal_key(str(key))
        if any(candidate in normalized for candidate in wanted) and isinstance(value, str):
            return value
    return ""


SEMANTIC_CHANNEL_RED = 0
SEMANTIC_CHANNEL_GREEN = 1
SEMANTIC_CHANNEL_BLUE = 2
SEMANTIC_CHANNEL_ALPHA = 3


def _texture_suffix(logical: str) -> str:
    stem = Path(str(logical or '').replace('\\', '/')).stem.lower()
    for suffix in ('_pmdg', '_ar', '_no', '_pgs', '_pgr', '_ap', '_d', '_n'):
        if stem.endswith(suffix):
            return suffix
    return ''


def _texture_by_suffix(textures: dict, suffix: str) -> str:
    for value in textures.values():
        if isinstance(value, str) and _texture_suffix(value) == suffix:
            return value
    return ''


def _classify_shader_family(area: dict, textures: dict) -> str:
    shader = str(area.get('shader') or '').replace('\\', '/').lower()
    suffixes = {_texture_suffix(value) for value in textures.values() if isinstance(value, str)}
    if {'_ar', '_no', '_pmdg'} <= suffixes:
        return 'v5_packed'
    if any(_normal_key(str(key)) == 'pmdgmap' for key in textures) or '_pmdg' in suffixes:
        return 'v5_packed'
    separate_semantics = ('RoughnessMap', 'PaintMaskMap', 'MaterialMap', 'DirtMap', 'GlowMap')
    if any(_texture_usage(textures, (name,)) for name in separate_semantics):
        return 'v5_separate'
    if _texture_usage(textures, ('PgsMap',)) or '_pgs' in suffixes:
        return 'legacy_pgs'
    if '_pgr' in suffixes or '_ap' in suffixes or 'v5' in shader:
        return 'v5_separate'
    return 'unknown'


def _semantic_texture_layout(area: dict, textures: dict) -> dict[str, object]:
    family = _classify_shader_family(area, textures)
    direct = {
        'albedo': _texture_usage(textures, ('AlbedoMap', 'DiffuseMap', 'DiffuseMap1', 'DetailMap')),
        'normal': _texture_usage(textures, ('NormalMap', 'NormalMap1')),
        'material': _texture_usage(textures, ('MaterialMap', 'PgsMap')),
        'glow': _texture_usage(textures, ('GlowMap', 'EmissiveMap')),
        'dirt': _texture_usage(textures, ('DirtMap', 'GrimeMap')),
        'ao': _texture_usage(textures, ('AoMap', 'AmbientOcclusionMap')),
        'paintMask': _texture_usage(textures, ('PaintMaskMap', 'PaintMask')),
        'roughnessMap': _texture_usage(textures, ('RoughnessMap',)),
    }
    channels = {
        'normalX': SEMANTIC_CHANNEL_RED, 'normalY': SEMANTIC_CHANNEL_GREEN,
        'roughness': SEMANTIC_CHANNEL_RED, 'material': SEMANTIC_CHANNEL_RED,
        'ao': SEMANTIC_CHANNEL_RED, 'paint': SEMANTIC_CHANNEL_RED,
        'dirt': SEMANTIC_CHANNEL_RED, 'glow': SEMANTIC_CHANNEL_RED,
    }
    required = ['albedo', 'normal', 'material', 'roughnessMap']

    if family == 'v5_packed':
        ar = _texture_by_suffix(textures, '_ar') or direct['albedo']
        no = _texture_by_suffix(textures, '_no') or direct['normal']
        pmdg = _texture_by_suffix(textures, '_pmdg') or direct['material']
        direct.update({
            'albedo': ar, 'roughnessMap': ar,
            'normal': no, 'ao': no,
            'material': pmdg, 'paintMask': pmdg, 'dirt': pmdg, 'glow': pmdg,
        })
        channels.update({
            'normalX': SEMANTIC_CHANNEL_ALPHA, 'normalY': SEMANTIC_CHANNEL_GREEN,
            'roughness': SEMANTIC_CHANNEL_ALPHA, 'material': SEMANTIC_CHANNEL_GREEN,
            'ao': SEMANTIC_CHANNEL_BLUE, 'paint': SEMANTIC_CHANNEL_RED,
            'dirt': SEMANTIC_CHANNEL_BLUE, 'glow': SEMANTIC_CHANNEL_ALPHA,
        })
        required = ['albedo', 'normal', 'material']
    elif family == 'legacy_pgs':
        # Pre-PBR PGS: R=sub-mask, G=specular and B=material mask. The
        # authored hull colour remains in the selected _d texture; PGS must not
        # be rebound as a generic emissive texture or used to recolour it.
        channels.update({
            'normalX': SEMANTIC_CHANNEL_ALPHA, 'normalY': SEMANTIC_CHANNEL_GREEN,
            'roughness': SEMANTIC_CHANNEL_GREEN, 'material': SEMANTIC_CHANNEL_BLUE,
            'ao': SEMANTIC_CHANNEL_RED, 'paint': SEMANTIC_CHANNEL_BLUE,
            'dirt': SEMANTIC_CHANNEL_RED, 'glow': SEMANTIC_CHANNEL_ALPHA,
        })
        required = ['albedo', 'normal', 'material']

    missing = [name for name in required if not direct.get(name)]
    return {
        **direct, 'shaderFamily': family, 'channels': channels,
        'requiredSemantics': required, 'missingSemantics': missing,
        'semanticComplete': not missing and family != 'unknown',
    }


def _material_manifest_columns() -> list[str]:
    columns = [
        "group", "pass", "area_type", "area_name", "shader", "shader_family",
        "albedo", "normal", "material", "glow", "dirt", "ao", "paint_mask", "roughness_map",
        "normal_x_channel", "normal_y_channel", "roughness_channel", "material_channel",
        "ao_channel", "paint_channel", "dirt_channel", "glow_channel",
        "tint_r", "tint_g", "tint_b", "detail_scale", "alpha", "glow_r", "glow_g", "glow_b",
    ]
    for index in range(1, 5):
        columns.extend([
            f"mtl{index}_r", f"mtl{index}_g", f"mtl{index}_b",
            f"mtl{index}_f0_r", f"mtl{index}_f0_g", f"mtl{index}_f0_b",
            f"mtl{index}_gloss",
        ])
    columns.extend(["general_data_x", "semantic_complete", "baseline_complete", "unresolved_count", "unresolved_semantics"])
    return columns


def _write_material_record(writer: csv.writer, record: dict[str, object]) -> None:
    tint = record["tint"]
    glow = record["glowColor"]
    row: list[object] = [
        record["group"], record["pass"], record["areaType"], record.get("areaName", ""),
        record.get("shader", ""), record.get("shaderFamily", "unknown"),
        record.get("albedo", ""), record.get("normal", ""), record.get("material", ""), record.get("glow", ""),
        record.get("dirt", ""), record.get("ao", ""), record.get("paintMask", ""), record.get("roughnessMap", ""),
        int(record.get("channels", {}).get("normalX", 0)), int(record.get("channels", {}).get("normalY", 1)),
        int(record.get("channels", {}).get("roughness", 0)), int(record.get("channels", {}).get("material", 0)),
        int(record.get("channels", {}).get("ao", 0)), int(record.get("channels", {}).get("paint", 0)),
        int(record.get("channels", {}).get("dirt", 0)), int(record.get("channels", {}).get("glow", 0)),
        f"{tint[0]:.7g}", f"{tint[1]:.7g}", f"{tint[2]:.7g}",
        f"{float(record['detailScale']):.7g}", f"{float(record['alpha']):.7g}",
        f"{glow[0]:.7g}", f"{glow[1]:.7g}", f"{glow[2]:.7g}",
    ]
    for slot in record["slots"]:
        color = slot["color"]
        f0 = slot["f0"]
        row.extend([
            f"{color[0]:.7g}", f"{color[1]:.7g}", f"{color[2]:.7g}",
            f"{f0[0]:.7g}", f"{f0[1]:.7g}", f"{f0[2]:.7g}",
            f"{float(slot.get('gloss', 1.0 - float(slot['roughness']))):.7g}",
        ])
    row.extend([
        f"{float(record.get('generalDataX', 1.0)):.7g}",
        "1" if record.get("semanticComplete", False) else "0",
        "1" if record.get("baselineComplete", False) else "0",
        str(int(record.get("unresolvedCount", 0))),
        ",".join(str(value) for value in record.get("missingSemantics", [])),
    ])
    writer.writerow(row)


def _prepare_sof_materials(
    repo_root: Path,
    rows: list[ResourceRow],
    resfiles: Path,
    output_dir: Path,
    conversion_summary: Path,
    sof_manifest_path: Path,
    race_name: str,
    prefer_faction_textures: bool = False,
    exact_source_textures: dict[str, Path] | None = None,
) -> tuple[Path, dict[str, dict]]:
    sof = json.loads(sof_manifest_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_summary.read_text(encoding="utf-8"))
    rows_by_logical = {row.logical.lower(): row for row in rows}
    res_path_insert = str(sof.get("resPathInsert") or "")
    materials_dir = output_dir / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    converted_cache: dict[str, Path] = {}
    copied_metadata: dict[str, dict] = {}
    exact_source_textures = exact_source_textures or {}
    lock_exact_legacy_source = (
        not prefer_faction_textures
        and bool(exact_source_textures.get("albedo"))
        and bool(exact_source_textures.get("normal"))
        and bool(exact_source_textures.get("pgs"))
    )
    if lock_exact_legacy_source:
        print(
            "Legacy source lock: exact selected _d/_n/_pgs; SOF area colours are not multiplied into authored albedo.",
            flush=True,
        )

    def prepare_texture(logical: str, usage: str) -> Path | None:
        if not logical:
            return None
        row = _resolve_sof_texture(
            rows_by_logical, resfiles, logical, res_path_insert, prefer_faction_textures
        )
        if row is None:
            print(f"WARNING: SOF texture not indexed: {logical}", flush=True)
            return None
        source = resfiles / Path(row.hashed)
        if not source.is_file():
            print(f"WARNING: SOF texture is not present locally: {row.logical}", flush=True)
            return None
        cache_key = row.logical.lower()
        if cache_key in converted_cache:
            metadata = copied_metadata.get(cache_key)
            if isinstance(metadata, dict):
                usages = metadata.setdefault("usages", [])
                if usage not in usages:
                    usages.append(usage)
                metadata["usage"] = ",".join(usages)
            return converted_cache[cache_key]
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row.logical).stem)
        unique_stem = f"{safe_stem}_{len(converted_cache):03d}"
        output = materials_dir / f"{unique_stem}.png"
        copied = materials_dir / f"{unique_stem}{Path(row.logical).suffix.lower()}"
        shutil.copy2(source, copied)
        convert_dds(repo_root, copied, output)
        converted_cache[cache_key] = output
        copied_metadata[cache_key] = {"usage": usage, "usages": [usage], **asdict(row), "local": str(copied), "converted": str(output)}
        return output

    draw_ranges = conversion.get("drawRanges", [])
    existing_groups = {int(draw.get("groupIndex", -1)) for draw in draw_ranges}
    # GR2 MaterialIndex values are SOF area indices. The OBJ group index is a
    # unique sequential draw identifier because the same SOF area can occur on
    # several model-bound meshes. Keep both values instead of assuming they are
    # interchangeable.
    draws_by_material_index: dict[int, list[dict]] = {}
    for draw in draw_ranges:
        group_index = int(draw.get("groupIndex", -1))
        material_index = int(draw.get("materialIndex", group_index))
        if group_index >= 0:
            draws_by_material_index.setdefault(material_index, []).append(draw)
    areas = sof.get("areas", []) if isinstance(sof.get("areas"), list) else []
    records: list[dict[str, object]] = []
    for area in areas:
        if not isinstance(area, dict):
            continue
        first_group = int(area.get("index") or 0)
        group_count = max(1, int(area.get("count") or 1))
        textures = area.get("textures") if isinstance(area.get("textures"), dict) else {}
        layout = _semantic_texture_layout(area, textures)
        pass_name = str(area.get("pass") or "opaque")
        area_shader_family = str(layout.get("shaderFamily") or "unknown")
        use_exact_legacy_area = (
            lock_exact_legacy_source
            and (
                area_shader_family == "legacy_pgs"
                or (area_shader_family == "unknown" and pass_name == "opaque")
            )
            and pass_name in {"opaque", "decal"}
        )
        texture_usages = {key: str(layout.get(key) or "") for key in
                          ("albedo", "normal", "material", "glow", "dirt", "ao", "paintMask", "roughnessMap")}
        prepared = {} if use_exact_legacy_area else {
            key: str(prepare_texture(value, key) or "") for key, value in texture_usages.items()
        }

        # The direct/base Raven preview previously rendered correctly from the
        # exact selected cb1_t1_d/_n/_pgs set. SOF extraction then replaced
        # those maps per area and multiplied the already-coloured _d texture by
        # material-library colours, producing the dark mixed-material result.
        # Keep SOF area/pass/F0/gloss data, but lock legacy base previews to the
        # exact selected texture triplet used before SOF extraction.
        if use_exact_legacy_area:
            layout = {
                **layout,
                "shaderFamily": "legacy_pgs",
                "requiredSemantics": ["albedo", "normal", "material"],
                "missingSemantics": [],
                "semanticComplete": True,
                "channels": {
                    "normalX": SEMANTIC_CHANNEL_ALPHA,
                    "normalY": SEMANTIC_CHANNEL_GREEN,
                    "roughness": SEMANTIC_CHANNEL_GREEN,
                    "material": SEMANTIC_CHANNEL_BLUE,
                    "ao": SEMANTIC_CHANNEL_RED,
                    "paint": SEMANTIC_CHANNEL_BLUE,
                    "dirt": SEMANTIC_CHANNEL_RED,
                    "glow": SEMANTIC_CHANNEL_ALPHA,
                },
            }
            prepared.update({
                "albedo": str(exact_source_textures["albedo"].resolve()),
                "normal": str(exact_source_textures["normal"].resolve()),
                "material": str(exact_source_textures["pgs"].resolve()),
                "glow": "",
                "dirt": "",
                "ao": "",
                "paintMask": "",
                "roughnessMap": "",
            })
            texture_usages.update({
                "albedo": "exact-selected-_d",
                "normal": "exact-selected-_n",
                "material": "exact-selected-_pgs",
                "glow": "",
                "dirt": "",
                "ao": "",
                "paintMask": "",
                "roughnessMap": "",
            })

        required_semantics = list(layout.get("requiredSemantics") or [])
        unresolved_declared = [key for key, value in texture_usages.items() if value and not prepared.get(key)]
        missing_semantics = list(dict.fromkeys(list(layout.get("missingSemantics") or []) + unresolved_declared))
        area_type = str(area.get("areaType") or "primary").replace("TYPE_", "").lower()
        slots, tint, glow_color, detail_scale, general_data_x, parameter_complete, unresolved_count = _resolved_area_material_slots(
            sof, area, area_type, race_name
        )
        if use_exact_legacy_area:
            # Legacy _d is an authored colour texture. Preserve SOF surface
            # parameters, but do not multiply its RGB by a second colour set.
            slots = [{**slot, "color": (1.0, 1.0, 1.0)} for slot in slots]
            tint = (1.0, 1.0, 1.0)
        alpha = 0.42 if pass_name == "transparent" else 1.0
        semantic_complete = bool(layout.get("semanticComplete")) and all(prepared.get(name) for name in required_semantics) and not unresolved_declared
        baseline_complete = parameter_complete and semantic_complete
        matching_draws: list[dict] = []
        for material_index in range(first_group, first_group + group_count):
            matching_draws.extend(draws_by_material_index.get(material_index, []))
        for draw in matching_draws:
            group_index = int(draw.get("groupIndex", -1))
            if group_index not in existing_groups:
                continue
            records.append({
                "group": group_index,
                "pass": pass_name,
                "areaType": area_type,
                "areaName": str(area.get("name") or ""),
                "shader": str(area.get("shader") or ""),
                "shaderFamily": str(layout.get("shaderFamily") or "unknown"),
                "channels": dict(layout.get("channels") or {}),
                "blockedMaterials": int(area.get("blockedMaterials") or 0),
                "materialNames": list(area.get("materialNames") or []),
                "materialLibraryMatches": list(area.get("materialLibraryMatches") or []),
                "unresolvedParameters": list(area.get("unresolvedParameters") or []),
                "parameterSources": dict(area.get("parameterSources") or {}),
                "sourceTexturePolicy": "exact-selected-legacy" if use_exact_legacy_area else "sof-area",
                **prepared,
                "tint": tint,
                "glowColor": glow_color,
                "detailScale": detail_scale,
                "generalDataX": general_data_x,
                "alpha": alpha,
                "slots": slots,
                "semanticComplete": semantic_complete,
                "parameterComplete": parameter_complete,
                "baselineComplete": baseline_complete,
                "missingSemantics": missing_semantics,
                "unresolvedCount": unresolved_count + len(missing_semantics),
            })

    assigned_groups = {int(record["group"]) for record in records}
    fallback_slots, fallback_tint, fallback_glow, fallback_scale = _area_material_slots(sof, "primary", race_name)
    for draw in draw_ranges:
        group_index = int(draw.get("groupIndex", 0))
        if group_index in assigned_groups:
            continue
        records.append({
            "group": group_index, "pass": "opaque", "areaType": "primary", "areaName": "unresolved", "shader": "",
            "shaderFamily": "unknown", "channels": {},
            "albedo": "", "normal": "", "material": "", "glow": "", "dirt": "", "ao": "", "paintMask": "", "roughnessMap": "",
            "tint": fallback_tint, "glowColor": fallback_glow, "detailScale": fallback_scale, "generalDataX": 1.0, "alpha": 1.0,
            "slots": fallback_slots,
            "semanticComplete": False, "parameterComplete": False, "baselineComplete": False,
            "missingSemantics": ["area_assignment"], "unresolvedCount": 1,
        })

    records.sort(key=lambda value: ({"opaque": 0, "decal": 1, "transparent": 2, "additive": 3}.get(str(value["pass"]), 4), int(value["group"])))
    manifest_path = output_dir / "ship.materials.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V3\n")
        handle.write("# Shader-family and channel columns define the effective semantic inputs.\n# V5 packed: AR.rgb/albedo, AR.a/roughness, NO.a+g/normal, NO.b/AO, PMDG.r/g/b/a=paint/material/dirt/glow.\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_material_manifest_columns())
        for record in records:
            _write_material_record(writer, record)

    baseline_report = {
        "schema": "NSAMDR_BASELINE_REPORT_V1",
        "complete": bool(records) and all(bool(record.get("baselineComplete")) for record in records),
        "unresolvedCount": sum(int(record.get("unresolvedCount", 0)) for record in records),
        "extractionDiagnostics": sof.get("extractionDiagnostics", {}),
        "areas": [
            {
                "group": int(record["group"]),
                "areaName": record.get("areaName", ""),
                "areaType": record.get("areaType", ""),
                "pass": record.get("pass", ""),
                "shader": record.get("shader", ""),
                "shaderFamily": record.get("shaderFamily", "unknown"),
                "channels": record.get("channels", {}),
                "blockedMaterials": int(record.get("blockedMaterials", 0)),
                "materialNames": record.get("materialNames", []),
                "materialLibraryMatches": record.get("materialLibraryMatches", []),
                "unresolvedParameters": record.get("unresolvedParameters", []),
                "parameterSources": record.get("parameterSources", {}),
                "sourceTexturePolicy": record.get("sourceTexturePolicy", "sof-area"),
                "semanticComplete": bool(record.get("semanticComplete")),
                "parameterComplete": bool(record.get("parameterComplete")),
                "baselineComplete": bool(record.get("baselineComplete")),
                "missingSemantics": record.get("missingSemantics", []),
                "textures": {name: record.get(name, "") for name in
                             ("albedo", "normal", "material", "roughnessMap", "ao", "paintMask", "dirt", "glow")},
            }
            for record in records
        ],
    }
    report_path = output_dir / "ship.materials.report.json"
    report_path.write_text(json.dumps(baseline_report, indent=2) + "\n", encoding="utf-8")
    print(f"Material baseline report: {report_path} complete={baseline_report['complete']} unresolved={baseline_report['unresolvedCount']}", flush=True)
    return manifest_path, copied_metadata


def _write_extracted_texture_material_manifest(
    output_dir: Path,
    conversion_summary: Path,
    race_name: str,
    reason: str,
    albedo: Path | None,
    normal: Path | None,
    pgs: Path | None,
) -> Path:
    """Build a usable fallback from the selected real DDS textures when SOF is unavailable.

    This is deliberately not marked as a complete production-EVE material baseline: SOF
    area assignments, faction colours and material-library parameters remain unresolved.
    It does, however, keep the authored albedo and normal maps bound on every GR2 draw
    range so Mode 1 is a real extracted-texture baseline and Mode 3 has material inputs
    to reconstruct instead of becoming a 1x1 neutral fallback.
    """
    conversion = json.loads(conversion_summary.read_text(encoding="utf-8"))
    draw_ranges = conversion.get("drawRanges", [])
    if not isinstance(draw_ranges, list) or not draw_ranges:
        raise RuntimeError("Cannot build extracted-texture fallback: conversion summary has no draw ranges")

    has_albedo = bool(albedo and albedo.is_file())
    has_normal = bool(normal and normal.is_file())
    has_pgs = bool(pgs and pgs.is_file())
    shader_family = "legacy_pgs" if has_pgs else "unknown"
    channels = {
        # EVE's legacy _n textures use DXT5nm-style A/G normal storage.
        "normalX": SEMANTIC_CHANNEL_ALPHA,
        "normalY": SEMANTIC_CHANNEL_GREEN,
        "roughness": SEMANTIC_CHANNEL_GREEN,
        "material": SEMANTIC_CHANNEL_BLUE,
        "ao": SEMANTIC_CHANNEL_RED,
        "paint": SEMANTIC_CHANNEL_BLUE,
        "dirt": SEMANTIC_CHANNEL_RED,
        "glow": SEMANTIC_CHANNEL_ALPHA,
    }
    neutral_slots = [
        {
            "name": "extracted-texture-fallback",
            "color": (1.0, 1.0, 1.0),
            "f0": (0.04, 0.04, 0.04),
            "roughness": 0.52,
            "gloss": 0.48,
        }
        for _ in range(4)
    ]
    missing = ["sof_visual_manifest"]
    if not has_albedo:
        missing.append("albedo")
    if not has_normal:
        missing.append("normal")

    manifest_path = output_dir / "ship.materials.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V3\n")
        handle.write(f"# extracted-texture fallback; production SOF material data unresolved: {_tsv_clean(reason)}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_material_manifest_columns())
        for draw in draw_ranges:
            _write_material_record(writer, {
                "group": int(draw.get("groupIndex", 0)),
                "pass": "opaque",
                "areaType": "primary",
                "areaName": "extracted-texture-fallback",
                "shader": "",
                "shaderFamily": shader_family,
                "channels": channels,
                "albedo": str(albedo.resolve()) if has_albedo and albedo else "",
                "normal": str(normal.resolve()) if has_normal and normal else "",
                "material": str(pgs.resolve()) if has_pgs and pgs else "",
                "glow": "",
                "dirt": "",
                "ao": "",
                "paintMask": "",
                "roughnessMap": "",
                # The selected _d texture already carries its authored colour. Avoid
                # multiplying it by guessed faction tints when SOF is unavailable.
                "tint": (1.0, 1.0, 1.0),
                "glowColor": (0.34, 0.58, 0.95),
                "detailScale": 1.0,
                "generalDataX": 1.0,
                "alpha": 1.0,
                "slots": neutral_slots,
                "semanticComplete": has_albedo and has_normal,
                "parameterComplete": False,
                "baselineComplete": False,
                "missingSemantics": missing,
                "unresolvedCount": len(missing),
            })

    report_path = output_dir / "ship.materials.report.json"
    report_path.write_text(json.dumps({
        "schema": "NSAMDR_BASELINE_REPORT_V1",
        "complete": False,
        "fallback": "real-extracted-textures",
        "unresolvedCount": len(draw_ranges) * len(missing),
        "reason": reason,
        "textures": {
            "albedo": str(albedo.resolve()) if has_albedo and albedo else "",
            "normal": str(normal.resolve()) if has_normal and normal else "",
            "material": str(pgs.resolve()) if has_pgs and pgs else "",
            "glow": "",
        },
        "areas": [
            {
                "group": int(draw.get("groupIndex", 0)),
                "areaName": "extracted-texture-fallback",
                "areaType": "primary",
                "pass": "opaque",
                "shader": "",
                "shaderFamily": shader_family,
                "semanticComplete": has_albedo and has_normal,
                "parameterComplete": False,
                "baselineComplete": False,
                "missingSemantics": missing,
            }
            for draw in draw_ranges
        ],
    }, indent=2) + "\n", encoding="utf-8")

    texture_summary = ", ".join((
        f"albedo={'yes' if has_albedo else 'no'}",
        f"normal={'yes' if has_normal else 'no'}",
        f"material={'yes' if has_pgs else 'no'}",
    ))
    print(
        f"WARNING: SOF production material extraction is unavailable; created a real extracted-texture "
        f"fallback ({texture_summary}; {reason}): {manifest_path}",
        flush=True,
    )
    return manifest_path


def prepare_asset(
    repo_root: Path,
    cache_arg: str,
    query: str,
    selection_key: str = "",
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, list[Path], Path | None, Path, Path, Path]:
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
    environment_scene, environment_textures = select_environment_sources(rows, model, resfiles)
    environment_texture = environment_textures[0] if environment_textures else None
    print(f"Selected model: {model.logical}", flush=True)
    for kind, row in textures.items():
        print(f"Selected {kind}: {row.logical}", flush=True)
    if environment_scene:
        print(f"Selected EVE environment scene: {environment_scene.logical}", flush=True)
    if environment_texture:
        print(f"Selected EVE environment texture: {environment_texture.logical}", flush=True)
        print(f"Available local EVE backgrounds: {len(environment_textures)}", flush=True)
    else:
        print("No local EVE nebula texture was available; the viewer will use its procedural fallback.", flush=True)

    asset_name = Path(model.logical.rsplit("/", 1)[-1]).stem
    output_dir = repo_root / "artifacts" / "nsamdr" / "eve_assets" / asset_name
    gr2_path = copy_resource(resfiles, model, output_dir)
    copied_environment_scene = copy_resource(resfiles, environment_scene, output_dir) if environment_scene else None
    copied_environment_textures: list[tuple[ResourceRow, Path]] = []
    for index, row in enumerate(environment_textures):
        copied_environment_textures.append((row, copy_resource(resfiles, row, output_dir / "backgrounds")))
    copied_environment_texture = copied_environment_textures[0][1] if copied_environment_textures else None
    copied: dict[str, Path] = {}
    for kind, row in textures.items():
        copied[kind] = copy_resource(resfiles, row, output_dir)

    # Convert the exact selected texture family before SOF material assembly so
    # a direct/base preview can lock every legacy area to this known identity.
    converted_textures: dict[str, Path] = {}
    for kind, copied_path in copied.items():
        try:
            converted_textures[kind] = convert_dds(
                repo_root,
                copied_path,
                output_dir / f"{asset_name}_{kind}.png",
            )
        except RuntimeError as exc:
            print(f"WARNING: {kind} texture conversion failed: {exc}", flush=True)

    obj_path = convert_gr2(repo_root, gr2_path, output_dir / f"{asset_name}.obj")
    conversion_summary = obj_path.with_suffix(".conversion.json")
    conversion_record = json.loads(conversion_summary.read_text(encoding="utf-8"))
    if conversion_record.get("schema") != "NSAMDR_GR2_CONVERSION_V5_BAKED_EVE_TEXTURE_V":
        raise RuntimeError(
            "GR2 conversion did not use the LOD-collapsing renderer. "
            f"Found schema={conversion_record.get('schema')!r}."
        )
    selected_meshes = list(conversion_record.get("selectedMeshIndices") or [])
    rejected_lods = list(conversion_record.get("rejectedLodMeshIndices") or [])
    if not selected_meshes:
        raise RuntimeError("GR2 conversion selected no render meshes")
    print(
        "GR2 LOD selection: "
        f"selected={selected_meshes}, rejectedAlternatives={rejected_lods}, "
        f"draws={len(conversion_record.get('drawRanges') or [])}",
        flush=True,
    )

    material_manifest: Path | None = None
    sof_manifest_path: Path | None = None
    sof_texture_metadata: dict[str, dict] = {}
    sof_identity = _resolve_sof_identity(rows, repo_root, model, selection_key)
    print(
        "SOF identity: "
        f"hull={sof_identity.get('hull') or '<missing>'}, "
        f"faction={sof_identity.get('faction') or '<missing>'}, "
        f"race={sof_identity.get('race') or '<missing>'} "
        f"({sof_identity.get('raceSource') or 'unknown'}; "
        f"identity={sof_identity.get('identitySource') or 'unknown'}; "
        f"textures={'faction-first' if sof_identity.get('preferFactionTextures') else 'source-first'})",
        flush=True,
    )
    data_black_row = next((row for row in rows if row.logical.lower() == SOF_DATA_PATH), None)
    if not data_black_row:
        raise RuntimeError("SOF data.black is unavailable; refusing to render an invented material fallback")
    if not sof_identity.get("hull") or not sof_identity.get("faction"):
        raise RuntimeError(
            "SOF hull/faction identity is unresolved; refusing to render an invented material fallback"
        )

    data_black = copy_resource(resfiles, data_black_row, output_dir / "sof")
    try:
        sof_manifest_path = convert_sof(
            repo_root, data_black, output_dir / f"{asset_name}.sof-visuals.json",
            str(sof_identity["hull"]), str(sof_identity["faction"]),
            str(sof_identity.get("race") or ""), model.logical,
        )
        material_manifest, sof_texture_metadata = _prepare_sof_materials(
            repo_root, rows, resfiles, output_dir, conversion_summary, sof_manifest_path,
            str(sof_identity.get("race") or ""),
            bool(sof_identity.get("preferFactionTextures", False)),
            converted_textures,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        for stale in (
            output_dir / f"{asset_name}.sof-visuals.json",
            output_dir / "ship.materials.tsv",
            output_dir / "ship.materials.report.json",
        ):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        raise RuntimeError(
            "SOF production material extraction failed; preview launch is blocked rather than "
            f"showing the known-wrong shared-texture fallback. {exc}"
        ) from exc

    report_path = output_dir / "ship.materials.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("complete")):
        unresolved = int(report.get("unresolvedCount") or 0)
        raise RuntimeError(
            "SOF material baseline remains incomplete "
            f"(unresolved={unresolved}); preview launch is blocked."
        )
    print(f"SOF material manifest: {material_manifest}", flush=True)

    environment_pngs: list[Path] = []
    environment_records: list[dict] = []
    for index, (row, copied_path) in enumerate(copied_environment_textures):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row.logical).stem)
        output_path = output_dir / "backgrounds" / f"{index:03d}_{safe_name}.png"
        try:
            converted = convert_environment_dds(repo_root, copied_path, output_path)
            environment_pngs.append(converted)
            environment_records.append({**asdict(row), "local": str(copied_path), "converted": str(converted)})
        except RuntimeError as exc:
            print(f"WARNING: EVE environment conversion failed for {row.logical}: {exc}", flush=True)
    environment_png = environment_pngs[0] if environment_pngs else None

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
        "conversionSummary": str(conversion_summary),
        "sofIdentity": sof_identity,
        "sofVisualManifest": str(sof_manifest_path) if sof_manifest_path else None,
        "materialManifest": str(material_manifest) if material_manifest else None,
        "materialBaselineReport": str(output_dir / "ship.materials.report.json") if (output_dir / "ship.materials.report.json").is_file() else None,
        "sofTextures": sof_texture_metadata,
        "albedoPng": str(converted_textures.get("albedo")) if converted_textures.get("albedo") else None,
        "normalPng": str(converted_textures.get("normal")) if converted_textures.get("normal") else None,
        "pgsPng": str(converted_textures.get("pgs")) if converted_textures.get("pgs") else None,
        "environment": {
            "scene": {**asdict(environment_scene), "local": str(copied_environment_scene)} if environment_scene else None,
            "texture": {**asdict(environment_texture), "local": str(copied_environment_texture)} if environment_texture else None,
            "converted": str(environment_png) if environment_png else None,
        },
        "environments": environment_records,
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
        environment_pngs,
        material_manifest,
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
    environments: list[Path],
    material_manifest: Path | None,
    manifest: Path,
    catalog: Path,
    cache_root: Path,
    current_query: str,
    strategy_candidates: dict[str, object] | None = None,
) -> int:
    if not launcher.is_file():
        raise RuntimeError(f"Missing preview launcher: {launcher}")
    command = [
        os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call",
        str(launcher), str(obj_path), str(albedo or ""), str(normal or ""), str(pgs or ""), str(environment or ""), str(material_manifest or ""),
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
        "NSAMDR_ENVIRONMENTS": ";".join(str(path) for path in environments),
        "NSAMDR_MATERIALS": str(material_manifest or ""),
        "NSAMDR_PYTHON_EXE": sys.executable,
    })
    if strategy_candidates:
        env.update({
            "NSAMDR_MODE3_OBJ": str(strategy_candidates.get("mode3Obj", "")),
            "NSAMDR_MODE3_MATERIALS": str(strategy_candidates.get("mode3Materials", "")),
            "NSAMDR_MODE3_ANALYSIS": str(strategy_candidates.get("mode3Analysis", "")),
            "NSAMDR_MODE3_VALIDATION": str(strategy_candidates.get("mode3Validation", "")),
            "NSAMDR_STRATEGY_CANDIDATES": str(Path(str(strategy_candidates.get("reportPath", ""))) if strategy_candidates.get("reportPath") else ""),
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
    obj_path, albedo, normal, pgs, environment, environments, material_manifest, manifest, catalog, cache_root = prepare_asset(
        repo_root, args.shared_cache, args.query, args.selection_key
    )
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    selected_query = str(manifest_data.get("model", {}).get("logical") or args.query)
    selected_catalog_key = str(args.selection_key or selected_query)

    strategy_candidates: dict[str, object] | None = None
    if material_manifest and material_manifest.is_file():
        generator = repo_root / "tools" / "nsamdr" / "generate_strategy_candidates.py"
        candidate_root = obj_path.parent / "strategy_candidates_4096"
        candidate_report = candidate_root / "strategy_candidates.json"
        if generator.is_file():
            neural_python_candidates = [
                repo_root / "artifacts" / "nsamdr" / "python-env" / "Scripts" / "python.exe",
                repo_root / "artifacts" / "nsamdr" / "python-env-cpu" / "Scripts" / "python.exe",
            ]
            candidate_python = next((path for path in neural_python_candidates if path.is_file()), Path(sys.executable))
            command = [
                str(candidate_python),
                str(generator),
                "--obj", str(obj_path),
                "--materials", str(material_manifest),
                "--asset-manifest", str(manifest),
                "--output-root", str(candidate_root),
                "--target-size", "4096",
                "--install-dependencies",
            ]
            print("Preparing the public Mode 3 NSAMDR candidate...", flush=True)
            result = subprocess.run(command, cwd=repo_root, check=False)
            if result.returncode == 0 and candidate_report.is_file():
                strategy_candidates = json.loads(candidate_report.read_text(encoding="utf-8"))
                strategy_candidates["reportPath"] = str(candidate_report)
            else:
                print(
                    "WARNING: Mode 3 candidate generation failed. Modes 1 and 2 remain available; Mode 3 will report the missing candidate.",
                    file=sys.stderr,
                    flush=True,
                )

    return launch_preview(
        repo_root, Path(args.launcher).resolve(), obj_path, albedo, normal, pgs, environment, environments, material_manifest,
        manifest, catalog, cache_root, selected_catalog_key, strategy_candidates
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
