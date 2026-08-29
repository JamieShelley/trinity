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
EXPECTED_GR2_CONVERSION_SCHEMA = "NSAMDR_GR2_CONVERSION_V6_EVE_DIRECTX_LH_HANDEDNESS"


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


class EveAssetValidationApplication:
    # Purpose: Implement eprint for EveAssetValidationApplication.
    # Called by: main, resolve_layout
    # Calls: No same-class helper methods.
    def eprint(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    # Purpose: Implement add candidate for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: No same-class helper methods.
    def _add_candidate(self, values: list[Path], value: str | Path | None) -> None:
        if value is None:
            return
        text = str(value).strip().strip('"')
        if text:
            values.append(Path(text))

    # Purpose: Implement registry cache roots for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: No same-class helper methods.
    def _registry_cache_roots(self) -> Iterable[Path]:
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

    # Purpose: Implement read registry string for EveAssetValidationApplication.
    # Called by: _installed_program_roots
    # Calls: No same-class helper methods.
    def _read_registry_string(self, key, name: str):
        try:
            import winreg  # type: ignore[import-not-found]
            value, kind = winreg.QueryValueEx(key, name)
        except (ImportError, OSError):
            return None
        if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) or not isinstance(value, str):
            return None
        value = os.path.expandvars(value.strip().strip('"'))
        return value or None

    # Purpose: Implement installed program roots for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: _read_registry_string
    def _installed_program_roots(self) -> Iterable[Path]:
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
                                        display_name = (self._read_registry_string(sub, "DisplayName") or "").lower()
                                        publisher = (self._read_registry_string(sub, "Publisher") or "").lower()
                                        combined = f"{display_name} {publisher} {sub_name.lower()}"
                                        if not (
                                            "eve online" in combined
                                            or "eve launcher" in combined
                                            or ("ccp" in combined and "eve" in combined)
                                        ):
                                            continue
                                        for value_name in ("InstallLocation", "InstallSource", "DisplayIcon", "UninstallString"):
                                            candidate = emit(self._read_registry_string(sub, value_name))
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
                                candidate = emit(self._read_registry_string(key, name))
                                if candidate:
                                    yield candidate
                    except OSError:
                        continue

    # Purpose: Implement launcher config cache roots for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: No same-class helper methods.
    def _launcher_config_cache_roots(self) -> Iterable[Path]:
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

    # Purpose: Implement launcher log cache roots for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: No same-class helper methods.
    def _launcher_log_cache_roots(self) -> Iterable[Path]:
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

    # Purpose: Implement saved cache roots for EveAssetValidationApplication.
    # Called by: candidate_cache_roots
    # Calls: No same-class helper methods.
    def _saved_cache_roots(self) -> Iterable[Path]:
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

    # Purpose: Implement save cache root for EveAssetValidationApplication.
    # Called by: resolve_layout
    # Calls: No same-class helper methods.
    def _save_cache_root(self, root: Path) -> None:
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            return
        try:
            path = Path(local) / "NSAMDR" / "eve_shared_cache.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(root) + "\n", encoding="utf-8")
        except OSError:
            pass

    # Purpose: Implement candidate cache roots for EveAssetValidationApplication.
    # Called by: resolve_layout
    # Calls: _add_candidate, _installed_program_roots, _launcher_config_cache_roots, _launcher_log_cache_roots, _registry_cache_roots, _saved_cache_roots
    def candidate_cache_roots(self) -> Iterable[Path]:
        seen: set[str] = set()
        values: list[Path] = []

        self._add_candidate(values, os.environ.get("EVE_SHARED_CACHE"))
        for value in self._saved_cache_roots():
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

        for value in self._registry_cache_roots() or ():
            values.append(value)
        for value in self._installed_program_roots() or ():
            values.append(value)
        for value in self._launcher_config_cache_roots():
            values.append(value)
        for value in self._launcher_log_cache_roots():
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

    # Purpose: Implement layout variants for EveAssetValidationApplication.
    # Called by: resolve_layout
    # Calls: No same-class helper methods.
    def _layout_variants(self, candidate: Path) -> Iterable[Path]:
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

    # Purpose: Implement inspect layout for EveAssetValidationApplication.
    # Called by: resolve_layout
    # Calls: No same-class helper methods.
    def _inspect_layout(self, root: Path) -> tuple[Path, list[Path], Path] | None:
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

    # Purpose: Implement prompt for cache for EveAssetValidationApplication.
    # Called by: resolve_layout
    # Calls: No same-class helper methods.
    def _prompt_for_cache(self) -> Path | None:
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

    # Purpose: Implement resolve layout for EveAssetValidationApplication.
    # Called by: command_list, prepare_asset
    # Calls: _inspect_layout, _layout_variants, _prompt_for_cache, _save_cache_root, candidate_cache_roots, eprint
    def resolve_layout(self, raw: str | None, *, allow_prompt: bool = False) -> tuple[Path, list[Path], Path]:
        candidates: list[Path] = []
        if raw and raw.strip():
            candidates.append(Path(raw.strip().strip('"')))
        candidates.extend(self.candidate_cache_roots())

        checked: list[str] = []
        seen_checked: set[str] = set()

        def try_candidate(candidate: Path) -> tuple[Path, list[Path], Path] | None:
            for root in self._layout_variants(candidate):
                key = os.path.normcase(os.path.normpath(str(root)))
                if key in seen_checked:
                    continue
                seen_checked.add(key)
                checked.append(str(root))
                result = self._inspect_layout(root)
                if result:
                    return result
            return None

        for candidate in candidates:
            result = try_candidate(candidate)
            if result:
                self._save_cache_root(result[0])
                return result

        if allow_prompt and os.name == "nt":
            self.eprint("Automatic detection did not find the EVE game files folder.")
            self.eprint("Opening a folder picker. Select the folder that contains ResFiles and tq.")
            selected = self._prompt_for_cache()
            if selected:
                result = try_candidate(selected)
                if result:
                    self._save_cache_root(result[0])
                    return result
                self.eprint(f"Selected folder was not an EVE game-files root: {selected}")

        self.eprint("ERROR: Automatic EVE installation discovery did not locate the SharedCache indexes.")
        self.eprint("The tool searched Windows Installed Apps registry entries, App Paths, launcher configuration/logs,")
        self.eprint("previous NSAMDR manifests, saved verified locations, and common EVE/Steam install folders.")
        self.eprint("Expected an installation folder containing ResFiles and tq\\resfileindex.txt.")
        if raw:
            self.eprint(f"Requested path: {raw}")
        if checked:
            self.eprint("Last locations checked:")
            for item in checked[-12:]:
                self.eprint(f"  {item}")
        raise SystemExit(10)

    # Purpose: Implement read rows for EveAssetValidationApplication.
    # Called by: command_list, prepare_asset
    # Calls: No same-class helper methods.
    def read_rows(self, indexes: list[Path]) -> list[ResourceRow]:
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

    # Purpose: Implement select model for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: No same-class helper methods.
    def select_model(self, rows: list[ResourceRow], query: str) -> ResourceRow:
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

    # Purpose: Implement related textures for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: No same-class helper methods.
    def related_textures(self, rows: list[ResourceRow], model: ResourceRow) -> dict[str, ResourceRow]:
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

    # Purpose: Implement environment code for model for EveAssetValidationApplication.
    # Called by: select_environment_source, select_environment_sources
    # Calls: No same-class helper methods.
    def _environment_code_for_model(self, model: ResourceRow) -> str:
        logical = model.logical.lower()
        for race, code in ENVIRONMENT_SCENES.items():
            if f"/ship/{race}/" in logical:
                return code
        return "c02"

    # Purpose: Implement resource references from bytes for EveAssetValidationApplication.
    # Called by: select_environment_source
    # Calls: No same-class helper methods.
    def _resource_references_from_bytes(self, data: bytes) -> list[str]:
        text = data.decode("utf-8", errors="ignore")
        references = re.findall(
            r"res:/[^\s\"'<>]+?\.(?:dds|png|tga|jpg|jpeg)",
            text,
            flags=re.IGNORECASE,
        )
        return [value.rstrip("),]}>").replace("\\", "/") for value in references]

    # Purpose: Implement select environment source for EveAssetValidationApplication.
    # Called by: select_environment_sources
    # Calls: _environment_code_for_model, _resource_references_from_bytes
    def select_environment_source(
        self,
        rows: list[ResourceRow],
        model: ResourceRow,
        resfiles: Path,
    ) -> tuple[ResourceRow | None, ResourceRow | None]:
        by_logical = {row.logical.lower(): row for row in rows}
        code = self._environment_code_for_model(model)
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
                for reference in self._resource_references_from_bytes(data):
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

    # Purpose: Implement select environment sources for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: _environment_code_for_model, select_environment_source
    def select_environment_sources(
        self,
        rows: list[ResourceRow],
        model: ResourceRow,
        resfiles: Path,
    ) -> tuple[ResourceRow | None, list[ResourceRow]]:
        """Return every locally available universe cubemap, with the ship-race map first."""
        scene, primary = self.select_environment_source(rows, model, resfiles)
        code = self._environment_code_for_model(model)
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

    # Purpose: Implement safe text for EveAssetValidationApplication.
    # Called by: _build_sde_ship_catalog
    # Calls: No same-class helper methods.
    def _safe_text(self, value: object, fallback: str = "") -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("en", "en-us", "en_US"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return fallback

    # Purpose: Implement sde cache directory for EveAssetValidationApplication.
    # Called by: _ensure_sde_archive
    # Calls: No same-class helper methods.
    def _sde_cache_directory(self, repo_root: Path) -> Path:
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "NSAMDR" / "sde"
        return repo_root / "artifacts" / "nsamdr" / "sde"

    # Purpose: Implement download with progress for EveAssetValidationApplication.
    # Called by: _ensure_sde_archive
    # Calls: No same-class helper methods.
    def _download_with_progress(self, url: str, destination: Path) -> None:
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

    # Purpose: Implement ensure sde archive for EveAssetValidationApplication.
    # Called by: _build_sde_ship_catalog, _resolve_sof_identity
    # Calls: _download_with_progress, _sde_cache_directory
    def _ensure_sde_archive(self, repo_root: Path) -> Path:
        cache_dir = self._sde_cache_directory(repo_root)
        archive = cache_dir / "eve-online-static-data-latest-jsonl.zip"
        has_cached_archive = archive.is_file() and archive.stat().st_size > 1024 * 1024
        if has_cached_archive:
            age = time.time() - archive.stat().st_mtime
            if age <= SDE_CACHE_MAX_AGE_SECONDS:
                return archive
        print("Fetching the official EVE Static Data Export for ship names (cached for seven days)...", flush=True)
        try:
            self._download_with_progress(SDE_LATEST_URL, archive)
        except (OSError, urllib.error.URLError) as exc:
            if has_cached_archive:
                print(f"WARNING: Could not refresh the SDE; using the existing cached copy: {exc}", flush=True)
                return archive
            raise
        return archive

    # Purpose: Implement find zip member for EveAssetValidationApplication.
    # Called by: _read_jsonl_member
    # Calls: No same-class helper methods.
    def _find_zip_member(self, archive: zipfile.ZipFile, basename: str) -> str:
        wanted = basename.lower()
        matches = [name for name in archive.namelist() if Path(name).name.lower() == wanted]
        if not matches:
            raise RuntimeError(f"Official SDE archive did not contain {basename}")
        return min(matches, key=len)

    # Purpose: Implement read jsonl member for EveAssetValidationApplication.
    # Called by: _build_sde_ship_catalog, _resolve_sof_identity
    # Calls: _find_zip_member
    def _read_jsonl_member(self, archive: zipfile.ZipFile, basename: str) -> Iterable[dict]:
        member = self._find_zip_member(archive, basename)
        with archive.open(member, "r") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                value = json.loads(raw.decode("utf-8"))
                if isinstance(value, dict):
                    yield value

    # Purpose: Implement normalise resource path for EveAssetValidationApplication.
    # Called by: _candidate_assets_for_graphic
    # Calls: No same-class helper methods.
    def _normalise_resource_path(self, value: str) -> str:
        return value.strip().replace("\\", "/").lower()

    # Purpose: Implement model family key for EveAssetValidationApplication.
    # Called by: _build_fallback_ship_catalog, _build_sde_ship_catalog, _candidate_assets_for_graphic, _expand_asset_variants, _infer_fallback_display, _resolve_sof_identity
    # Calls: No same-class helper methods.
    def _model_family_key(self, logical: str) -> str:
        stem = Path(logical.rsplit("/", 1)[-1]).stem.lower()
        stem = re.sub(r"(?:[_-](?:lod|l|level|detail)[_-]?\d+)$", "", stem)
        stem = re.sub(r"(?:[_-](?:low|medium|med|high|proxy))$", "", stem)
        return stem

    # Purpose: Implement asset quality score for EveAssetValidationApplication.
    # Called by: _build_fallback_ship_catalog, _build_sde_ship_catalog, _expand_asset_variants
    # Calls: No same-class helper methods.
    def _asset_quality_score(self, logical: str, exact_stem: str = "") -> tuple[int, int, int, int, str]:
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

    # Purpose: Implement candidate assets for graphic for EveAssetValidationApplication.
    # Called by: _build_sde_ship_catalog, _resolve_sof_identity
    # Calls: _model_family_key, _normalise_resource_path
    def _candidate_assets_for_graphic(
        self,
        ship_rows: list[ResourceRow],
        by_logical: dict[str, ResourceRow],
        graphic: dict,
    ) -> list[ResourceRow]:
        graphic_file = self._normalise_resource_path(str(graphic.get("graphicFile") or ""))
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
                    or self._model_family_key(row.logical) == self._model_family_key(graphic_file)
                )
            )
        if sof_hull:
            hull_family = self._model_family_key(sof_hull)
            candidates.extend(
                row for row in ship_rows
                if self._model_family_key(row.logical) == hull_family
                or f"/{hull_family}/" in row.logical.lower()
            )
        unique = {row.logical.lower(): row for row in candidates}
        return list(unique.values())

    # Purpose: Implement expand asset variants for EveAssetValidationApplication.
    # Called by: _build_sde_ship_catalog
    # Calls: _asset_quality_score, _model_family_key
    def _expand_asset_variants(self, ship_rows: list[ResourceRow], preferred: ResourceRow) -> tuple[str, ...]:
        family = self._model_family_key(preferred.logical)
        directory = preferred.logical.lower().rsplit("/", 1)[0] + "/"
        variants = sorted({
            row.logical.replace("\\", "/")
            for row in ship_rows
            if row.logical.lower().startswith(directory)
            and self._model_family_key(row.logical) == family
        }, key=lambda value: self._asset_quality_score(value, Path(preferred.logical).stem.lower()))
        if preferred.logical not in variants:
            variants.insert(0, preferred.logical)
        return tuple(variants)

    # Purpose: Implement infer fallback display for EveAssetValidationApplication.
    # Called by: _build_fallback_ship_catalog, _build_sde_ship_catalog
    # Calls: _model_family_key
    def _infer_fallback_display(self, logical: str) -> tuple[str, str, str]:
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
        code = self._model_family_key(logical)
        return f"Unmapped {race} {ship_class} ({code})", ship_class, race

    # Purpose: Implement build sde ship catalog for EveAssetValidationApplication.
    # Called by: write_ship_catalog
    # Calls: _asset_quality_score, _candidate_assets_for_graphic, _ensure_sde_archive, _expand_asset_variants, _infer_fallback_display, _model_family_key, _read_jsonl_member, _safe_text
    def _build_sde_ship_catalog(self, rows: list[ResourceRow], repo_root: Path) -> list[ShipCatalogEntry]:
        ship_rows = [
            row for row in rows
            if row.logical.lower().endswith(".gr2") and "/model/ship/" in row.logical.lower()
        ]
        by_logical = {row.logical.lower(): row for row in ship_rows}
        archive_path = self._ensure_sde_archive(repo_root)
        with zipfile.ZipFile(archive_path, "r") as archive:
            groups: dict[int, tuple[int, str]] = {}
            for value in self._read_jsonl_member(archive, "groups.jsonl"):
                group_id = value.get("_key")
                if isinstance(group_id, int):
                    groups[group_id] = (
                        int(value.get("categoryID") or -1),
                        self._safe_text(value.get("name"), f"Group {group_id}"),
                    )

            graphics: dict[int, dict] = {}
            for value in self._read_jsonl_member(archive, "graphics.jsonl"):
                graphic_id = value.get("_key")
                if isinstance(graphic_id, int):
                    graphics[graphic_id] = value

            entries: list[ShipCatalogEntry] = []
            claimed_assets: set[str] = set()
            for value in self._read_jsonl_member(archive, "types.jsonl"):
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
                candidates = self._candidate_assets_for_graphic(ship_rows, by_logical, graphic)
                if not candidates:
                    continue
                sof_hull = str(graphic.get("sofHullName") or "").strip().lower()
                graphic_stem = self._model_family_key(sof_hull) if sof_hull else Path(str(graphic.get("graphicFile") or "")).stem.lower()
                preferred = min(candidates, key=lambda row: self._asset_quality_score(row.logical, graphic_stem))
                variants = self._expand_asset_variants(ship_rows, preferred)
                claimed_assets.update(asset.lower() for asset in variants)
                display_name = self._safe_text(value.get("name"), f"Type {type_id}")
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
            fallback_groups.setdefault((directory, self._model_family_key(row.logical)), []).append(row)
        for (_, family), family_rows in fallback_groups.items():
            preferred = min(family_rows, key=lambda row: self._asset_quality_score(row.logical))
            display_name, group_name, faction_name = self._infer_fallback_display(preferred.logical)
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

    # Purpose: Implement build fallback ship catalog for EveAssetValidationApplication.
    # Called by: write_ship_catalog
    # Calls: _asset_quality_score, _infer_fallback_display, _model_family_key
    def _build_fallback_ship_catalog(self, rows: list[ResourceRow]) -> list[ShipCatalogEntry]:
        ship_rows = [
            row for row in rows
            if row.logical.lower().endswith(".gr2") and "/model/ship/" in row.logical.lower()
        ]
        grouped: dict[tuple[str, str], list[ResourceRow]] = {}
        for row in ship_rows:
            directory = row.logical.lower().rsplit("/", 1)[0]
            grouped.setdefault((directory, self._model_family_key(row.logical)), []).append(row)
        entries: list[ShipCatalogEntry] = []
        for (_, family), family_rows in grouped.items():
            preferred = min(family_rows, key=lambda row: self._asset_quality_score(row.logical))
            display_name, group_name, faction_name = self._infer_fallback_display(preferred.logical)
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

    # Purpose: Implement tsv clean for EveAssetValidationApplication.
    # Called by: _write_tint_only_material_manifest, write_ship_catalog
    # Calls: No same-class helper methods.
    def _tsv_clean(self, value: str) -> str:
        return value.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()

    # Purpose: Implement write ship catalog for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: _build_fallback_ship_catalog, _build_sde_ship_catalog, _tsv_clean
    def write_ship_catalog(self, rows: list[ResourceRow], output_path: Path, repo_root: Path) -> Path:
        source = "official SDE"
        try:
            entries = self._build_sde_ship_catalog(rows, repo_root)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            source = f"cache-path fallback ({exc})"
            print(f"WARNING: Real-name SDE mapping unavailable: {exc}", flush=True)
            entries = self._build_fallback_ship_catalog(rows)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# NSAMDR_SHIP_CATALOG_V2\tdisplay_name\tgroup\tfaction\ttype_id\tcanonical_key\tpreferred_asset\tvariants"]
        for entry in entries:
            lines.append("\t".join([
                self._tsv_clean(entry.display_name),
                self._tsv_clean(entry.group_name),
                self._tsv_clean(entry.faction_name),
                str(entry.type_id or 0),
                self._tsv_clean(entry.canonical_key),
                self._tsv_clean(entry.preferred_asset),
                "|".join(self._tsv_clean(value) for value in entry.variants),
            ]))
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Ship catalog: {output_path} ({len(entries)} grouped ships; names from {source})", flush=True)
        return output_path

    # Purpose: Implement source path for EveAssetValidationApplication.
    # Called by: _resolve_sof_texture, copy_resource
    # Calls: No same-class helper methods.
    def source_path(self, resfiles: Path, row: ResourceRow) -> Path:
        return resfiles / Path(row.hashed)

    # Purpose: Implement same size file for EveAssetValidationApplication.
    # Called by: copy_resource
    # Calls: No same-class helper methods.
    def _same_size_file(self, source: Path, destination: Path) -> bool:
        """Return True when an already-copied cache resource can be reused safely.

        EVE background assets are immutable content-addressed files. Re-opening hundreds
        of existing DDS destinations on every Raven dataset probe is unnecessary and,
        on Windows, can also fail if another process momentarily has one of those files
        open. Size equality is sufficient here because the source path itself is the
        indexed content-addressed cache object.
        """
        try:
            return destination.is_file() and destination.stat().st_size == source.stat().st_size
        except OSError:
            return False

    # Purpose: Implement copy resource for EveAssetValidationApplication.
    # Called by: _copy_optional_resource, prepare_asset
    # Calls: _same_size_file, source_path
    def copy_resource(self, resfiles: Path, row: ResourceRow, output_dir: Path) -> Path:
        source = self.source_path(resfiles, row)
        if not source.is_file():
            raise RuntimeError(
                f"Indexed EVE resource is not present locally: {row.logical}\n"
                f"Expected cache file: {source}\n"
                "In the EVE Launcher, verify the Shared Cache and enable full resource download, then retry."
            )
        logical_name = row.logical.rsplit("/", 1)[-1]
        destination = output_dir / logical_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if self._same_size_file(source, destination):
            return destination
        shutil.copy2(source, destination)
        return destination

    # Purpose: Implement copy optional resource for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: copy_resource
    def _copy_optional_resource(
        self,
        resfiles: Path,
        row: ResourceRow,
        output_dir: Path,
        *,
        kind: str,
    ) -> Path | None:
        """Best-effort copy for non-authoritative preview environment resources."""
        try:
            return self.copy_resource(resfiles, row, output_dir)
        except (OSError, RuntimeError) as exc:
            print(f"WARNING: Optional EVE {kind} copy skipped for {row.logical}: {exc}", flush=True)
            return None

    # Purpose: Implement find command for EveAssetValidationApplication.
    # Called by: ensure_converter
    # Calls: No same-class helper methods.
    def find_command(self, names: list[str]) -> str:
        for name in names:
            found = shutil.which(name)
            if found:
                return found
        raise RuntimeError(f"Required command not found: {' or '.join(names)}")

    # Purpose: Implement probe converter modules for EveAssetValidationApplication.
    # Called by: ensure_converter
    # Calls: No same-class helper methods.
    def probe_converter_modules(self, node: str, converter_dir: Path, *, show_failure: bool = False) -> bool:
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
                print("CarbonEngineJS module-resolution probe failed:", flush=True)
                print("\n".join(diagnostic.splitlines()[-20:]), flush=True)
        return result.returncode == 0

    # Purpose: Implement ensure converter for EveAssetValidationApplication.
    # Called by: convert_dds, convert_environment_dds, convert_gr2, convert_sof
    # Calls: find_command, probe_converter_modules
    def ensure_converter(self, converter_dir: Path) -> tuple[str, Path]:
        node = self.find_command(["node.exe", "node"])
        script = converter_dir / "convert_eve_asset.mjs"
        package = converter_dir / "package.json"
        if not script.is_file() or not package.is_file():
            raise RuntimeError(f"Missing NSAMDR converter source under {converter_dir}")

        # npm is free to hoist or nest transitive packages. Directory-layout checks
        # therefore produce false failures even when Node can resolve the two public
        # entry points used by the converter. Test the imports themselves instead.
        if not self.probe_converter_modules(node, converter_dir):
            npm = self.find_command(["npm.cmd", "npm.exe", "npm"])
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
                    "npm could not install the CarbonEngineJS readers from their public GitHub source archives. "
                    "Check Node.js 18+, HTTPS access to github.com, and npm connectivity, then rerun."
                )

        if not self.probe_converter_modules(node, converter_dir, show_failure=True):
            raise RuntimeError(
                "CarbonEngineJS installation finished, but Node could not import the converter's actual entry points. "
                "The module-resolution error printed above identifies the unresolved package."
            )
        return node, script

    # Purpose: Implement run checked for EveAssetValidationApplication.
    # Called by: convert_dds, convert_environment_dds, convert_gr2, convert_sof
    # Calls: No same-class helper methods.
    def run_checked(self, command: list[str], cwd: Path | None = None) -> None:
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

    # Purpose: Implement convert gr2 for EveAssetValidationApplication.
    # Called by: command_convert_gr2, prepare_asset
    # Calls: ensure_converter, run_checked
    def convert_gr2(self, repo_root: Path, input_path: Path, output_path: Path) -> Path:
        converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
        node, script = self.ensure_converter(converter_dir)
        summary = output_path.with_suffix(".conversion.json")
        self.run_checked([node, str(script), "gr2-to-obj", str(input_path), str(output_path), str(summary)], converter_dir)
        if not output_path.is_file():
            raise RuntimeError(f"GR2 converter did not create {output_path}")
        if not summary.is_file():
            raise RuntimeError(f"GR2 converter did not create coordinate-contract summary {summary}")
        try:
            conversion_record = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GR2 coordinate-contract summary is unreadable: {summary}: {exc}") from exc
        coordinate_basis = conversion_record.get("coordinateBasis")
        if conversion_record.get("schema") != EXPECTED_GR2_CONVERSION_SCHEMA:
            raise RuntimeError(
                f"Stale GR2 conversion schema in {summary}: "
                f"{conversion_record.get('schema')!r}; expected {EXPECTED_GR2_CONVERSION_SCHEMA!r}")
        if not isinstance(coordinate_basis, dict) or (
            coordinate_basis.get("mirrorAxis") != "x"
            or coordinate_basis.get("triangleWindingTransform") != "a,b,c -> a,c,b"
            or coordinate_basis.get("uvTransform") != "u_out = u_gr2; v_out = v_gr2"
        ):
            raise RuntimeError(f"Invalid GR2 handedness contract in {summary}: {coordinate_basis!r}")
        print(
            "GR2 coordinate contract: "
            "EVE_GR2_LOCAL -> NSAMDR_DIRECTX_LEFT_HANDED, mirror X, reverse winding, preserve UV",
            flush=True,
        )
        return output_path

    # Purpose: Implement convert dds for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials, prepare_asset
    # Calls: ensure_converter, run_checked
    def convert_dds(self, repo_root: Path, input_path: Path, output_path: Path) -> Path:
        if output_path.is_file() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
            return output_path
        converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
        node, script = self.ensure_converter(converter_dir)
        self.run_checked([node, str(script), "dds-to-png", str(input_path), str(output_path)], converter_dir)
        if not output_path.is_file():
            raise RuntimeError(f"DDS converter did not create {output_path}")
        return output_path

    # Purpose: Implement convert environment dds for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: ensure_converter, run_checked
    def convert_environment_dds(self, repo_root: Path, input_path: Path, output_path: Path) -> Path:
        if output_path.is_file() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
            return output_path
        converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
        node, script = self.ensure_converter(converter_dir)
        self.run_checked([node, str(script), "dds-to-environment-png", str(input_path), str(output_path)], converter_dir)
        if not output_path.is_file():
            raise RuntimeError(f"Environment DDS converter did not create {output_path}")
        return output_path

    # Purpose: Implement race from ship model for EveAssetValidationApplication.
    # Called by: _resolve_sof_identity
    # Calls: No same-class helper methods.
    def _race_from_ship_model(self, logical_path: str) -> str:
        """Derive the SOF race from the selected hull's canonical model path.

        EVE ship geometry is stored beneath ``model/ship/<race>/...``. This is the
        hull race and remains valid for faction variants which reuse that hull.
        """
        normalized = logical_path.strip().replace("\\", "/").lower()
        match = re.search(r"(?:^|/)model/ship/([^/]+)/", normalized)
        return match.group(1).strip() if match else ""

    # Purpose: Implement resolve sof identity for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: _candidate_assets_for_graphic, _ensure_sde_archive, _model_family_key, _race_from_ship_model, _read_jsonl_member
    def _resolve_sof_identity(
        self,
        rows: list[ResourceRow],
        repo_root: Path,
        model: ResourceRow,
        selection_key: str,
    ) -> dict[str, str | int | None]:
        archive_path = self._ensure_sde_archive(repo_root)
        wanted_type: int | None = None
        match = re.fullmatch(r"type:(\d+)", (selection_key or "").strip().lower())
        if match:
            wanted_type = int(match.group(1))

        ship_rows = [row for row in rows if row.logical.lower().endswith(".gr2") and "/model/ship/" in row.logical.lower()]
        by_logical = {row.logical.lower(): row for row in ship_rows}
        selected_graphic: dict | None = None
        selected_type: int | None = None
        with zipfile.ZipFile(archive_path, "r") as archive:
            graphics = {
                value.get("_key"): value
                for value in self._read_jsonl_member(archive, "graphics.jsonl")
                if isinstance(value.get("_key"), int)
            }
            if wanted_type is not None:
                for value in self._read_jsonl_member(archive, "types.jsonl"):
                    if value.get("_key") == wanted_type:
                        selected_type = wanted_type
                        selected_graphic = graphics.get(value.get("graphicID"))
                        break
            else:
                model_key = model.logical.lower()
                for graphic in graphics.values():
                    candidates = self._candidate_assets_for_graphic(ship_rows, by_logical, graphic)
                    if any(candidate.logical.lower() == model_key for candidate in candidates):
                        selected_graphic = graphic
                        break

        model_race = self._race_from_ship_model(model.logical)
        if selected_graphic is None:
            hull = self._model_family_key(model.logical)
            return {
                "typeID": selected_type,
                "hull": hull,
                "faction": f"{model_race}base" if model_race else "",
                "race": model_race,
                "raceSource": "modelPath" if model_race else "unresolved",
            }

        # ``sofRaceName`` is absent on a number of valid SDE graphic records. Race
        # is not optional for rendering: derive it from the selected hull path.
        # The explicit SDE field wins when present, but no user input is required.
        explicit_race = str(selected_graphic.get("sofRaceName") or "").strip()
        race = explicit_race or model_race
        faction = str(selected_graphic.get("sofFactionName") or (f"{race}base" if race else "")).strip()
        return {
            "typeID": selected_type,
            "hull": str(selected_graphic.get("sofHullName") or self._model_family_key(model.logical)).strip(),
            "faction": faction,
            "race": race,
            "raceSource": "sde" if explicit_race else ("modelPath" if model_race else "unresolved"),
        }

    # Purpose: Implement convert sof for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: ensure_converter, run_checked
    def convert_sof(
        self,
        repo_root: Path,
        data_black: Path,
        output_path: Path,
        hull: str,
        faction: str,
        race: str,
    ) -> Path:
        converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
        node, script = self.ensure_converter(converter_dir)
        self.run_checked([node, str(script), "sof-to-json", str(data_black), str(output_path), hull, faction, race], converter_dir)
        if not output_path.is_file():
            raise RuntimeError(f"SOF converter did not create {output_path}")
        return output_path

    # Purpose: Implement resource row for EveAssetValidationApplication.
    # Called by: _resolve_sof_texture
    # Calls: No same-class helper methods.
    def _resource_row(self, rows_by_logical: dict[str, ResourceRow], logical: str) -> ResourceRow | None:
        return rows_by_logical.get(logical.strip().replace("\\", "/").lower())

    # Purpose: Implement faction texture path for EveAssetValidationApplication.
    # Called by: _resolve_sof_texture
    # Calls: No same-class helper methods.
    def _faction_texture_path(self, logical: str, insert: str) -> str:
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

    # Purpose: Implement resolve sof texture for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials
    # Calls: _faction_texture_path, _resource_row, source_path
    def _resolve_sof_texture(
        self,
        rows_by_logical: dict[str, ResourceRow],
        resfiles: Path,
        logical: str,
        res_path_insert: str,
    ) -> ResourceRow | None:
        modified = self._resource_row(rows_by_logical, self._faction_texture_path(logical, res_path_insert))
        original = self._resource_row(rows_by_logical, logical)
        for candidate in (modified, original):
            if candidate is not None and self.source_path(resfiles, candidate).is_file():
                return candidate
        return modified or original

    # Purpose: Implement numeric vector for EveAssetValidationApplication.
    # Called by: _faction_color, _numeric_scalar, _numeric_vector, _parameter_vector, _resolved_area_material_slots
    # Calls: _numeric_vector
    def _numeric_vector(self, value: object) -> tuple[float, float, float, float] | None:
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
                    return self._numeric_vector(ordered)
        return None

    # Purpose: Implement normal key for EveAssetValidationApplication.
    # Called by: _classify_shader_family, _exact_parameter, _find_named_value, _material_for_area, _texture_usage
    # Calls: No same-class helper methods.
    def _normal_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    # Purpose: Implement find named value for EveAssetValidationApplication.
    # Called by: _area_base_visuals, _faction_color, _find_named_value, _material_names_for_area, _parameter_scalar, _parameter_vector
    # Calls: _find_named_value, _normal_key
    def _find_named_value(self, value: object, wanted_names: Iterable[str]) -> object | None:
        wanted = {self._normal_key(name) for name in wanted_names}
        if isinstance(value, dict):
            for key, child in value.items():
                if self._normal_key(str(key)) in wanted:
                    return child
            for child in value.values():
                result = self._find_named_value(child, wanted)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = self._find_named_value(child, wanted)
                if result is not None:
                    return result
        return None

    # Purpose: Implement faction color for EveAssetValidationApplication.
    # Called by: _area_base_visuals, _slot_fallback_colors
    # Calls: _find_named_value, _numeric_vector
    def _faction_color(self, sof: dict, names: Iterable[str]) -> tuple[float, float, float, float] | None:
        value = self._find_named_value(sof.get("colors", {}), names)
        vector = self._numeric_vector(value)
        if vector is None:
            return None
        return tuple(max(0.0, min(4.0, component)) for component in vector)  # type: ignore[return-value]

    # Purpose: Implement material for area for EveAssetValidationApplication.
    # Called by: _area_base_visuals, _material_names_for_area
    # Calls: _normal_key
    def _material_for_area(self, sof: dict, area_type: str) -> dict:
        materials = sof.get("areaMaterials", {})
        if not isinstance(materials, dict):
            return {}
        for key, value in materials.items():
            if self._normal_key(str(key)) == self._normal_key(area_type) and isinstance(value, dict):
                return value
        return {}

    # Purpose: Implement clamp for EveAssetValidationApplication.
    # Called by: _gloss_to_roughness, _material_slot_visuals, _resolved_area_material_slots, _slot_fallback_colors
    # Calls: No same-class helper methods.
    def _clamp(self, value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    # Purpose: Implement numeric scalar for EveAssetValidationApplication.
    # Called by: _parameter_scalar, _resolved_area_material_slots
    # Calls: _numeric_vector
    def _numeric_scalar(self, value: object) -> float | None:
        if isinstance(value, (int, float)):
            number = float(value)
            return number if number == number else None
        vector = self._numeric_vector(value)
        return vector[0] if vector is not None else None

    # Purpose: Implement area base visuals for EveAssetValidationApplication.
    # Called by: _area_material_slots, _area_visual_parameters
    # Calls: _faction_color, _find_named_value, _material_for_area
    def _area_base_visuals(
        self,
        sof: dict,
        area_type: str,
        race_name: str,
    ) -> tuple[tuple[float, float, float], float, float, float, tuple[float, float, float]]:
        material = self._material_for_area(sof, area_type)
        if not material and area_type != "primary":
            material = self._material_for_area(sof, "primary")
        color_type = self._find_named_value(material, ("colorType", "glowColorType"))
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
        tint = self._faction_color(sof, area_color_names)

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
        glow = self._faction_color(sof, glow_names)
        if glow is not None:
            glow_rgb = glow[:3]
        elif area_type == "reactor":
            glow_rgb = (1.0, 0.26, 0.04)
        elif area_type == "glass":
            glow_rgb = (0.12, 0.48, 0.85)
        else:
            glow_rgb = (0.34, 0.58, 0.95)
        return tint_rgb, 1.0, default_roughness, default_specular, glow_rgb

    # Purpose: Implement material names for area for EveAssetValidationApplication.
    # Called by: _area_material_slots
    # Calls: _find_named_value, _material_for_area
    def _material_names_for_area(self, sof: dict, area_type: str) -> list[str]:
        material = self._material_for_area(sof, area_type)
        if not material and area_type != "primary":
            material = self._material_for_area(sof, "primary")
        names: list[str] = []
        for index in range(1, 5):
            value = self._find_named_value(material, (f"material{index}",))
            names.append(value.strip() if isinstance(value, str) else "")
        return names

    # Purpose: Implement lookup material parameters for EveAssetValidationApplication.
    # Called by: _area_material_slots
    # Calls: No same-class helper methods.
    def _lookup_material_parameters(self, sof: dict, material_name: str) -> dict:
        library = sof.get("materialLibrary", {})
        if not material_name or not isinstance(library, dict):
            return {}
        wanted = material_name.lower()
        for key, value in library.items():
            if str(key).lower() == wanted and isinstance(value, dict):
                return value
        return {}

    # Purpose: Implement slot fallback colors for EveAssetValidationApplication.
    # Called by: _area_material_slots
    # Calls: _clamp, _faction_color
    def _slot_fallback_colors(
        self,
        sof: dict,
        area_type: str,
        race_name: str,
        area_tint: tuple[float, float, float],
    ) -> list[tuple[float, float, float]]:
        def color(names: tuple[str, ...], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
            value = self._faction_color(sof, names)
            return value[:3] if value is not None else fallback

        primary = color(("primary", "hull"), area_tint)
        secondary = color(("secondary", "white"), tuple(self._clamp(component * 1.18, 0.0, 2.0) for component in area_tint))
        tertiary = color(("tertiary", "darkhull"), tuple(self._clamp(component * 0.62, 0.0, 2.0) for component in area_tint))
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

    # Purpose: Implement parameter vector for EveAssetValidationApplication.
    # Called by: _material_slot_visuals
    # Calls: _find_named_value, _numeric_vector
    def _parameter_vector(self, parameters: dict, names: Iterable[str]) -> tuple[float, float, float, float] | None:
        return self._numeric_vector(self._find_named_value(parameters, names))

    # Purpose: Implement parameter scalar for EveAssetValidationApplication.
    # Called by: _material_slot_visuals
    # Calls: _find_named_value, _numeric_scalar
    def _parameter_scalar(self, parameters: dict, names: Iterable[str]) -> float | None:
        return self._numeric_scalar(self._find_named_value(parameters, names))

    # Purpose: Implement material slot visuals for EveAssetValidationApplication.
    # Called by: _area_material_slots
    # Calls: _clamp, _parameter_scalar, _parameter_vector
    def _material_slot_visuals(
        self,
        material_name: str,
        parameters: dict,
        fallback_color: tuple[float, float, float],
        area_type: str,
        default_roughness: float,
    ) -> dict[str, object]:
        base_vector = self._parameter_vector(parameters, (
            "DiffuseColor", "AlbedoColor", "BaseColor", "MaterialColor", "GeneralColor", "Color",
        ))
        if base_vector is not None and max(base_vector[:3]) > 1.0e-5:
            base_color = tuple(self._clamp(component, 0.0, 4.0) for component in base_vector[:3])
        else:
            base_color = fallback_color

        roughness = default_roughness
        rough_value = self._parameter_scalar(parameters, (
            "Roughness", "RoughnessFactors", "MaterialRoughness", "DiffuseRoughness",
        ))
        if rough_value is not None:
            roughness = self._clamp(rough_value, 0.035, 0.98)
        else:
            gloss = self._parameter_scalar(parameters, ("Gloss", "Glossiness", "GlossFactors", "SpecularPower"))
            if gloss is not None:
                roughness = self._clamp(1.0 - gloss, 0.035, 0.98) if gloss <= 1.0 else self._clamp((2.0 / (gloss + 2.0)) ** 0.5, 0.035, 0.98)

        lower_name = material_name.lower()
        metallic_hint = any(token in lower_name for token in ("metal", "steel", "chrome", "silver", "gold", "copper", "brass"))
        metalness = self._parameter_scalar(parameters, ("Metallic", "Metalness", "MetallicFactor"))
        if metalness is None:
            metalness = 0.88 if metallic_hint else (0.35 if area_type in ("ornament", "reactor") else 0.0)
        metalness = self._clamp(metalness)

        f0_vector = self._parameter_vector(parameters, ("FresnelColor", "SpecularColor", "ReflectanceColor"))
        if f0_vector is not None:
            f0 = tuple(self._clamp(component, 0.018, 1.0) for component in f0_vector[:3])
        else:
            f0_scalar = self._parameter_scalar(parameters, ("Reflectance", "Specular", "SpecularFactor", "FresnelFactors"))
            dielectric = self._clamp(0.04 if f0_scalar is None else f0_scalar, 0.018, 0.24)
            f0 = tuple((1.0 - metalness) * dielectric + metalness * self._clamp(component, 0.02, 1.0) for component in base_color)

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
            "gloss": self._clamp(1.0 - roughness, 0.0, 1.0),
        }

    # Purpose: Implement area visual parameters for EveAssetValidationApplication.
    # Called by: External callers and the owning workflow.
    # Calls: _area_base_visuals
    def _area_visual_parameters(self, sof: dict, area_type: str, race_name: str) -> tuple[tuple[float, float, float], float, float, float]:
        """Legacy V1 fallback parameters retained for old manifests."""
        tint, detail_scale, roughness, specular, _ = self._area_base_visuals(sof, area_type, race_name)
        return tint, detail_scale, roughness, specular

    # Purpose: Implement area material slots for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials, _resolved_area_material_slots, _write_tint_only_material_manifest
    # Calls: _area_base_visuals, _lookup_material_parameters, _material_names_for_area, _material_slot_visuals, _slot_fallback_colors
    def _area_material_slots(self, sof: dict, area_type: str, race_name: str) -> tuple[list[dict[str, object]], tuple[float, float, float], tuple[float, float, float], float]:
        tint, detail_scale, default_roughness, _, glow = self._area_base_visuals(sof, area_type, race_name)
        names = self._material_names_for_area(sof, area_type)
        fallbacks = self._slot_fallback_colors(sof, area_type, race_name, tint)
        slots = [
            self._material_slot_visuals(name, self._lookup_material_parameters(sof, name), fallbacks[index], area_type, default_roughness)
            for index, name in enumerate(names)
        ]
        while len(slots) < 4:
            index = len(slots)
            slots.append(self._material_slot_visuals("", {}, fallbacks[index], area_type, default_roughness))
        return slots[:4], tint, glow, detail_scale

    # Purpose: Implement exact parameter for EveAssetValidationApplication.
    # Called by: _resolved_area_material_slots
    # Calls: _normal_key
    def _exact_parameter(self, parameters: object, name: str) -> object | None:
        if not isinstance(parameters, dict):
            return None
        wanted = self._normal_key(name)
        for key, value in parameters.items():
            if self._normal_key(str(key)) == wanted:
                return value
        return None

    # Purpose: Implement gloss to roughness for EveAssetValidationApplication.
    # Called by: _resolved_area_material_slots
    # Calls: _clamp
    def _gloss_to_roughness(self, gloss: float, fallback: float) -> float:
        if gloss != gloss:
            return fallback
        if gloss <= 1.0:
            return self._clamp(1.0 - gloss, 0.035, 0.98)
        return self._clamp((2.0 / (gloss + 2.0)) ** 0.5, 0.035, 0.98)

    # Purpose: Implement resolved area material slots for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials
    # Calls: _area_material_slots, _clamp, _exact_parameter, _gloss_to_roughness, _numeric_scalar, _numeric_vector
    def _resolved_area_material_slots(
        self,
        sof: dict,
        area: dict,
        area_type: str,
        race_name: str,
    ) -> tuple[list[dict[str, object]], tuple[float, float, float], tuple[float, float, float], float, float, bool, int]:
        fallback_slots, tint, fallback_glow, detail_scale = self._area_material_slots(sof, area_type, race_name)
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
            color_value = self._numeric_vector(self._exact_parameter(parameters, f"{prefix}DiffuseColor"))
            f0_value = self._numeric_vector(self._exact_parameter(parameters, f"{prefix}FresnelColor"))
            gloss_value = self._numeric_scalar(self._exact_parameter(parameters, f"{prefix}Gloss"))
            if color_value is None:
                unresolved += 1
            if f0_value is None:
                unresolved += 1
            if gloss_value is None:
                unresolved += 1
            slots.append({
                "name": str(names[index]) if index < len(names) else str(fallback.get("name", "")),
                "color": tuple(self._clamp(value, 0.0, 4.0) for value in color_value[:3]) if color_value else fallback["color"],
                "f0": tuple(self._clamp(value, 0.0, 1.0) for value in f0_value[:3]) if f0_value else fallback["f0"],
                "roughness": self._gloss_to_roughness(gloss_value, float(fallback["roughness"])) if gloss_value is not None else fallback["roughness"],
                "gloss": self._clamp(gloss_value, 0.0, 256.0) if gloss_value is not None else float(fallback.get("gloss", 1.0 - float(fallback["roughness"]))),
            })

        glow_value = self._numeric_vector(self._exact_parameter(parameters, "GeneralGlowColor"))
        glow = tuple(self._clamp(value, 0.0, 8.0) for value in glow_value[:3]) if glow_value else fallback_glow
        # The explicit slot checks above already account for unresolved visible
        # parameters. Do not count the converter's unresolved list a second time.
        general_data = self._numeric_vector(self._exact_parameter(parameters, "GeneralData"))
        general_data_x = float(general_data[0]) if general_data is not None else 1.0
        return slots, tint, glow, detail_scale, general_data_x, unresolved == 0, unresolved

    # Purpose: Implement texture usage for EveAssetValidationApplication.
    # Called by: _classify_shader_family, _semantic_texture_layout
    # Calls: _normal_key
    def _texture_usage(self, textures: dict, candidates: Iterable[str]) -> str:
        wanted = {self._normal_key(value) for value in candidates}
        for key, value in textures.items():
            if self._normal_key(str(key)) in wanted and isinstance(value, str):
                return value
        for key, value in textures.items():
            normalized = self._normal_key(str(key))
            if any(candidate in normalized for candidate in wanted) and isinstance(value, str):
                return value
        return ""

    # Purpose: Implement texture suffix for EveAssetValidationApplication.
    # Called by: _classify_shader_family, _texture_by_suffix
    # Calls: No same-class helper methods.
    def _texture_suffix(self, logical: str) -> str:
        stem = Path(str(logical or '').replace('\\', '/')).stem.lower()
        for suffix in ('_pmdg', '_ar', '_no', '_pgs', '_pgr', '_ap', '_d', '_n'):
            if stem.endswith(suffix):
                return suffix
        return ''

    # Purpose: Implement texture by suffix for EveAssetValidationApplication.
    # Called by: _semantic_texture_layout
    # Calls: _texture_suffix
    def _texture_by_suffix(self, textures: dict, suffix: str) -> str:
        for value in textures.values():
            if isinstance(value, str) and self._texture_suffix(value) == suffix:
                return value
        return ''

    # Purpose: Implement classify shader family for EveAssetValidationApplication.
    # Called by: _semantic_texture_layout
    # Calls: _normal_key, _texture_suffix, _texture_usage
    def _classify_shader_family(self, area: dict, textures: dict) -> str:
        shader = str(area.get('shader') or '').replace('\\', '/').lower()
        suffixes = {self._texture_suffix(value) for value in textures.values() if isinstance(value, str)}
        if {'_ar', '_no', '_pmdg'} <= suffixes:
            return 'v5_packed'
        if any(self._normal_key(str(key)) == 'pmdgmap' for key in textures) or '_pmdg' in suffixes:
            return 'v5_packed'
        separate_semantics = ('RoughnessMap', 'PaintMaskMap', 'MaterialMap', 'DirtMap', 'GlowMap')
        if any(self._texture_usage(textures, (name,)) for name in separate_semantics):
            return 'v5_separate'
        if self._texture_usage(textures, ('PgsMap',)) or '_pgs' in suffixes:
            return 'legacy_pgs'
        if '_pgr' in suffixes or '_ap' in suffixes or 'v5' in shader:
            return 'v5_separate'
        return 'unknown'

    # Purpose: Implement semantic texture layout for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials
    # Calls: _classify_shader_family, _texture_by_suffix, _texture_usage
    def _semantic_texture_layout(self, area: dict, textures: dict) -> dict[str, object]:
        family = self._classify_shader_family(area, textures)
        direct = {
            'albedo': self._texture_usage(textures, ('AlbedoMap', 'DiffuseMap', 'DiffuseMap1', 'DetailMap')),
            'normal': self._texture_usage(textures, ('NormalMap', 'NormalMap1')),
            'material': self._texture_usage(textures, ('MaterialMap', 'PgsMap')),
            'glow': self._texture_usage(textures, ('GlowMap', 'EmissiveMap')),
            'dirt': self._texture_usage(textures, ('DirtMap', 'GrimeMap')),
            'ao': self._texture_usage(textures, ('AoMap', 'AmbientOcclusionMap')),
            'paintMask': self._texture_usage(textures, ('PaintMaskMap', 'PaintMask')),
            'roughnessMap': self._texture_usage(textures, ('RoughnessMap',)),
        }
        channels = {
            'normalX': SEMANTIC_CHANNEL_RED, 'normalY': SEMANTIC_CHANNEL_GREEN,
            'roughness': SEMANTIC_CHANNEL_RED, 'material': SEMANTIC_CHANNEL_RED,
            'ao': SEMANTIC_CHANNEL_RED, 'paint': SEMANTIC_CHANNEL_RED,
            'dirt': SEMANTIC_CHANNEL_RED, 'glow': SEMANTIC_CHANNEL_RED,
        }
        required = ['albedo', 'normal', 'material', 'roughnessMap']

        if family == 'v5_packed':
            ar = self._texture_by_suffix(textures, '_ar') or direct['albedo']
            no = self._texture_by_suffix(textures, '_no') or direct['normal']
            pmdg = self._texture_by_suffix(textures, '_pmdg') or direct['material']
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
            # Pre-PBR PGS: R=sub-mask, G=specular, B=mask, A=glow/opacity.
            # The renderer keeps this family explicit rather than pretending it is V5.
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

    # Purpose: Implement material manifest columns for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials, _write_tint_only_material_manifest
    # Calls: No same-class helper methods.
    def _material_manifest_columns(self) -> list[str]:
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

    # Purpose: Implement write material record for EveAssetValidationApplication.
    # Called by: _prepare_sof_materials, _write_tint_only_material_manifest
    # Calls: No same-class helper methods.
    def _write_material_record(self, writer: csv.writer, record: dict[str, object]) -> None:
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

    # Purpose: Implement prepare sof materials for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: _area_material_slots, _material_manifest_columns, _resolve_sof_texture, _resolved_area_material_slots, _semantic_texture_layout, _write_material_record, convert_dds
    def _prepare_sof_materials(
        self,
        repo_root: Path,
        rows: list[ResourceRow],
        resfiles: Path,
        output_dir: Path,
        conversion_summary: Path,
        sof_manifest_path: Path,
        race_name: str,
    ) -> tuple[Path, dict[str, dict]]:
        sof = json.loads(sof_manifest_path.read_text(encoding="utf-8"))
        conversion = json.loads(conversion_summary.read_text(encoding="utf-8"))
        rows_by_logical = {row.logical.lower(): row for row in rows}
        res_path_insert = str(sof.get("resPathInsert") or "")
        materials_dir = output_dir / "materials"
        materials_dir.mkdir(parents=True, exist_ok=True)
        converted_cache: dict[str, Path] = {}
        copied_metadata: dict[str, dict] = {}

        def prepare_texture(logical: str, usage: str) -> Path | None:
            if not logical:
                return None
            row = self._resolve_sof_texture(rows_by_logical, resfiles, logical, res_path_insert)
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
            self.convert_dds(repo_root, copied, output)
            converted_cache[cache_key] = output
            copied_metadata[cache_key] = {"usage": usage, "usages": [usage], **asdict(row), "local": str(copied), "converted": str(output)}
            return output

        draw_ranges = conversion.get("drawRanges", [])
        existing_groups = {int(draw.get("groupIndex", -1)) for draw in draw_ranges}
        areas = sof.get("areas", []) if isinstance(sof.get("areas"), list) else []
        records: list[dict[str, object]] = []
        for area in areas:
            if not isinstance(area, dict):
                continue
            first_group = int(area.get("index") or 0)
            group_count = max(1, int(area.get("count") or 1))
            textures = area.get("textures") if isinstance(area.get("textures"), dict) else {}
            layout = self._semantic_texture_layout(area, textures)
            texture_usages = {key: str(layout.get(key) or "") for key in
                              ("albedo", "normal", "material", "glow", "dirt", "ao", "paintMask", "roughnessMap")}
            prepared = {key: str(prepare_texture(value, key) or "") for key, value in texture_usages.items()}
            required_semantics = list(layout.get("requiredSemantics") or [])
            unresolved_declared = [key for key, value in texture_usages.items() if value and not prepared.get(key)]
            missing_semantics = list(dict.fromkeys(list(layout.get("missingSemantics") or []) + unresolved_declared))
            area_type = str(area.get("areaType") or "primary").replace("TYPE_", "").lower()
            slots, tint, glow_color, detail_scale, general_data_x, parameter_complete, unresolved_count = self._resolved_area_material_slots(
                sof, area, area_type, race_name
            )
            pass_name = str(area.get("pass") or "opaque")
            alpha = 0.42 if pass_name == "transparent" else 1.0
            semantic_complete = bool(layout.get("semanticComplete")) and all(prepared.get(name) for name in required_semantics) and not unresolved_declared
            baseline_complete = parameter_complete and semantic_complete
            for group_index in range(first_group, first_group + group_count):
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
        fallback_slots, fallback_tint, fallback_glow, fallback_scale = self._area_material_slots(sof, "primary", race_name)
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
            writer.writerow(self._material_manifest_columns())
            for record in records:
                self._write_material_record(writer, record)

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

    # Purpose: Implement write tint only material manifest for EveAssetValidationApplication.
    # Called by: prepare_asset
    # Calls: _area_material_slots, _material_manifest_columns, _tsv_clean, _write_material_record
    def _write_tint_only_material_manifest(
        self,
        output_dir: Path,
        conversion_summary: Path,
        race_name: str,
        reason: str,
    ) -> Path:
        """Emit an explicitly incomplete fallback without applying guessed texture assignments."""
        conversion = json.loads(conversion_summary.read_text(encoding="utf-8"))
        draw_ranges = conversion.get("drawRanges", [])
        if not isinstance(draw_ranges, list) or not draw_ranges:
            raise RuntimeError("Cannot build tint-only fallback: conversion summary has no draw ranges")

        slots, tint, glow, detail_scale = self._area_material_slots({}, "primary", race_name)
        manifest_path = output_dir / "ship.materials.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# NSAMDR_MATERIALS_V3\n")
            handle.write(f"# incomplete tint-only fallback: {self._tsv_clean(reason)}\n")
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(self._material_manifest_columns())
            for draw in draw_ranges:
                self._write_material_record(writer, {
                    "group": int(draw.get("groupIndex", 0)), "pass": "opaque", "areaType": "primary", "areaName": "tint-only", "shader": "",
                    "shaderFamily": "unknown", "channels": {},
                    "albedo": "", "normal": "", "material": "", "glow": "", "dirt": "", "ao": "", "paintMask": "", "roughnessMap": "",
                    "tint": tint, "glowColor": glow, "detailScale": detail_scale, "generalDataX": 1.0, "alpha": 1.0, "slots": slots,
                    "semanticComplete": False, "parameterComplete": False, "baselineComplete": False,
                    "missingSemantics": ["sof_visual_manifest"], "unresolvedCount": 1,
                })
        report_path = output_dir / "ship.materials.report.json"
        report_path.write_text(json.dumps({
            "schema": "NSAMDR_BASELINE_REPORT_V1",
            "complete": False,
            "unresolvedCount": len(draw_ranges),
            "reason": reason,
            "areas": [
                {
                    "group": int(draw.get("groupIndex", 0)),
                    "areaName": "tint-only",
                    "areaType": "primary",
                    "pass": "opaque",
                    "shader": "",
                    "shaderFamily": "unknown",
                    "channels": {},
                    "semanticComplete": False,
                    "parameterComplete": False,
                    "baselineComplete": False,
                    "missingSemantics": ["sof_visual_manifest"],
                    "textures": {},
                }
                for draw in draw_ranges
            ],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"WARNING: Created incomplete tint-only material fallback ({reason}): {manifest_path}", flush=True)
        return manifest_path

    # Purpose: Implement prepare asset for EveAssetValidationApplication.
    # Called by: command_prepare_run
    # Calls: _copy_optional_resource, _prepare_sof_materials, _resolve_sof_identity, _write_tint_only_material_manifest, convert_dds, convert_environment_dds, convert_gr2, convert_sof, copy_resource, read_rows, related_textures, resolve_layout, select_environment_sources, select_model, write_ship_catalog
    def prepare_asset(
        self,
        repo_root: Path,
        cache_arg: str,
        query: str,
        selection_key: str = "",
    ) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, list[Path], Path | None, Path, Path, Path]:
        cache_root, indexes, resfiles = self.resolve_layout(cache_arg, allow_prompt=False)
        print(f"EVE SharedCache: {cache_root}", flush=True)
        print("Reading EVE resource indexes...", flush=True)
        rows = self.read_rows(indexes)
        print(f"Indexed resources: {len(rows)}", flush=True)

        catalog_path = self.write_ship_catalog(
            rows,
            repo_root / "artifacts" / "nsamdr" / "eve_assets" / "ship_catalog.tsv",
            repo_root,
        )

        model = self.select_model(rows, query)
        textures = self.related_textures(rows, model)
        environment_scene, environment_textures = self.select_environment_sources(rows, model, resfiles)
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
        gr2_path = self.copy_resource(resfiles, model, output_dir)
        copied_environment_scene = (
            self._copy_optional_resource(resfiles, environment_scene, output_dir, kind="environment scene")
            if environment_scene else None
        )

        copied_environment_textures: list[tuple[ResourceRow, Path]] = []
        for row in environment_textures:
            # Backgrounds are preview context, not reconstruction/training authority.
            # One bad/locked destination must not abort the fixed Raven dataset.
            copied = self._copy_optional_resource(resfiles, row, output_dir / "backgrounds", kind="background")
            if copied is not None:
                copied_environment_textures.append((row, copied))
        copied_environment_texture = copied_environment_textures[0][1] if copied_environment_textures else None
        copied_environment_row = copied_environment_textures[0][0] if copied_environment_textures else None
        copied: dict[str, Path] = {}
        for kind, row in textures.items():
            copied[kind] = self.copy_resource(resfiles, row, output_dir)

        obj_path = self.convert_gr2(repo_root, gr2_path, output_dir / f"{asset_name}.obj")
        conversion_summary = obj_path.with_suffix(".conversion.json")

        material_manifest: Path | None = None
        sof_manifest_path: Path | None = None
        sof_texture_metadata: dict[str, dict] = {}
        sof_identity = self._resolve_sof_identity(rows, repo_root, model, selection_key)
        print(
            "SOF identity: "
            f"hull={sof_identity.get('hull') or '<missing>'}, "
            f"faction={sof_identity.get('faction') or '<missing>'}, "
            f"race={sof_identity.get('race') or '<missing>'} "
            f"({sof_identity.get('raceSource') or 'unknown'})",
            flush=True,
        )
        data_black_row = next((row for row in rows if row.logical.lower() == SOF_DATA_PATH), None)
        fallback_reason = ""
        if data_black_row and sof_identity.get("hull") and sof_identity.get("faction"):
            try:
                data_black = self.copy_resource(resfiles, data_black_row, output_dir / "sof")
                sof_manifest_path = self.convert_sof(
                    repo_root, data_black, output_dir / f"{asset_name}.sof-visuals.json",
                    str(sof_identity["hull"]), str(sof_identity["faction"]), str(sof_identity.get("race") or ""),
                )
                material_manifest, sof_texture_metadata = self._prepare_sof_materials(
                    repo_root, rows, resfiles, output_dir, conversion_summary, sof_manifest_path, str(sof_identity.get("race") or ""),
                )
                print(f"SOF material manifest: {material_manifest}", flush=True)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                fallback_reason = f"SOF extraction failed: {exc}"
        else:
            fallback_reason = "SOF identity or data.black unavailable"

        if material_manifest is None:
            material_manifest = self._write_tint_only_material_manifest(
                output_dir,
                conversion_summary,
                str(sof_identity.get("race") or ""),
                fallback_reason or "SOF visual data unavailable",
            )

        converted_textures: dict[str, Path] = {}
        for kind, copied_path in copied.items():
            try:
                converted_textures[kind] = self.convert_dds(
                    repo_root,
                    copied_path,
                    output_dir / f"{asset_name}_{kind}.png",
                )
            except RuntimeError as exc:
                print(f"WARNING: {kind} texture conversion failed: {exc}", flush=True)

        environment_pngs: list[Path] = []
        environment_records: list[dict] = []
        for index, (row, copied_path) in enumerate(copied_environment_textures):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(row.logical).stem)
            output_path = output_dir / "backgrounds" / f"{index:03d}_{safe_name}.png"
            try:
                converted = self.convert_environment_dds(repo_root, copied_path, output_path)
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
                "scene": (
                    {**asdict(environment_scene), "local": str(copied_environment_scene)}
                    if environment_scene and copied_environment_scene else None
                ),
                "texture": (
                    {**asdict(copied_environment_row), "local": str(copied_environment_texture)}
                    if copied_environment_row and copied_environment_texture else None
                ),
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

    # Purpose: Implement launch preview for EveAssetValidationApplication.
    # Called by: command_prepare_run
    # Calls: No same-class helper methods.
    def launch_preview(
        self,
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
            provenance = strategy_candidates.get("controlProvenance", {})
            primary = provenance.get("primaryAlbedo", {}) if isinstance(provenance, dict) else {}
            env.update({
                "NSAMDR_FINAL_OBJ": str(strategy_candidates.get("candidateObj", "")),
                "NSAMDR_FINAL_MATERIALS": str(strategy_candidates.get("candidateMaterials", "")),
                "NSAMDR_FINAL_ANALYSIS": str(strategy_candidates.get("candidateAnalysis", "")),
                "NSAMDR_FINAL_VALIDATION": str(strategy_candidates.get("candidateValidation", "")),
                "NSAMDR_CANDIDATE_MANIFEST": str(Path(str(strategy_candidates.get("reportPath", ""))) if strategy_candidates.get("reportPath") else ""),
                "NSAMDR_PROVENANCE_STATUS": "VERIFIED" if isinstance(provenance, dict) and provenance.get("verified") else "FAILED",
                "NSAMDR_PROVENANCE_FILE": str(strategy_candidates.get("controlProvenancePath", "")),
                "NSAMDR_PROVENANCE_SOURCE": str(primary.get("sourcePath", "")) if isinstance(primary, dict) else "",
                "NSAMDR_PROVENANCE_SOURCE_SHA": str(primary.get("sourceSha256After", "")) if isinstance(primary, dict) else "",
                "NSAMDR_PROVENANCE_CANDIDATE": str(primary.get("candidatePath", "")) if isinstance(primary, dict) else "",
                "NSAMDR_PROVENANCE_CANDIDATE_SHA": str(primary.get("candidateSha256", "")) if isinstance(primary, dict) else "",
            })
        print("Launching the Granny-free Trinity NSAMDR viewer...", flush=True)
        return subprocess.run(command, cwd=repo_root, env=env, check=False).returncode

    # Purpose: Implement command list for EveAssetValidationApplication.
    # Called by: External callers and the owning workflow.
    # Calls: read_rows, resolve_layout
    def command_list(self, args: argparse.Namespace) -> int:
        _, indexes, _ = self.resolve_layout(args.shared_cache)
        rows = self.read_rows(indexes)
        query = args.query.lower()
        matches = [row for row in rows if query in row.logical.lower()]
        matches.sort(key=lambda row: row.logical.lower())
        for row in matches[: args.limit]:
            print(f"{row.logical},{row.hashed}")
        print(f"Matches shown: {min(len(matches), args.limit)} of {len(matches)}", file=sys.stderr)
        return 0

    # Purpose: Implement command prepare run for EveAssetValidationApplication.
    # Called by: External callers and the owning workflow.
    # Calls: launch_preview, prepare_asset
    def command_prepare_run(self, args: argparse.Namespace) -> int:
        repo_root = Path(args.repo_root).resolve()
        obj_path, albedo, normal, pgs, environment, environments, material_manifest, manifest, catalog, cache_root = self.prepare_asset(
            repo_root, args.shared_cache, args.query, args.selection_key
        )
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        selected_query = str(manifest_data.get("model", {}).get("logical") or args.query)
        selected_catalog_key = str(args.selection_key or selected_query)

        strategy_candidates: dict[str, object] | None = None
        if material_manifest and material_manifest.is_file():
            candidate_manifest_raw = os.environ.get("NSAMDR_CANDIDATE_MANIFEST", "").strip()
            if not candidate_manifest_raw:
                raise RuntimeError(
                    "Direct EVE candidate generation is not a production checkpoint-selection path. "
                    r"Use scripts\build\nsamdr.bat preview EXP_####."
                )
            candidate_report = Path(candidate_manifest_raw).resolve()
            if not candidate_report.is_file():
                raise RuntimeError(f"NSAMDR candidate manifest is missing: {candidate_report}")
            strategy_candidates = json.loads(candidate_report.read_text(encoding="utf-8"))
            required = (
                "candidateObj",
                "candidateMaterials",
                "candidateAnalysis",
                "candidateValidation",
                "checkpointSha256",
            )
            missing = [key for key in required if not str(strategy_candidates.get(key, "")).strip()]
            if missing:
                raise RuntimeError(
                    "NSAMDR candidate manifest is incomplete; missing: " + ", ".join(missing)
                )
            strategy_candidates["reportPath"] = str(candidate_report)

        return self.launch_preview(
            repo_root, Path(args.launcher).resolve(), obj_path, albedo, normal, pgs, environment, environments, material_manifest,
            manifest, catalog, cache_root, selected_catalog_key, strategy_candidates
        )

    # Purpose: Implement command convert gr2 for EveAssetValidationApplication.
    # Called by: External callers and the owning workflow.
    # Calls: convert_gr2
    def command_convert_gr2(self, args: argparse.Namespace) -> int:
        self.convert_gr2(Path(args.repo_root).resolve(), Path(args.input).resolve(), Path(args.output).resolve())
        return 0

    # Purpose: Implement build parser for EveAssetValidationApplication.
    # Called by: main
    # Calls: No same-class helper methods.
    def build_parser(self) -> argparse.ArgumentParser:
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
        prepare.add_argument(
            "--neural-checkpoint-dir",
            default="",
            help=(
                "Directory containing nsamdr_tile_context.pt/json. "
                "Defaults to NSAMDR_NEURAL_CHECKPOINT_DIR or the production neural directory."
            ),
        )
        prepare.add_argument("--launcher", required=True)
        prepare.set_defaults(func=command_prepare_run)

        convert = sub.add_parser("convert-gr2", help="Convert a local GR2 file to OBJ using CarbonEngineJS")
        convert.add_argument("--repo-root", required=True)
        convert.add_argument("--input", required=True)
        convert.add_argument("--output", required=True)
        convert.set_defaults(func=command_convert_gr2)
        return parser

    # Purpose: Implement main for EveAssetValidationApplication.
    # Called by: External callers and the owning workflow.
    # Calls: build_parser, eprint
    def main(self) -> int:
        args = self.build_parser().parse_args()
        try:
            return int(args.func(args))
        except KeyboardInterrupt:
            self.eprint("Cancelled.")
            return 130
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            self.eprint(f"ERROR: {exc}")
            return 1

_eve_asset_validation_application = EveAssetValidationApplication()
eprint = _eve_asset_validation_application.eprint
_add_candidate = _eve_asset_validation_application._add_candidate
_registry_cache_roots = _eve_asset_validation_application._registry_cache_roots
_read_registry_string = _eve_asset_validation_application._read_registry_string
_installed_program_roots = _eve_asset_validation_application._installed_program_roots
_launcher_config_cache_roots = _eve_asset_validation_application._launcher_config_cache_roots
_launcher_log_cache_roots = _eve_asset_validation_application._launcher_log_cache_roots
_saved_cache_roots = _eve_asset_validation_application._saved_cache_roots
_save_cache_root = _eve_asset_validation_application._save_cache_root
candidate_cache_roots = _eve_asset_validation_application.candidate_cache_roots
_layout_variants = _eve_asset_validation_application._layout_variants
_inspect_layout = _eve_asset_validation_application._inspect_layout
_prompt_for_cache = _eve_asset_validation_application._prompt_for_cache
resolve_layout = _eve_asset_validation_application.resolve_layout
read_rows = _eve_asset_validation_application.read_rows
select_model = _eve_asset_validation_application.select_model
related_textures = _eve_asset_validation_application.related_textures
_environment_code_for_model = _eve_asset_validation_application._environment_code_for_model
_resource_references_from_bytes = _eve_asset_validation_application._resource_references_from_bytes
select_environment_source = _eve_asset_validation_application.select_environment_source
select_environment_sources = _eve_asset_validation_application.select_environment_sources
_safe_text = _eve_asset_validation_application._safe_text
_sde_cache_directory = _eve_asset_validation_application._sde_cache_directory
_download_with_progress = _eve_asset_validation_application._download_with_progress
_ensure_sde_archive = _eve_asset_validation_application._ensure_sde_archive
_find_zip_member = _eve_asset_validation_application._find_zip_member
_read_jsonl_member = _eve_asset_validation_application._read_jsonl_member
_normalise_resource_path = _eve_asset_validation_application._normalise_resource_path
_model_family_key = _eve_asset_validation_application._model_family_key
_asset_quality_score = _eve_asset_validation_application._asset_quality_score
_candidate_assets_for_graphic = _eve_asset_validation_application._candidate_assets_for_graphic
_expand_asset_variants = _eve_asset_validation_application._expand_asset_variants
_infer_fallback_display = _eve_asset_validation_application._infer_fallback_display
_build_sde_ship_catalog = _eve_asset_validation_application._build_sde_ship_catalog
_build_fallback_ship_catalog = _eve_asset_validation_application._build_fallback_ship_catalog
_tsv_clean = _eve_asset_validation_application._tsv_clean
write_ship_catalog = _eve_asset_validation_application.write_ship_catalog
source_path = _eve_asset_validation_application.source_path
_same_size_file = _eve_asset_validation_application._same_size_file
copy_resource = _eve_asset_validation_application.copy_resource
_copy_optional_resource = _eve_asset_validation_application._copy_optional_resource
find_command = _eve_asset_validation_application.find_command
probe_converter_modules = _eve_asset_validation_application.probe_converter_modules
ensure_converter = _eve_asset_validation_application.ensure_converter
run_checked = _eve_asset_validation_application.run_checked
convert_gr2 = _eve_asset_validation_application.convert_gr2
convert_dds = _eve_asset_validation_application.convert_dds
convert_environment_dds = _eve_asset_validation_application.convert_environment_dds
_race_from_ship_model = _eve_asset_validation_application._race_from_ship_model
_resolve_sof_identity = _eve_asset_validation_application._resolve_sof_identity
convert_sof = _eve_asset_validation_application.convert_sof
_resource_row = _eve_asset_validation_application._resource_row
_faction_texture_path = _eve_asset_validation_application._faction_texture_path
_resolve_sof_texture = _eve_asset_validation_application._resolve_sof_texture
_numeric_vector = _eve_asset_validation_application._numeric_vector
_normal_key = _eve_asset_validation_application._normal_key
_find_named_value = _eve_asset_validation_application._find_named_value
_faction_color = _eve_asset_validation_application._faction_color
_material_for_area = _eve_asset_validation_application._material_for_area
_clamp = _eve_asset_validation_application._clamp
_numeric_scalar = _eve_asset_validation_application._numeric_scalar
_area_base_visuals = _eve_asset_validation_application._area_base_visuals
_material_names_for_area = _eve_asset_validation_application._material_names_for_area
_lookup_material_parameters = _eve_asset_validation_application._lookup_material_parameters
_slot_fallback_colors = _eve_asset_validation_application._slot_fallback_colors
_parameter_vector = _eve_asset_validation_application._parameter_vector
_parameter_scalar = _eve_asset_validation_application._parameter_scalar
_material_slot_visuals = _eve_asset_validation_application._material_slot_visuals
_area_visual_parameters = _eve_asset_validation_application._area_visual_parameters
_area_material_slots = _eve_asset_validation_application._area_material_slots
_exact_parameter = _eve_asset_validation_application._exact_parameter
_gloss_to_roughness = _eve_asset_validation_application._gloss_to_roughness
_resolved_area_material_slots = _eve_asset_validation_application._resolved_area_material_slots
_texture_usage = _eve_asset_validation_application._texture_usage
_texture_suffix = _eve_asset_validation_application._texture_suffix
_texture_by_suffix = _eve_asset_validation_application._texture_by_suffix
_classify_shader_family = _eve_asset_validation_application._classify_shader_family
_semantic_texture_layout = _eve_asset_validation_application._semantic_texture_layout
_material_manifest_columns = _eve_asset_validation_application._material_manifest_columns
_write_material_record = _eve_asset_validation_application._write_material_record
_prepare_sof_materials = _eve_asset_validation_application._prepare_sof_materials
_write_tint_only_material_manifest = _eve_asset_validation_application._write_tint_only_material_manifest
prepare_asset = _eve_asset_validation_application.prepare_asset
launch_preview = _eve_asset_validation_application.launch_preview
command_list = _eve_asset_validation_application.command_list
command_prepare_run = _eve_asset_validation_application.command_prepare_run
command_convert_gr2 = _eve_asset_validation_application.command_convert_gr2
build_parser = _eve_asset_validation_application.build_parser
main = _eve_asset_validation_application.main


ENVIRONMENT_SCENES = {
    "amarr": "a03",
    "caldari": "c02",
    "gallente": "g03",
    "minmatar": "m02",
}


CONVERTER_MODULE_PROBE = (
    "await import('@carbonenginejs/format-gr2'); "
    "await import('@carbonenginejs/runtime-resource/formats/dds');"
)


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


SEMANTIC_CHANNEL_RED = 0
SEMANTIC_CHANNEL_GREEN = 1
SEMANTIC_CHANNEL_BLUE = 2
SEMANTIC_CHANNEL_ALPHA = 3


if __name__ == "__main__":
    raise SystemExit(main())
