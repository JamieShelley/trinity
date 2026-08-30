#!/usr/bin/env python3
"""Apply the NSAMDR quality setting to Trinity's real scene-renderer API.

This patch is deliberately anchor-based and idempotent so it can be applied to
an existing Carbon/Trinity checkout without replacing whole upstream source
files.  It wires the shared NSAMDRQuality policy into
EveSpaceSceneRenderDriver::Settings and Blue exposure, which is the boundary a
Carbon/EVE Display & Graphics UI should bind to.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


class TrinitySettingsApplication:
    # Purpose: Implement fail for TrinitySettingsApplication.
    # Called by: _insert_after, _insert_before, apply
    # Calls: No same-class helper methods.
    def _fail(self, message: str) -> None:
        raise RuntimeError(message)

    # Purpose: Implement insert after for TrinitySettingsApplication.
    # Called by: apply
    # Calls: _fail
    def _insert_after(self, text: str, anchor: str, insertion: str, *, label: str) -> str:
        if insertion.strip() in text:
            return text
        if anchor not in text:
            self._fail(f"cannot locate {label} anchor; Trinity source layout changed")
        return text.replace(anchor, anchor + insertion, 1)

    # Purpose: Implement insert before for TrinitySettingsApplication.
    # Called by: apply
    # Calls: _fail
    def _insert_before(self, text: str, anchor: str, insertion: str, *, label: str) -> str:
        if insertion.strip() in text:
            return text
        if anchor not in text:
            self._fail(f"cannot locate {label} anchor; Trinity source layout changed")
        return text.replace(anchor, insertion + anchor, 1)

    # Purpose: Implement apply for TrinitySettingsApplication.
    # Called by: main
    # Calls: _fail, _insert_after, _insert_before
    def apply(self, repo_root: Path, *, check_only: bool = False) -> bool:
        repo_root = repo_root.resolve()
        header = repo_root / "trinity" / "Eve" / "EveSpaceSceneRenderDriver.h"
        blue = repo_root / "trinity" / "Eve" / "EveSpaceSceneRenderDriver_Blue.cpp"
        shared = repo_root / "trinity" / "NSAMDR" / "NSAMDRSettings.h"

        for path in (header, blue, shared):
            if not path.is_file():
                self._fail(f"required file is missing: {path}")

        h = header.read_text(encoding="utf-8")
        b = blue.read_text(encoding="utf-8")

        h = self._insert_after(
            h,
            '#include "../Tr2ProfileTimer.h"\n',
            '#include "../NSAMDR/NSAMDRSettings.h"\n',
            label="renderer header include",
        )
        h = self._insert_after(
            h,
            '\t\tTr2VolumerticQuality volumetricQuality = Tr2VolumerticQuality::High;\n',
            '\t\tnsamdr::NSAMDRQuality neuralSurfaceReconstructionQuality = nsamdr::NSAMDRQuality::Off;\n',
            label="renderer Settings",
        )

        b = self._insert_after(
            b,
            '#include "../Tr2TextureReference.h"\n',
            '\nusing NSAMDRQuality = nsamdr::NSAMDRQuality;\n',
            label="Blue NSAMDR type alias",
        )

        chooser = '''Be::VarChooser NSAMDRQualityChooser[] = {\n\t{ "Off", BeCast( NSAMDRQuality::Off ), "Disable neural surface reconstruction" },\n\t{ "Balanced", BeCast( NSAMDRQuality::Balanced ), "Ships, conservative 2x reconstruction policy" },\n\t{ "High", BeCast( NSAMDRQuality::High ), "Ships and structures, high-quality reconstruction policy" },\n\t{ "Ultra", BeCast( NSAMDRQuality::Ultra ), "Maximum eligible hard-surface reconstruction policy" },\n\t{ 0 }\n};\n'''
        b = self._insert_before(
            b,
            'const Be::VarChooser TriRMChooser[] = {\n',
            chooser,
            label="Blue chooser",
        )

        registration = 'BLUE_REGISTER_ENUM_EX( "NSAMDRQuality", NSAMDRQuality, NSAMDRQualityChooser, ENUM_REG_ENUM_OBJECT_ON_MODULE );\n'
        b = self._insert_after(
            b,
            'BLUE_REGISTER_ENUM_EX( "ShadowQuality", ShadowQuality, ShadowQualityChooser, ENUM_REG_ENUM_OBJECT_ON_MODULE );\n',
            registration,
            label="Blue enum registration",
        )

        exposure = '''\n\t\tMAP_ATTRIBUTE_WITH_CHOOSER(\n\t\t\t"neuralSurfaceReconstructionQuality",\n\t\t\tm_settings.neuralSurfaceReconstructionQuality,\n\t\t\t"Requested NSAMDR neural surface reconstruction quality. Carbon/EVE UI should bind to this single policy setting; Trinity owns the detailed reconstruction policy.",\n\t\t\tBe::READWRITE | Be::ENUM,\n\t\t\tNSAMDRQualityChooser )\n'''
        enable_upscaling = '''\t\tMAP_ATTRIBUTE(\n\t\t\t"enableUpscaling",\n\t\t\tm_settings.enableUpscaling,\n\t\t\t"Allows disabling upscaling even if it is enabled globally",\n\t\t\tBe::READWRITE )\n'''
        if '"neuralSurfaceReconstructionQuality"' not in b:
            if enable_upscaling not in b:
                self._fail("cannot locate Blue enableUpscaling exposure; Trinity source layout changed")
            b = b.replace(enable_upscaling, enable_upscaling + exposure, 1)

        required_h = (
            '../NSAMDR/NSAMDRSettings.h',
            'neuralSurfaceReconstructionQuality',
            'nsamdr::NSAMDRQuality::Off',
        )
        required_b = (
            'NSAMDRQualityChooser',
            'BLUE_REGISTER_ENUM_EX( "NSAMDRQuality"',
            '"neuralSurfaceReconstructionQuality"',
            'm_settings.neuralSurfaceReconstructionQuality',
        )
        for token in required_h:
            if token not in h:
                self._fail(f"renderer header verification failed: {token}")
        for token in required_b:
            if token not in b:
                self._fail(f"Blue exposure verification failed: {token}")

        changed = h != header.read_text(encoding="utf-8") or b != blue.read_text(encoding="utf-8")
        if not check_only and changed:
            header.write_text(h, encoding="utf-8")
            blue.write_text(b, encoding="utf-8")

        if check_only:
            status = "CHANGES REQUIRED (dry run)" if changed else "ALREADY APPLIED"
        else:
            status = "APPLIED" if changed else "ALREADY APPLIED"
        print("NSAMDR Trinity graphics-setting integration: " + status)
        print("  property : neuralSurfaceReconstructionQuality")
        print("  values   : Off / Balanced / High / Ultra")
        print("  default  : Off")
        print("  Blue API : NSAMDRQuality")
        return changed

    # Purpose: Implement main for TrinitySettingsApplication.
    # Called by: External callers and the owning workflow.
    # Calls: apply
    def main(self) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument(
            "--check",
            action="store_true",
            help="dry-run the integration and verify that its anchors/output remain valid",
        )
        args = parser.parse_args()
        try:
            self.apply(args.repo_root, check_only=args.check)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

_trinity_settings_application = TrinitySettingsApplication()
_fail = _trinity_settings_application._fail
_insert_after = _trinity_settings_application._insert_after
_insert_before = _trinity_settings_application._insert_before
apply = _trinity_settings_application.apply
main = _trinity_settings_application.main


if __name__ == "__main__":
    raise SystemExit(main())
