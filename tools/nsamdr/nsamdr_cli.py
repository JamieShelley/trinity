#!/usr/bin/env python3
"""Single NSAMDR V17.6 operator command surface.

Historical proof modules remain internal provenance dependencies. No legacy
training engine, preview source or GUI is selectable from this CLI.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
NEURAL_ROOT = REPO_ROOT / "tools/nsamdr/neural"
CUDA_PYTHON = REPO_ROOT / "artifacts/nsamdr/python-env/Scripts/python.exe"
CPU_PYTHON = REPO_ROOT / "artifacts/nsamdr/python-env-cpu/Scripts/python.exe"


class NSAMDRCommandLineApplication:
    def _python(self, kind: str = "cuda", *, bootstrap: bool = True) -> Path:
        requested = os.environ.get("NSAMDR_PYTHON_EXE", "").strip()
        if requested:
            candidate = Path(requested).expanduser()
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if not candidate.is_file():
                raise RuntimeError(f"NSAMDR_PYTHON_EXE not found: {candidate}")
            return candidate.resolve()
        candidate = CUDA_PYTHON if kind == "cuda" else CPU_PYTHON
        if candidate.is_file():
            return candidate
        if not bootstrap:
            return Path(sys.executable)
        script = REPO_ROOT / f"scripts/build/setup_nsamdr_{kind}.bat"
        if os.name != "nt":
            raise RuntimeError("NSAMDR environment bootstrap requires Windows; set NSAMDR_PYTHON_EXE")
        result = subprocess.run(["cmd.exe", "/d", "/s", "/c", f'"{script}"'], cwd=REPO_ROOT,
                                check=False)
        if result.returncode:
            raise RuntimeError(f"CUDA environment setup failed: {result.returncode}")
        if not candidate.is_file():
            raise RuntimeError(f"NSAMDR setup did not create {candidate}")
        return candidate

    @staticmethod
    def _run(command: Sequence[object]) -> int:
        argv = [os.fspath(v) for v in command]
        print("[nsamdr] RUN: " + subprocess.list2cmdline(argv), flush=True)
        return subprocess.run(argv, cwd=REPO_ROOT, check=False).returncode

    @staticmethod
    def _git(*args: str) -> tuple[int, str]:
        completed = subprocess.run(["git", *args], cwd=REPO_ROOT,
                                   capture_output=True, text=True, check=False)
        return completed.returncode, completed.stdout.strip()

    def _source_freshness_preflight(self) -> int:
        if not (REPO_ROOT / ".git").exists():
            print("[nsamdr] WARNING: Git metadata unavailable; source freshness unverified.", flush=True)
            return 0
        status_code, status = self._git("status", "--porcelain=v1", "--untracked-files=no")
        sha_code, head = self._git("rev-parse", "HEAD")
        branch_code, branch = self._git("branch", "--show-current")
        if status_code or sha_code or branch_code:
            print("ERROR: Git source provenance could not be read.", file=sys.stderr)
            return 5
        if status:
            print("ERROR: tracked source files modified; commit or restore before V17.6 execution.",
                  file=sys.stderr)
            print(status, file=sys.stderr)
            return 5
        if branch != "NSAMDR":
            print(f"[nsamdr] WARNING: branch={branch}; origin/NSAMDR comparison skipped.", flush=True)
            return 0
        rc, remote = self._git("ls-remote", "origin", "refs/heads/NSAMDR")
        if rc or not remote:
            print("[nsamdr] WARNING: remote HEAD unavailable; local provenance only.", flush=True)
            return 0
        upstream = remote.split()[0].strip()
        if head != upstream:
            print(f"ERROR: stale checkout {head}; origin/NSAMDR={upstream}", file=sys.stderr)
            print("Action: git fetch origin refs/heads/NSAMDR:refs/remotes/origin/NSAMDR",
                  file=sys.stderr)
            print("        git merge --ff-only refs/remotes/origin/NSAMDR", file=sys.stderr)
            return 5
        print(f"[nsamdr] Source revision verified: NSAMDR {head}", flush=True)
        return 0

    def _cuda_preflight(self, python: Path) -> int:
        return self._run([python, "-u", NEURAL_ROOT / "verify_cuda.py",
                          "--require-arch", "sm_120", "--quick"])

    def _command_gui(self, _: argparse.Namespace) -> int:
        python = self._python("cpu", bootstrap=False)
        return self._run([python, "-u", REPO_ROOT / "tools/nsamdr/gui/nsamdr_v176_workflow_gui.py"])

    def _command_setup(self, args: argparse.Namespace) -> int:
        if os.name != "nt":
            raise RuntimeError("Setup requires Windows")
        script = REPO_ROOT / f"scripts/build/setup_nsamdr_{args.kind}.bat"
        return self._run(["cmd.exe", "/d", "/s", "/c", f'"{script}"' + (" --force" if args.force else "")])

    def _command_v176_detail(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if not args.audit_only:
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        argv = [python, "-u", NEURAL_ROOT / "probe_nsamdr_v176_detail.py",
                "--repo-root", REPO_ROOT, "--resume", args.resume,
                "--amp-precision", args.amp_precision]
        if args.audit_only:
            argv += ["--audit-only"]
        if args.stage9_report:
            argv += ["--stage9-report", args.stage9_report]
        if args.v3_report:
            argv += ["--v3-report", args.v3_report]
        return self._run(argv)

    def _command_v176_capacity(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if not args.audit_only and args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        argv = [python, "-u", NEURAL_ROOT / "probe_nsamdr_v176_capacity.py",
                "--repo-root", REPO_ROOT, "--phase", args.phase,
                "--resume", args.resume, "--device", args.device,
                "--amp-precision", args.amp_precision]
        if args.audit_only:
            argv += ["--audit-only"]
        for name, flag in (("stage10a_report", "--stage10a-report"),
                           ("comparison_report", "--comparison-report"),
                           ("v3_report", "--v3-report")):
            value = getattr(args, name)
            if value:
                argv += [flag, value]
        return self._run(argv)

    def _command_v177_enhancement(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        argv = [python, "-u", NEURAL_ROOT / "train_nsamdr_v177_enhancement.py",
                "--repo-root", REPO_ROOT, "--device", args.device,
                "--steps", str(args.steps), "--batch-size", str(args.batch_size),
                "--patch-size", str(args.patch_size), "--seed", str(args.seed),
                "--learning-rate", str(args.learning_rate), "--resume", args.resume]
        return self._run(argv)

    def _command_raven(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        argv = [python, "-u", NEURAL_ROOT / "render_nsamdr_v176_raven_preview.py",
                "--repo-root", REPO_ROOT, "--shared-cache", args.shared_cache,
                "--device", args.device,
                "--authority-index", str(args.authority_index), "--mode", args.mode]
        if args.comparison_report:
            argv += ["--comparison-report", args.comparison_report]
        if args.enhancer_checkpoint:
            argv += ["--enhancer-checkpoint", args.enhancer_checkpoint]
        return self._run(argv)

    def _command_raven_channel_audit(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        argv = [python, "-u",
                NEURAL_ROOT / "audit_nsamdr_v176_raven_channels.py",
                "--repo-root", REPO_ROOT,
                "--shared-cache", args.shared_cache,
                "--crop-size", str(args.crop_size),
                "--max-textures", str(args.max_textures)]
        if args.crop_x is not None or args.crop_y is not None:
            if args.crop_x is None or args.crop_y is None:
                raise ValueError("both --crop-x and --crop-y are required")
            argv += ["--crop-x", str(args.crop_x), "--crop-y", str(args.crop_y)]
        return self._run(argv)

    def _command_v178_golden(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        cmd = [python, "-u", NEURAL_ROOT / "golden_nsamdr_v178.py",
               "--repo-root", REPO_ROOT, "--action", args.action,
               "--shared-cache", args.shared_cache,
               "--crop-x", str(args.crop_x), "--crop-y", str(args.crop_y),
               "--size", str(args.size)]
        if args.source_name:
            cmd += ["--source-name", args.source_name]
        if args.folder:
            cmd += ["--folder", args.folder]
        return self._run(cmd)

    def _command_v178_contour(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        return self._run([
            python, "-u", NEURAL_ROOT / "train_nsamdr_v178_contour.py",
            "--repo-root", REPO_ROOT,
            "--device", args.device, "--steps", str(args.steps),
            "--batch-size", str(args.batch_size), "--patch-size", str(args.patch_size),
            "--seed", str(args.seed), "--learning-rate", str(args.learning_rate),
            "--resume", args.resume,
        ])

    def _command_v178_real_fit(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        return self._run([
            python, "-u", NEURAL_ROOT / "fit_nsamdr_v178_golden.py",
            "--folder", args.folder, "--device", args.device,
            "--steps", str(args.steps), "--patch-size", str(args.patch_size),
            "--seed", str(args.seed), "--resume", args.resume,
        ])

    def _command_v178_auto(self, args: argparse.Namespace) -> int:
        rc = self._source_freshness_preflight()
        if rc:
            return rc
        python = self._python("cuda")
        if args.device == "cuda":
            rc = self._cuda_preflight(python)
            if rc:
                return rc
        argv = [python, "-u", NEURAL_ROOT / "preview_nsamdr_v178_automatic.py",
                "--folder", args.folder, "--device", args.device,
                "--tile", str(args.tile)]
        if args.checkpoint:
            argv.extend(("--checkpoint", args.checkpoint))
        return self._run(argv)

    def _command_tests(self, args: argparse.Namespace) -> int:
        python = self._python("cuda")
        cases = [
            "tools.nsamdr.test_nsamdr_cli",
            "tools.nsamdr.neural.test_nsamdr_v176_detail",
            "tools.nsamdr.neural.test_nsamdr_v176_training",
            "tools.nsamdr.neural.test_nsamdr_v176_capacity",
            "tools.nsamdr.neural.test_nsamdr_v176_stability",
            "tools.nsamdr.neural.test_nsamdr_v176_attainable",
            "tools.nsamdr.neural.test_nsamdr_v176_raven_preview",
            "tools.nsamdr.neural.test_nsamdr_v177_baseline",
            "tools.nsamdr.neural.test_nsamdr_v177_enhancement",
            "tools.nsamdr.neural.test_nsamdr_v178_golden",
            "tools.nsamdr.neural.test_nsamdr_v176_channel_audit",
            "tools.nsamdr.gui.test_nsamdr_v176_workflow_gui",
        ]
        return self._run([python, "-m", "unittest", "-v", *cases])

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("gui").set_defaults(handler=self._command_gui)
        setup = commands.add_parser("setup")
        setup.add_argument("kind", choices=("cuda", "cpu"))
        setup.add_argument("--force", action="store_true")
        setup.set_defaults(handler=self._command_setup)
        detail = commands.add_parser("v176-detail-proof")
        detail.add_argument("--stage9-report", default="")
        detail.add_argument("--v3-report", default="")
        detail.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
        detail.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
        detail.add_argument("--audit-only", action="store_true")
        detail.set_defaults(handler=self._command_v176_detail)
        capacity = commands.add_parser("v176-capacity-proof")
        capacity.add_argument("--phase", choices=("diagnose", "stability", "capacity",
                                                  "compare-features", "attainable"), default="diagnose")
        capacity.add_argument("--stage10a-report", default="")
        capacity.add_argument("--comparison-report", default="")
        capacity.add_argument("--v3-report", default="")
        capacity.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        capacity.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
        capacity.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
        capacity.add_argument("--audit-only", action="store_true")
        capacity.set_defaults(handler=self._command_v176_capacity)
        enhanced = commands.add_parser("v177-enhancement-train")
        enhanced.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        enhanced.add_argument("--steps", type=int, default=300)
        enhanced.add_argument("--batch-size", type=int, default=4)
        enhanced.add_argument("--patch-size", type=int, default=32)
        enhanced.add_argument("--seed", type=int, default=178)
        enhanced.add_argument("--learning-rate", type=float, default=.001)
        enhanced.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
        enhanced.set_defaults(handler=self._command_v177_enhancement)
        golden = commands.add_parser("v178-golden-patch")
        golden.add_argument("--action", choices=("prepare", "render-trace"), default="prepare")
        golden.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        golden.add_argument("--source-name", default="")
        golden.add_argument("--crop-x", type=int, default=320)
        golden.add_argument("--crop-y", type=int, default=448)
        golden.add_argument("--size", type=int, default=256)
        golden.add_argument("--folder", default="")
        golden.set_defaults(handler=self._command_v178_golden)
        contour = commands.add_parser("v178-contour-train")
        contour.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        contour.add_argument("--steps", type=int, default=400)
        contour.add_argument("--batch-size", type=int, default=4)
        contour.add_argument("--patch-size", type=int, default=32)
        contour.add_argument("--seed", type=int, default=17812)
        contour.add_argument("--learning-rate", type=float, default=.0007)
        contour.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
        contour.set_defaults(handler=self._command_v178_contour)
        real_fit = commands.add_parser("v178-golden-fit")
        real_fit.add_argument("--folder", required=True)
        real_fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        real_fit.add_argument("--steps", type=int, default=300)
        real_fit.add_argument("--patch-size", type=int, default=48)
        real_fit.add_argument("--seed", type=int, default=17803)
        real_fit.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
        real_fit.set_defaults(handler=self._command_v178_real_fit)
        auto = commands.add_parser("v178-golden-auto")
        auto.add_argument("--folder", required=True)
        auto.add_argument("--checkpoint", default="")
        auto.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        auto.add_argument("--tile", type=int, default=96)
        auto.set_defaults(handler=self._command_v178_auto)
        raven = commands.add_parser("v176-raven-preview")
        raven.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        raven.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
        raven.add_argument("--authority-index", type=int, choices=(1, 10), default=10)
        raven.add_argument("--mode", choices=("output10", "hr_features58"), default="output10")
        raven.add_argument("--comparison-report", default="")
        raven.add_argument("--enhancer-checkpoint", default="",
                           help="Explicit validated synthetic-trained V17.7 model")
        raven.set_defaults(handler=self._command_raven)
        audit = commands.add_parser("v176-raven-channel-audit")
        audit.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        audit.add_argument("--crop-size", type=int, default=128)
        audit.add_argument("--max-textures", type=int, default=6)
        audit.add_argument("--crop-x", type=int)
        audit.add_argument("--crop-y", type=int)
        audit.set_defaults(handler=self._command_raven_channel_audit)
        tests = commands.add_parser("test")
        tests.set_defaults(handler=self._command_tests)
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        try:
            return int(args.handler(args))
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1


_app = NSAMDRCommandLineApplication()
if __name__ == "__main__":
    raise SystemExit(_app.main())
