#!/usr/bin/env python3
"""One operator stage: build/reuse Raven set -> train experiment -> renderer preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], *, cwd: Path) -> None:
    print("[tuning-stage] RUN: " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _set_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            raise RuntimeError(f"malformed command option {option}")
        result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def _update_manifest(experiment_dir: Path, **values) -> None:
    path = experiment_dir / "experiment.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined NSAMDR V9 Raven tune + preview stage")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-config", default="tools/nsamdr/neural/configs/v9_preview_raven.json")
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--max-train-regions", type=int, default=12)
    parser.add_argument("--max-validation-regions", type=int, default=4)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--experiment", default="new")
    parser.add_argument("--control", choices=("auto", "resume", "restart"), default="auto")
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--preset", default="Baseline")
    parser.add_argument("--preview-target-size", type=int, default=4096)
    parser.add_argument("--preview-device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--force-candidate", action="store_true")
    parser.add_argument("--geometry-audit", dest="geometry_audit", action="store_true", default=True)
    parser.add_argument("--no-geometry-audit", dest="geometry_audit", action="store_false")
    parser.add_argument("--geometry-critic", choices=("off", "auto", "required"), default="auto")
    parser.add_argument("--geometry-audit-policy", choices=("report", "strict"), default="report")
    parser.add_argument("--geometry-evidence-regions", type=int, default=12)
    parser.add_argument("--critic-steps", type=int, default=120)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--performance-profile", default="fast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-precision", default="auto")
    # Forwarded semantic tuning controls.
    for name in (
        "learning-rate", "weight-decay", "optimizer-name", "scheduler-name",
        "batch-size", "tiles-per-epoch", "validation-tiles", "augmentation-strength",
        "regret-weight", "normal-regret-weight", "edge-weight", "detail-laplacian-weight",
        "geometric-alignment-weight", "tangent-coherence-weight", "curvature-coherence-weight",
        "synthetic-geometry-probability", "boundary-sampling-probability",
        "boundary-renderer-band-pixels", "boundary-renderer-sample-pixels",
        "boundary-renderer-hard-width-pixels", "boundary-renderer-soft-width-pixels",
        "boundary-renderer-gate-gain",
        "boundary-renderer-far-sample-multiplier", "boundary-renderer-far-sample-weight",
        "boundary-gate-need-scale", "boundary-gate-exact-floor",
        "boundary-sdf-zero-weight", "boundary-edge-sdf-consistency-weight",
        "boundary-pixel-regret-weight",
        "boundary-profile-weight", "boundary-regret-weight",
        "sdf-surface-weight", "sdf-sign-weight", "sdf-eikonal-weight",
        "sdf-gradient-alignment-weight", "sdf-metric-gradient-weight",
        "sdf-metric-band-pixels", "sdf-coarse-init-std",
        "sdf-synthetic-validation-tiles",
        "sdf-zero-band-pixels", "sdf-bootstrap-residual-pixels",
        "sdf-proof-residual-pixels", "sdf-proof-renderer-weight",
        "implicit-sdf-hidden-channels", "implicit-sdf-residual-pixels",
        "coarse-sdf-surface-weight", "sdf-residual-l1-weight",
        "boundary-fuzz-weight", "boundary-halo-weight",
        "boundary-renderer-plateau-samples",
        "boundary-renderer-plateau-max-multiplier",
        "boundary-renderer-plateau-stability-scale", "seed",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--randomise-seed", action="store_true")
    parser.add_argument("--advanced-overrides", default="")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    python = sys.executable
    base_config = str((root / args.base_config).resolve()) if not Path(args.base_config).is_absolute() else args.base_config

    print("=" * 72, flush=True)
    print("NSAMDR V9 COMBINED RAVEN TUNE + PREVIEW", flush=True)
    print(f"Training mode            : {args.training_mode}", flush=True)
    print("Pipeline                 : Stage-A renderer preflight -> metric SDF -> Stage-B SDF proof -> learned gate -> Raven audit", flush=True)
    print(f"Geometry audit           : {'enabled' if args.geometry_audit else 'disabled'} ({args.geometry_audit_policy})", flush=True)
    print(f"Geometry critic          : {args.geometry_critic}", flush=True)
    print("=" * 72, flush=True)

    dataset_cmd = [
        python, "-u", "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py",
        "--repo-root", str(root), "--config", base_config,
        "--shared-cache", args.shared_cache,
        "--train-crops", str(args.max_train_regions),
        "--validation-crops", str(args.max_validation_regions),
    ]
    if args.rebuild_dataset:
        dataset_cmd.append("--rebuild")
    _run(dataset_cmd, cwd=root)

    result_file = root / "artifacts/nsamdr/gui/last_raven_tune_result.json"
    result_file.unlink(missing_ok=True)
    train_cmd = [
        python, "-u", "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py",
        "--repo-root", str(root), "--base-config", base_config,
        "--experiment", args.experiment, "--control", args.control,
        "--training-mode", args.training_mode, "--preset", args.preset,
        "--performance-profile", args.performance_profile,
        "--workers", str(args.workers), "--prefetch-factor", str(args.prefetch_factor),
        "--amp-precision", args.amp_precision,
        "--early-stop-patience", str(args.early_stop_patience),
        "--early-stop-min-delta", str(args.early_stop_min_delta),
        "--result-file", str(result_file),
    ]
    forwarded = {
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "optimizer-name": args.optimizer_name,
        "scheduler-name": args.scheduler_name,
        "batch-size": args.batch_size,
        "tiles-per-epoch": args.tiles_per_epoch,
        "validation-tiles": args.validation_tiles,
        "augmentation-strength": args.augmentation_strength,
        "regret-weight": args.regret_weight,
        "normal-regret-weight": args.normal_regret_weight,
        "edge-weight": args.edge_weight,
        "detail-laplacian-weight": args.detail_laplacian_weight,
        "geometric-alignment-weight": args.geometric_alignment_weight,
        "tangent-coherence-weight": args.tangent_coherence_weight,
        "curvature-coherence-weight": args.curvature_coherence_weight,
        "synthetic-geometry-probability": args.synthetic_geometry_probability,
        "boundary-sampling-probability": args.boundary_sampling_probability,
        "boundary-renderer-band-pixels": args.boundary_renderer_band_pixels,
        "boundary-renderer-sample-pixels": args.boundary_renderer_sample_pixels,
        "boundary-renderer-hard-width-pixels": args.boundary_renderer_hard_width_pixels,
        "boundary-renderer-soft-width-pixels": args.boundary_renderer_soft_width_pixels,
        "boundary-renderer-gate-gain": args.boundary_renderer_gate_gain,
        "boundary-renderer-far-sample-multiplier": args.boundary_renderer_far_sample_multiplier,
        "boundary-renderer-far-sample-weight": args.boundary_renderer_far_sample_weight,
        "boundary-gate-need-scale": args.boundary_gate_need_scale,
        "boundary-gate-exact-floor": args.boundary_gate_exact_floor,
        "boundary-sdf-zero-weight": args.boundary_sdf_zero_weight,
        "boundary-edge-sdf-consistency-weight": args.boundary_edge_sdf_consistency_weight,
        "boundary-pixel-regret-weight": args.boundary_pixel_regret_weight,
        "boundary-profile-weight": args.boundary_profile_weight,
        "boundary-regret-weight": args.boundary_regret_weight,
        "sdf-surface-weight": args.sdf_surface_weight,
        "sdf-sign-weight": args.sdf_sign_weight,
        "sdf-eikonal-weight": args.sdf_eikonal_weight,
        "sdf-gradient-alignment-weight": args.sdf_gradient_alignment_weight,
        "sdf-metric-gradient-weight": args.sdf_metric_gradient_weight,
        "sdf-metric-band-pixels": args.sdf_metric_band_pixels,
        "sdf-coarse-init-std": args.sdf_coarse_init_std,
        "sdf-synthetic-validation-tiles": args.sdf_synthetic_validation_tiles,
        "sdf-zero-band-pixels": args.sdf_zero_band_pixels,
        "sdf-bootstrap-residual-pixels": args.sdf_bootstrap_residual_pixels,
        "sdf-proof-residual-pixels": args.sdf_proof_residual_pixels,
        "sdf-proof-renderer-weight": args.sdf_proof_renderer_weight,
        "implicit-sdf-hidden-channels": args.implicit_sdf_hidden_channels,
        "implicit-sdf-residual-pixels": args.implicit_sdf_residual_pixels,
        "coarse-sdf-surface-weight": args.coarse_sdf_surface_weight,
        "sdf-residual-l1-weight": args.sdf_residual_l1_weight,
        "boundary-fuzz-weight": args.boundary_fuzz_weight,
        "boundary-halo-weight": args.boundary_halo_weight,
        "boundary-renderer-plateau-samples": args.boundary_renderer_plateau_samples,
        "boundary-renderer-plateau-max-multiplier": args.boundary_renderer_plateau_max_multiplier,
        "boundary-renderer-plateau-stability-scale": args.boundary_renderer_plateau_stability_scale,
        "seed": args.seed,
    }
    for key, value in forwarded.items():
        if value is not None and str(value) != "":
            train_cmd.extend(["--" + key, str(value)])
    if args.randomise_seed:
        train_cmd.append("--randomise-seed")
    if args.advanced_overrides:
        train_cmd.extend(["--advanced-overrides", args.advanced_overrides])
    # V9.8.3 uses staged hard gates. A new experiment is allocated first so
    # the exact resolved configuration can be used by the parameter-free
    # renderer preflight before any GPU training time is spent.
    requested_new = str(args.experiment).strip().upper() in {"", "NEW"}
    if requested_new:
        allocate_cmd = list(train_cmd) + ["--allocate-only"]
        _run(allocate_cmd, cwd=root)
        if not result_file.is_file():
            raise RuntimeError(f"allocation completed without result pointer: {result_file}")
        allocation = json.loads(result_file.read_text(encoding="utf-8"))
        experiment = str(allocation["experiment"])
        experiment_dir = root / "artifacts/nsamdr/experiments" / experiment
        train_cmd = _set_option(train_cmd, "--experiment", experiment)
        train_cmd = _set_option(train_cmd, "--control", "auto")
    else:
        experiment = str(args.experiment).strip().upper()
        experiment_dir = root / "artifacts/nsamdr/experiments" / experiment

    critic_checkpoint = root / "artifacts/nsamdr/geometry_critic/geometry_pair_critic.pt"

    existing_manifest = {}
    if not requested_new:
        manifest_path = experiment_dir / "experiment.json"
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.geometry_audit and requested_new:
        preflight_dir = experiment_dir / "previews/oracle_renderer_preflight"
        preflight_path = preflight_dir / "synthetic_geometry_audit.json"
        print(
            f"[tuning-stage] Stage A preflight: parameter-free oracle renderer for {experiment}...",
            flush=True,
        )
        preflight_cmd = [
            python, "-u", "tools/nsamdr/neural/audit_nsamdr_v9_geometry_checkpoint.py",
            "--repo-root", str(root),
            "--config", str(experiment_dir / "resolved_config.json"),
            "--oracle-only",
            "--output-dir", str(preflight_dir),
            "--device", args.preview_device,
            "--critic", "off",
            "--evidence-regions", "0",
        ]
        _run(preflight_cmd, cwd=root)
        if not preflight_path.is_file():
            raise RuntimeError(f"Stage-A preflight did not write {preflight_path}")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        renderer_pass = bool(preflight.get("rendererProof", {}).get("pass"))
        renderer_verdict = str(preflight.get("verdict") or "UNKNOWN")
        _update_manifest(
            experiment_dir,
            rendererPreflightVerdict=renderer_verdict,
            rendererPreflightPass=renderer_pass,
            reconstructionAcceptancePass=False,
        )
        if not renderer_pass:
            print("=" * 72, flush=True)
            print(f"TRAINING BLOCKED BY STAGE-A RENDERER PREFLIGHT: {renderer_verdict}", flush=True)
            print("V9.8.3 spends zero training epochs while exact-SDF rendering is below contract.", flush=True)
            print(f"Experiment               : {experiment}", flush=True)
            print(f"Renderer preflight       : {preflight_path}", flush=True)
            print("Reconstruction acceptance: REJECT", flush=True)
            print("=" * 72, flush=True)
            return 0

        # Geometry only: stop immediately after sdf-proof. Gate-proof cannot start
        # until the trained field itself passes Stage B.
        result_file.unlink(missing_ok=True)
        sdf_train_cmd = list(train_cmd) + ["--stop-after-phase", "sdf-proof"]
        _run(sdf_train_cmd, cwd=root)
        if not result_file.is_file():
            raise RuntimeError(f"SDF-stage training completed without result pointer: {result_file}")

        sdf_audit_dir = experiment_dir / "previews/sdf_stage_audit"
        sdf_audit_path = sdf_audit_dir / "synthetic_geometry_audit.json"
        print(f"[tuning-stage] Stage B proof: predicted SDF + forced exact gate for {experiment}...", flush=True)
        sdf_audit_cmd = [
            python, "-u", "tools/nsamdr/neural/audit_nsamdr_v9_geometry_checkpoint.py",
            "--repo-root", str(root),
            "--checkpoint", str(experiment_dir / "nsamdr_v9_fidelity.pt"),
            "--output-dir", str(sdf_audit_dir),
            "--device", args.preview_device,
            "--critic", "off",
            "--evidence-regions", "0",
        ]
        _run(sdf_audit_cmd, cwd=root)
        if not sdf_audit_path.is_file():
            raise RuntimeError(f"Stage-B audit did not write {sdf_audit_path}")
        sdf_gate = json.loads(sdf_audit_path.read_text(encoding="utf-8"))
        sdf_pass = bool(sdf_gate.get("sdfProof", {}).get("pass"))
        _update_manifest(
            experiment_dir,
            sdfStageProofVerdict=str(sdf_gate.get("verdict") or "UNKNOWN"),
            sdfStageProofPass=sdf_pass,
            reconstructionAcceptancePass=False,
        )
        if not sdf_pass:
            print("=" * 72, flush=True)
            print("GATE TRAINING BLOCKED BY STAGE-B SDF PROOF: SDF_FAIL", flush=True)
            print("V9.8.3 preserves the SDF checkpoint and does not teach the gate from harmful geometry.", flush=True)
            print(f"Experiment               : {experiment}", flush=True)
            print(f"SDF audit                : {sdf_audit_path}", flush=True)
            print("Reconstruction acceptance: REJECT", flush=True)
            print("=" * 72, flush=True)
            return 0

        # Stage B passed: resume the exact optimizer/RNG state into gate-proof and
        # later phases. No configuration is changed between stages.
        result_file.unlink(missing_ok=True)
        resume_cmd = _set_option(train_cmd, "--control", "resume")
        _run(resume_cmd, cwd=root)
    elif (
        args.geometry_audit
        and bool(existing_manifest.get("rendererPreflightPass"))
        and not bool(existing_manifest.get("sdfStageProofPass"))
    ):
        # V9.8.3 crash/retry recovery: if Stage A already passed but the
        # process died during the staged SDF section, preserve that experiment
        # and resume only through sdf-proof. This keeps the hard Stage-B gate
        # intact instead of accidentally continuing into gate-proof.
        print(
            f"[tuning-stage] Resuming staged SDF proof for existing {experiment}; "
            "Stage-A renderer preflight already passed.",
            flush=True,
        )
        result_file.unlink(missing_ok=True)
        sdf_train_cmd = _set_option(list(train_cmd), "--control", "auto")
        sdf_train_cmd += ["--stop-after-phase", "sdf-proof"]
        _run(sdf_train_cmd, cwd=root)
        if not result_file.is_file():
            raise RuntimeError(f"SDF-stage resume completed without result pointer: {result_file}")

        sdf_audit_dir = experiment_dir / "previews/sdf_stage_audit"
        sdf_audit_path = sdf_audit_dir / "synthetic_geometry_audit.json"
        print(f"[tuning-stage] Stage B proof: predicted SDF + forced exact gate for {experiment}...", flush=True)
        sdf_audit_cmd = [
            python, "-u", "tools/nsamdr/neural/audit_nsamdr_v9_geometry_checkpoint.py",
            "--repo-root", str(root),
            "--checkpoint", str(experiment_dir / "nsamdr_v9_fidelity.pt"),
            "--output-dir", str(sdf_audit_dir),
            "--device", args.preview_device,
            "--critic", "off",
            "--evidence-regions", "0",
        ]
        _run(sdf_audit_cmd, cwd=root)
        if not sdf_audit_path.is_file():
            raise RuntimeError(f"Stage-B audit did not write {sdf_audit_path}")
        sdf_gate = json.loads(sdf_audit_path.read_text(encoding="utf-8"))
        sdf_pass = bool(sdf_gate.get("sdfProof", {}).get("pass"))
        _update_manifest(
            experiment_dir,
            sdfStageProofVerdict=str(sdf_gate.get("verdict") or "UNKNOWN"),
            sdfStageProofPass=sdf_pass,
            reconstructionAcceptancePass=False,
        )
        if not sdf_pass:
            print("=" * 72, flush=True)
            print("GATE TRAINING BLOCKED BY STAGE-B SDF PROOF: SDF_FAIL", flush=True)
            print("V9.8.3 preserves the SDF checkpoint and does not teach the gate from harmful geometry.", flush=True)
            print(f"Experiment               : {experiment}", flush=True)
            print(f"SDF audit                : {sdf_audit_path}", flush=True)
            print("Reconstruction acceptance: REJECT", flush=True)
            print("=" * 72, flush=True)
            return 0

        result_file.unlink(missing_ok=True)
        resume_cmd = _set_option(list(train_cmd), "--control", "resume")
        _run(resume_cmd, cwd=root)
    else:
        # Existing experiments that have already passed their staged gates retain
        # normal resume/restart semantics.
        _run(train_cmd, cwd=root)

    if not result_file.is_file():
        raise RuntimeError(f"training completed without result pointer: {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    experiment = str(result["experiment"])
    experiment_dir = root / "artifacts/nsamdr/experiments" / experiment
    synthetic_audit_path = experiment_dir / "previews/synthetic_geometry_audit/synthetic_geometry_audit.json"

    if args.geometry_audit:
        print(f"[tuning-stage] Final Stage A/B/C synthetic proof for {experiment}...", flush=True)
        audit_cmd = [
            python, "-u", "tools/nsamdr/neural/audit_nsamdr_v9_geometry_checkpoint.py",
            "--repo-root", str(root),
            "--checkpoint", str(experiment_dir / "nsamdr_v9_fidelity.pt"),
            "--output-dir", str(experiment_dir / "previews/synthetic_geometry_audit"),
            "--device", args.preview_device,
            "--critic", args.geometry_critic,
            "--critic-checkpoint", str(critic_checkpoint),
            "--evidence-regions", str(max(1, args.geometry_evidence_regions // 2)),
            "--critic-steps", str(args.critic_steps),
        ]
        _run(audit_cmd, cwd=root)
        if synthetic_audit_path.is_file():
            synthetic_gate = json.loads(synthetic_audit_path.read_text(encoding="utf-8"))
            proof_verdict = str(synthetic_gate.get("verdict") or "UNKNOWN")
            _update_manifest(
                experiment_dir,
                syntheticGeometryProofVerdict=proof_verdict,
                reconstructionAcceptancePass=False,
            )
            if proof_verdict != "PASS":
                print("=" * 72, flush=True)
                print(f"RAVEN PREVIEW BLOCKED BY SYNTHETIC PROOF: {proof_verdict}", flush=True)
                print("V9.8.3 does not spend a real-asset preview on failed renderer/SDF/gate geometry.", flush=True)
                print(f"Experiment               : {experiment}", flush=True)
                print(f"Synthetic audit          : {synthetic_audit_path}", flush=True)
                print("Reconstruction acceptance: REJECT", flush=True)
                print("=" * 72, flush=True)
                return 0
    print(f"[tuning-stage] Launching renderer candidate audit + preview for {experiment}...", flush=True)
    preview_cmd = [
        python, "-u", "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
        "--repo-root", str(root), "--experiment", experiment,
        "--shared-cache", args.shared_cache,
        "--target-size", str(args.preview_target_size),
        "--device", args.preview_device,
        "--geometry-critic", args.geometry_critic,
        "--geometry-audit-policy", args.geometry_audit_policy,
        "--geometry-evidence-regions", str(args.geometry_evidence_regions),
        "--critic-checkpoint", str(critic_checkpoint),
    ]
    if not args.geometry_audit:
        preview_cmd.append("--no-geometry-audit")
    if args.force_candidate:
        preview_cmd.append("--force-candidate")
    _run(preview_cmd, cwd=root)

    print("=" * 72, flush=True)
    print("RAVEN TUNING STAGE COMPLETE", flush=True)
    print(f"Experiment               : {experiment}", flush=True)
    print(f"Training mode            : {args.training_mode}", flush=True)
    print("Renderer preview         : completed", flush=True)
    if args.geometry_audit and synthetic_audit_path.is_file():
        synthetic = json.loads(synthetic_audit_path.read_text(encoding="utf-8"))
        renderer = synthetic.get("rendererProof", {}) if isinstance(synthetic, dict) else {}
        sdf_proof = synthetic.get("sdfProof", {}) if isinstance(synthetic, dict) else {}
        gate_proof = synthetic.get("gateProof", {}) if isinstance(synthetic, dict) else {}
        print(
            f"Staged geometry proof    : {synthetic.get('verdict')} | "
            f"A={float(renderer.get('chamferImprovementMean',0.0)):+.2%} "
            f"B={float(sdf_proof.get('chamferImprovementMean',0.0)):+.2%} "
            f"C={float(gate_proof.get('chamferImprovementMean',0.0)):+.2%}",
            flush=True,
        )
    feedback_bundle = experiment_dir / "previews" / f"{experiment}_geometry_feedback.zip"
    if feedback_bundle.is_file():
        print(f"Feedback bundle          : {feedback_bundle}", flush=True)
    from v9.experiments import load_experiment_manifest, load_resolved_config
    manifest = load_experiment_manifest(root, experiment)
    resolved = load_resolved_config(root, experiment)
    if not bool(resolved.appearance_enabled):
        print("Promotion                : LOCKED (V9.4 geometry-only proof)", flush=True)
        print("Next                     : compare deterministic baseline vs GeometryNet warp", flush=True)
    elif args.training_mode == "quick":
        print("Promotion                : LOCKED (Quick experiment)", flush=True)
        print("Next                     : compare/tune again, or run Full / promotion proof", flush=True)
    elif bool(manifest.get("acceptancePass")):
        print("Promotion                : ENABLED (quantitative gate passed)", flush=True)
        print("Next                     : promote this exact experiment configuration", flush=True)
    else:
        print("Promotion                : LOCKED (quantitative acceptance failed)", flush=True)
        print("Next                     : inspect preview/metrics and tune again", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
