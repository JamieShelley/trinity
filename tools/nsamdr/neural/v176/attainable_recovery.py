"""Target-assisted, training-only probes of the existing scalar correction path.

The free field is an offline diagnostic, never an inference input or a model
checkpoint. A feasible fit demonstrates attainability on that region. Failure
of this finite-budget local solver does NOT establish a representation ceiling.
"""
from __future__ import annotations

import torch

from .physical_composition import compose_albedo_detail, gaussian_highpass
from .stability import STABILITY_POLICY, NumericalStabilityError, require_tensor
from .wavelet_supervision import detail_supervision_loss, stationary_haar_details


ATTAINABLE_POLICY = {
    "revision": "V176_ATTAINABLE_RECOVERY_V1",
    "targetAssisted": True,
    "deployable": False,
    "precision": "fp32",
    "objectives": ["edge", "stage10a"],
    "stages": [0, 128, 512],
    "optimizer": "Adam on one free raw scalar per HR pixel",
    "learningRate": 0.03,
    "adamBetas": [0.0, 0.999],
    "maxFeasibilityBacktracks": 24,
    "maxObjectiveBacktracks": 24,
    "stepSelection": "non-increasing objective; backtrack parameter steps without clipping outputs",
    "selection": "lowest objective, including the zero-field control",
    "stage10aInitialization": "better of zero and the target-assisted edge fit under the unchanged Stage10A loss",
    "globalOptimalityProven": False,
    "physicalCapsChanged": False,
    "networkWeightsUpdated": False,
    "rgbRelaxation": "independent channels, high-pass constraint removed",
    "rgbUpperBoundAppliesTo": ["global_recovery", "edge_recovery"],
}


def scalar_candidate(original: dict, raw: torch.Tensor, cap: float) -> dict:
    result = compose_albedo_detail(
        original, gaussian_highpass(raw, sigma=1.25, radius=4),
        albedo_residual_cap=cap,
    )
    result["fine_detail_raw"] = raw
    return result


def rgb_projection_relaxation(original: dict, target: torch.Tensor, cap: float) -> torch.Tensor:
    """Pointwise L1 optimum after relaxing shared-scalar and spatial constraints.

    Every feasible scalar correction is in [-highpassMaxAbs, highpassMaxAbs].
    Monotonic tanh + clamp bounds each channel's attainable interval. Projecting
    the target into independent intervals minimizes both RGB L1 and the fixed
    target-weighted RGB L1 used by the edge recovery metric. This gives upper
    bounds for THOSE TWO metrics only, not for lattice, gradient or detail.
    """
    raw = original["candidate_raw_residual_albedo"].detach().float()
    baseline = original["baseline_albedo"].detach().float()
    target = target.detach().float()
    if target.shape != raw.shape or baseline.shape != raw.shape:
        raise ValueError("projection relaxation requires aligned RGB tensors")
    if not 0 < cap <= 1:
        raise ValueError("albedo residual cap must be within (0,1]")
    require_tensor("attainable/target", target, 1.0)
    limit = STABILITY_POLICY["highpassMaxAbs"]
    lo = (baseline + cap * torch.tanh(raw - limit)).clamp(0, 1)
    hi = (baseline + cap * torch.tanh(raw + limit)).clamp(0, 1)
    return target.clamp(min=lo, max=hi)


def edge_objective(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same target-weighted RGB L1 numerator as sample_metrics.edge_recovery."""
    gray = target.float().mean(dim=1, keepdim=True)
    dx = torch.nn.functional.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    gradient = dx + dy
    weight = 1 + 4 * gradient / gradient.mean().clamp_min(1e-6)
    return ((candidate.float() - target.float()).abs().mean(dim=1, keepdim=True) * weight).mean()


def band_relative_l1(outputs: dict, batch: dict) -> dict[str, float]:
    baseline = outputs["baseline_albedo"].float().mean(dim=1, keepdim=True)
    target = batch["target_albedo"].float().mean(dim=1, keepdim=True)
    candidate = outputs["candidate_albedo"].float().mean(dim=1, keepdim=True)
    reference = stationary_haar_details(target - baseline)
    predicted = stationary_haar_details(candidate - baseline)
    return {name: float((predicted[name] - value).abs().mean() /
                        value.abs().mean().clamp_min(STABILITY_POLICY["denominatorFloor"]))
            for name, value in reference.items()}


def _objective(outputs: dict, batch: dict, objective: str) -> torch.Tensor:
    if objective == "stage10a":
        return detail_supervision_loss(outputs, batch)["total"]
    return edge_objective(outputs["candidate_albedo"], batch["target_albedo"])


def _accept_feasible_step(raw: torch.Tensor, before: torch.Tensor) -> int:
    """Backtrack the parameter step, without clipping the composed correction.

    The raw/HP magnitude limits are convex constraints on this free field.
    Interpolation toward the previous feasible parameter therefore gives a
    feasible update. Nonfinite proposals are errors, not recoverable steps.
    """
    require_tensor("attainable/proposedRaw", raw)
    proposal = raw.detach().clone()
    for count in range(ATTAINABLE_POLICY["maxFeasibilityBacktracks"] + 1):
        try:
            gaussian_highpass(raw)
            return count
        except NumericalStabilityError:
            if count == ATTAINABLE_POLICY["maxFeasibilityBacktracks"]:
                raise
            with torch.no_grad():
                raw.copy_(torch.lerp(before, proposal, 0.5 ** (count + 1)))
    raise AssertionError("unreachable")


def fit_scalar_field(original: dict, batch: dict, cap: float, objective: str,
                     *, initial_raw: torch.Tensor | None = None,
                     stages: tuple[int, ...] | None = None, progress=None) -> tuple[dict, dict]:
    """Fit only a disposable scalar image; all input model tensors are detached."""
    if objective not in ATTAINABLE_POLICY["objectives"]:
        raise ValueError("unknown attainable-recovery objective")
    stages = tuple(ATTAINABLE_POLICY["stages"] if stages is None else stages)
    if len(stages) < 2 or stages[0] != 0 or any(a >= b for a, b in zip(stages, stages[1:])):
        raise ValueError("attainable stages must increase from zero")
    frozen = {k: v.detach().float() for k, v in original.items() if isinstance(v, torch.Tensor)}
    target_batch = {k: v.detach() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    require_tensor("attainable/target", target_batch["target_albedo"], 1.0)
    raw = torch.nn.Parameter(torch.zeros_like(frozen["baseline_albedo"][:, :1]))
    with torch.no_grad():
        zero_objective = float(_objective(scalar_candidate(frozen, raw, cap), target_batch, objective))
        initialization = "zero"
        seed_objective = None
        if initial_raw is not None:
            seed = initial_raw.detach().float()
            if seed.shape != raw.shape or seed.device != raw.device:
                raise ValueError("initial scalar field must match the probe shape and device")
            seed_objective = float(_objective(scalar_candidate(frozen, seed, cap), target_batch, objective))
            if seed_objective < zero_objective:
                raw.copy_(seed)
                initialization = "target-assisted edge fit"
    optimizer = torch.optim.Adam([raw], lr=ATTAINABLE_POLICY["learningRate"],
                                 betas=tuple(ATTAINABLE_POLICY["adamBetas"]))
    best_raw = raw.detach().clone()
    best_value = float("inf")
    best_update = 0
    history = []
    backtracks = 0
    objective_backtracks = 0
    stalled_updates = 0
    peak_gradient = 0.0
    # Disable mixed precision for the complete target-assisted fit.
    with torch.autocast(device_type=raw.device.type, enabled=False):
        for update in range(stages[-1] + 1):
            optimizer.zero_grad(set_to_none=True)
            outputs = scalar_candidate(frozen, raw, cap)
            loss = _objective(outputs, target_batch, objective)
            require_tensor("attainable/objective", loss, STABILITY_POLICY["lossMaxAbs"])
            value = float(loss.detach())
            if value < best_value:
                best_value, best_update = value, update
                best_raw = raw.detach().clone()
            if update in stages:
                row = {"update": update, "objective": value,
                       "bestObjective": best_value, "bestUpdate": best_update,
                       "feasibilityBacktracks": backtracks,
                       "objectiveBacktracks": objective_backtracks}
                history.append(row)
                if progress is not None:
                    progress(row)
            if update == stages[-1]:
                break
            loss.backward()
            require_tensor("attainable/rawGradient", raw.grad)
            gradient = float(raw.grad.detach().double().norm())
            require_tensor("attainable/gradientL2", torch.tensor(gradient),
                           STABILITY_POLICY["gradientMaxL2"])
            peak_gradient = max(peak_gradient, gradient)
            torch.nn.utils.clip_grad_norm_([raw], STABILITY_POLICY["gradientClipL2"],
                                          error_if_nonfinite=True)
            before = raw.detach().clone()
            optimizer.step()
            feasible_backtracks = _accept_feasible_step(raw, before)
            backtracks += feasible_backtracks
            proposal = raw.detach().clone()
            # beta1=0 makes the Adam direction a diagonally scaled negative
            # gradient. A line search prevents the normalized Haar L1 terms
            # from making a fixed per-pixel Adam step oscillate past the fit.
            accepted = False
            with torch.no_grad():
                for count in range(ATTAINABLE_POLICY["maxObjectiveBacktracks"] + 1):
                    trial = _objective(scalar_candidate(frozen, raw, cap), target_batch, objective)
                    require_tensor("attainable/trialObjective", trial)
                    if float(trial) <= value:
                        accepted = True
                        break
                    raw.copy_(torch.lerp(before, proposal, 0.5 ** (count + 1)))
                objective_backtracks += count
                if not accepted:
                    raw.copy_(before)
                    stalled_updates += 1
            optimizer.param_groups[0]["lr"] = min(
                ATTAINABLE_POLICY["learningRate"],
                optimizer.param_groups[0]["lr"] * 1.5 * 0.5 ** (count + feasible_backtracks),
            )
            for key, value in optimizer.state[raw].items():
                if isinstance(value, torch.Tensor) and key != "step":
                    require_tensor(f"attainable/optimizer/{key}", value,
                                   STABILITY_POLICY["optimizerMomentMaxAbs"])
        with torch.no_grad():
            best_outputs = scalar_candidate(frozen, best_raw, cap)
    return best_outputs, {
        "objective": objective, "updates": stages[-1], "bestUpdate": best_update,
        "initialization": initialization, "zeroFieldObjective": zero_objective,
        "seedObjective": seed_objective,
        "bestObjective": best_value, "lastObjective": history[-1]["objective"],
        "history": history, "peakGradientL2": peak_gradient,
        "feasibilityBacktracks": backtracks,
        "objectiveBacktracks": objective_backtracks, "stalledUpdates": stalled_updates,
        "globalOptimalityProven": False,
    }
