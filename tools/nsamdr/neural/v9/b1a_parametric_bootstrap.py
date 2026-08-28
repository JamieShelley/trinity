"""Production B1a classifier-first parametric bootstrap.

EXP_0001 (2026-08-26) after the class-balanced/direct-loss refactor proved that
B1a is now using the intended direct teachers, but the two-epoch bootstrap still
collapses its seven-way classifier mostly onto ``junction``.

Observed held-out 29-case audit after epoch 2:
    primitive class accuracy             20.69%
    confusion: 15/15 line cases -> junction
    ring -> ellipse
    corners/parallel -> junction
    primitive parameter MAE              0.0454
    source contour Chamfer               2.404 px
    predicted contour Chamfer           45.713 px
    zero-contour win fraction             0.0%

The parameter branch is already much closer than the class decision.  Because
the renderer-facing family is chosen by argmax, a wrong class dominates the
geometry error no matter how reasonable the per-class parameter hypotheses are.

B1a therefore needs to be classifier-first.  The classifier and regressor are
already separate production branches.  This contract keeps the deterministic,
class-balanced complete-teacher B1a dataset and changes only the B1a loss
balance so the classifier owns the clipped gradient until the representation
stops collapsing.  Parameter/render teachers remain active at lower authority.

No model, head, inference path, checkpoint schema, gate, renderer or Raven-only
architecture is added.
"""
from __future__ import annotations

import random
from typing import Any, Callable

from . import dataset as _dataset
from . import losses as _losses
from .parametric_primitives import PRIMITIVE_COUNT

_ORIGINAL_COMPUTE_LOSSES: Callable[..., dict[str, Any]] | None = None
_ORIGINAL_PHYSICAL_GETITEM: Callable[..., dict[str, Any]] | None = None


def _is_b1a_training_dataset(instance: Any) -> bool:
    return (
        getattr(instance, "split", None) == "train"
        and float(
            getattr(
                getattr(instance, "config", None),
                "synthetic_geometry_probability",
                0.0,
            )
        ) > 0.0
    )


def _balanced_b1a_getitem(instance: Any, index: int) -> dict[str, Any]:
    """Return one deterministic complete-teacher primitive, balanced by class."""
    if not _is_b1a_training_dataset(instance):
        assert _ORIGINAL_PHYSICAL_GETITEM is not None
        return _ORIGINAL_PHYSICAL_GETITEM(instance, index)

    rng = random.Random(int(instance.seed) + int(index) * 1_000_003)
    target_size = int(instance.config.tile_size) * int(instance.config.target_scale)
    forced_class = int(index) % int(PRIMITIVE_COUNT)
    (
        albedo_hr,
        normal_hr,
        material_hr,
        sdf,
        orientation,
        edge,
        primitive_target,
    ) = _dataset._synthetic_parametric_geometry_sample(
        target_size,
        instance.config,
        rng,
        forced_class=forced_class,
    )
    return _dataset._pack_sample(
        albedo_hr,
        normal_hr,
        material_hr,
        1.0,
        sdf,
        orientation,
        edge,
        instance.config,
        rng,
        geometry_exact=1.0,
        primitive_target=primitive_target,
    )


def _compute_losses_b1a(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    config: Any,
    phase: str,
) -> dict[str, Any]:
    assert _ORIGINAL_COMPUTE_LOSSES is not None
    result = _ORIGINAL_COMPUTE_LOSSES(outputs, batch, config, phase)
    if phase != "sdf-bootstrap":
        return result

    # Classifier-first B1a.  The existing config values remain authoritative
    # everywhere else.  The factors below are B1a curriculum weights, shared by
    # Quick and Full Training:
    #
    #   class  : 8.0 * 8.0 = 64.0
    #   param  : 48.0 * 2/3 = 32.0
    #   render : 4.0 * 0.5 = 2.0
    #
    # With EXP_0001's losses this makes the class gradient the largest branch
    # before the global 1.25 clip, instead of allowing wrong-class render error
    # to consume most of the clipped update.
    class_weight = float(
        getattr(config, "parametric_primitive_class_weight", 8.0)
    ) * 8.0
    param_weight = float(
        getattr(config, "parametric_primitive_param_weight", 48.0)
    ) * (2.0 / 3.0)
    render_weight = float(
        getattr(config, "parametric_primitive_render_weight", 4.0)
    ) * 0.5

    primitive_supervision = (
        result["primitive_class"] * class_weight
        + result["primitive_param"] * param_weight
        + result["primitive_render"] * render_weight
    )

    # Auxiliary geometry heads remain trainable, but deliberately low authority
    # while the discrete manufactured family is being established.
    auxiliary_supervision = (
        result["edge"] * 0.25
        + result["orientation"] * 0.10
        + result["hardness"] * 0.10
    )

    result["total"] = (primitive_supervision + auxiliary_supervision).float()
    result["b1a_parametric_supervision"] = primitive_supervision.detach()
    result["b1a_auxiliary_supervision"] = auxiliary_supervision.detach()
    return result


def install_b1a_parametric_bootstrap() -> None:
    """Install the production B1a classifier-first data/loss contract once."""
    global _ORIGINAL_COMPUTE_LOSSES, _ORIGINAL_PHYSICAL_GETITEM

    current_loss = _losses.compute_losses
    if not bool(getattr(current_loss, "_nsamdr_b1a_classifier_first", False)):
        _ORIGINAL_COMPUTE_LOSSES = current_loss
        _compute_losses_b1a._nsamdr_b1a_classifier_first = True  # type: ignore[attr-defined]
        _losses.compute_losses = _compute_losses_b1a

    current_getitem = _dataset.PhysicalTileDatasetV9.__getitem__
    if not bool(getattr(current_getitem, "_nsamdr_b1a_balanced_dataset", False)):
        _ORIGINAL_PHYSICAL_GETITEM = current_getitem
        _balanced_b1a_getitem._nsamdr_b1a_balanced_dataset = True  # type: ignore[attr-defined]
        _dataset.PhysicalTileDatasetV9.__getitem__ = _balanced_b1a_getitem
