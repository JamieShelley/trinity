# NSAMDR baseline-relative reconstruction contract

## Non-negotiable comparison

Every learned stage is judged against the deterministic reconstruction available from the same degraded LR evidence.

- **A — Authored source:** held-out HR target / real EVE authored texture.
- **B — Deterministic 4x baseline:** bicubic albedo, normalized bilinear normal XY, nearest physical material channels.
- **C — Current learned stage:** the stage actually being trained, before downstream selectors can hide it.

Training is useful only when C improves on B while moving toward A. Production-final remains a separate fail-closed authority.

## Literature corrections carried into V11.7

1. Residual SR systems (VDSR, LapSRN, SwinIR) preserve a low-frequency/interpolated path and learn the missing correction rather than forcing the network to repaint the full image. NSAMDR already had an internal baseline, but its proof/preview did not expose it as a first-class control.
2. Deep Vectorization of Technical Drawings uses neural estimates as an initializer and then refines explicit geometric parameters. This remains the next structural escalation if the connected-spline learned proposal cannot beat B reliably.
3. End-to-End Line Drawing Vectorization supports hard ordered connectivity: connectivity should be represented, not merely penalized.
4. DiffVG supplies differentiable anti-aliased rasterization but does not solve discrete topology changes; topology remains an explicit NSAMDR responsibility.
5. LIVE reinforces that low raster error alone is not a topology guarantee.
6. Long smoothing B-splines support smoothing the parameterized curve itself, with corners/junctions exempted structurally rather than blurring output pixels.

## Quick feedback contract

The first Quick B1b epoch is a two-examples-per-primitive smoke pass. It is not a promotion proof. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete training bank and all existing hard qualification gates.
