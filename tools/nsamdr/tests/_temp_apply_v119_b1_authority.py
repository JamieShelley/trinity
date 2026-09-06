from pathlib import Path
ROOT=Path.cwd()
def r(path,old,new):
 p=ROOT/path; s=p.read_text(encoding='utf-8'); n=s.count(old)
 if n!=1: raise RuntimeError(f'{path}: anchor count {n}: {old[:80]!r}')
 p.write_text(s.replace(old,new,1),encoding='utf-8')
L='tools/nsamdr/neural/v9/local_boundary_production_contract.py'
E='tools/nsamdr/neural/v9/edge_constrained_spline_graph.py'
S='tools/nsamdr/neural/v9/application/baseline_relative_smoke.py'
P='tools/nsamdr/neural/v9/application/pipeline.py'
T='tools/nsamdr/tests/test_v117_baseline_relative_contract.py'
D='tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md'
old='NSAMDR_RAVEN_PRODUCTION_BASELINE_RESIDUAL_SPLINE_GRAPH_4X_V11_8_0'
new='NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_RESIDUAL_SPLINE_GRAPH_4X_V11_9_0'
r(L,old,new); r(E,old,new)
r(L,'''        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
''','''        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(False)
        # B1a proves topology only: C must remain exactly B until B1b.
        nn.init.zeros_(self.structural_residual_gain_head[-1].weight)
        nn.init.zeros_(self.structural_residual_gain_head[-1].bias)
        for parameter in self.structural_residual_gain_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
''')
r(L,'''        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(True)
        for parameter in head.geometry_net.parameters():
''','''        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(True)
        # B1b is the first phase allowed to earn structural authority over B.
        for parameter in self.structural_residual_gain_head.parameters():
            parameter.requires_grad_(True)
        for parameter in head.geometry_net.parameters():
''')
r(S,'Evaluate held-out real-Raven B1a validation against deterministic baseline B.','Evaluate held-out real-Raven structural stages against deterministic baseline B.')
r(S,'BaselineRelativeSmokeService.passed(), PassDrivenPipeline._run_quick_b1a_smoke().','BaselineRelativeSmokeService.safe_to_refine(), BaselineRelativeSmokeService.passed(),\n            PassDrivenPipeline._run_quick_b1a_smoke(), PassDrivenPipeline._run_quick_b1b_smoke().')
sp=(ROOT/S).read_text(encoding='utf-8'); marker='    def passed(self, validation: dict[str, Any], config: V9Config) -> bool:\n'; i=sp.index(marker)
sp=sp[:i]+'''    def safe_to_refine(self, validation: dict[str, Any], config: V9Config) -> bool:
        """Allow B1b only when topology-only B1a preserves baseline safety."""
        metrics = self.metrics(validation)
        finite = all(math.isfinite(value) for value in metrics.values())
        tolerance = max(1.0e-6, abs(metrics["baselineMae"]) * 1.0e-5)
        return bool(
            finite
            and metrics["candidateMae"] <= metrics["baselineMae"] + tolerance
            and metrics["regressionFraction"]
            <= float(config.maximum_validation_regression_fraction)
        )

    def passed(self, validation: dict[str, Any], config: V9Config) -> bool:
        """Require B1b C to beat B without exceeding the regression budget."""
        metrics = self.metrics(validation)
        finite = all(math.isfinite(value) for value in metrics.values())
        return bool(
            finite
            and metrics["candidateMae"] < metrics["baselineMae"]
            and metrics["relativeGain"] > 0.0
            and metrics["regressionFraction"]
            <= float(config.maximum_validation_regression_fraction)
        )
'''
(ROOT/S).write_text(sp,encoding='utf-8')
r(P,'Run/reuse one Quick B1a epoch and reject before B1b when C loses to B.','Run/reuse one Quick B1a epoch and require identity-safe real-Raven output.')
r(P,'''            Turn the live A/B/C comparison into an executable fail-fast contract on
            held-out authored Raven data, independently of the synthetic topology gate.''','''            Keep topology-only B1a fail-closed without demanding improvement before
            continuous geometry and residual authority become trainable in B1b.''')
r(P,'baseline_safe = self.baseline_smoke.passed(validation, context.config)','baseline_safe = self.baseline_smoke.safe_to_refine(validation, context.config)')
r(P,'"[quick-smoke] real Raven B1a: "','"[quick-smoke] real Raven B1a identity safety: "')
r(P,'''        if baseline_safe:
            print(
                "[quick-smoke] PASS: C beats deterministic baseline B on held-out Raven; "
                "synthetic topology remains independently fail-closed.",
                flush=True,
            )''','''        if topology_safe and baseline_safe:
            print(
                "[quick-smoke] PASS B1a: C preserved deterministic baseline B within "
                "the real-Raven safety budget; B1b may now earn positive authority.",
                flush=True,
            )''')
r(P,'''        print(
            "[quick-smoke] REJECTED before B1b: B1a did not satisfy the real-Raven "
            "baseline-relative contract. C must beat B and held-out regressions must "
            "remain within the configured safety limit.",
            flush=True,
        )
        code = self.experiments.reject(
            context,
            phase="sdf-bootstrap-baseline-relative-smoke",
            gate_label="real Raven baseline-relative smoke: C must beat B",
            metadata=rejected,
        )''','''        print(
            "[quick-smoke] REJECTED before B1b: topology-only B1a did not preserve "
            "baseline safety or topology did not bootstrap.", flush=True,
        )
        code = self.experiments.reject(
            context,
            phase="sdf-bootstrap-baseline-relative-safety",
            gate_label="real Raven B1a identity safety + topology bootstrap",
            metadata=rejected,
        )''')
helper='''
    def _run_quick_b1b_smoke(
        self, context: ExperimentContext, latest: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Require bounded Quick B1b refinement to produce strict real C > B."""
        validation = self.state.latest_phase_validation(
            context.directory, context.config, phase="sdf-proof"
        )
        snapshot = self.state.snapshot(context.directory, context.config)
        metrics = self.baseline_smoke.metrics(validation)
        passed = self.baseline_smoke.passed(validation, context.config)
        print(
            "[quick-smoke] real Raven B1b strict improvement: "
            f"B={metrics['baselineMae']:.6f} C={metrics['candidateMae']:.6f} "
            f"gain={metrics['relativeGain']:+.2%} "
            f"wins={metrics['improvementFraction']:.1%} "
            f"regress={metrics['regressionFraction']:.1%}/"
            f"{float(context.config.maximum_validation_regression_fraction):.1%}",
            flush=True,
        )
        if passed:
            print(
                "[quick-smoke] PASS B1b: C now beats deterministic baseline B on "
                "held-out Raven; normal structural qualification remains fail-closed.",
                flush=True,
            )
            return latest, 0
        rejected = dict(latest or snapshot)
        rejected["baselineRelativeB1bSmokeValidation"] = validation
        rejected["baselineRelativeB1bSmokeMetrics"] = metrics
        print(
            "[quick-smoke] REJECTED after B1b: refinement did not earn positive "
            "real-Raven improvement over B within the safety budget.", flush=True,
        )
        code = self.experiments.reject(
            context,
            phase="sdf-proof-baseline-relative-smoke",
            gate_label="real Raven B1b smoke: C must beat B",
            metadata=rejected,
        )
        return rejected, code

'''
r(P,'    def _run_stage(\n',helper+'    def _run_stage(\n')
r(P,'''                latest = self._invoke(
                    context,
                    resume=True,
                    stop_after_phase="sdf-proof",
                )
            else:
''','''                latest = self._invoke(
                    context,
                    resume=True,
                    stop_after_phase="sdf-proof",
                )
                latest, b1b_smoke_code = self._run_quick_b1b_smoke(context, latest)
                if b1b_smoke_code != 0:
                    return latest, current_resume, b1b_smoke_code
            else:
''')
r(D,'## Literature corrections carried into V11.7','''## V11.9 B1 authority contract

B1a is topology-only. Its structural residual-gain head is reset to exact zero and frozen, so **C == B by construction throughout B1a**. B1a may advance only when topology bootstraps and held-out Raven remains inside the configured regression safety budget. B1b then freezes topology, unlocks continuous spline geometry and the residual-gain head, and receives the first opportunity to earn structural authority. The bounded Quick B1b smoke must show strict **C > B** on held-out Raven before normal structural qualification is considered. Equality is safe for B1a but is not a B1b success.

## Literature corrections carried into V11.7''')
r(D,'The first Quick B1b epoch is a bounded 14-batch authored-Raven structural refinement smoke pass.','Quick first runs one B1a topology epoch with structural residual authority frozen at exact identity; B1a is checked for topology and non-regression safety, not strict improvement. The first Quick B1b epoch is a bounded 14-batch authored-Raven structural refinement smoke pass and must show strict C > B.')
tp=ROOT/T; ts=tp.read_text(encoding='utf-8').replace(old,new)
add='''


def test_v119_b1a_freezes_gain_and_b1b_unlocks_it():
    local = text("tools/nsamdr/neural/v9/local_boundary_production_contract.py")
    a = local.index("    def unlock_topology_for_bootstrap")
    b = local.index("    def lock_topology_for_proof")
    c = local.index("    def restore_locked_topology_parameters")
    b1a, b1b = local[a:b], local[b:c]
    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].weight)" in b1a
    assert "self.structural_residual_gain_head.parameters()" in b1a
    assert "parameter.requires_grad_(False)" in b1a
    assert "self.structural_residual_gain_head.parameters()" in b1b
    assert "parameter.requires_grad_(True)" in b1b


def test_v119_quick_moves_strict_baseline_win_to_b1b():
    smoke = text("tools/nsamdr/neural/v9/application/baseline_relative_smoke.py")
    pipeline = text("tools/nsamdr/neural/v9/application/pipeline.py")
    assert "def safe_to_refine(" in smoke
    assert 'metrics["candidateMae"] <= metrics["baselineMae"] + tolerance' in smoke
    assert 'metrics["candidateMae"] < metrics["baselineMae"]' in smoke
    assert "def _run_quick_b1b_smoke(" in pipeline
    assert "PASS B1a: C preserved deterministic baseline B" in pipeline
    assert "PASS B1b: C now beats deterministic baseline B" in pipeline
    assert 'phase="sdf-proof-baseline-relative-smoke"' in pipeline
'''
if 'test_v119_b1a_freezes_gain_and_b1b_unlocks_it' in ts: raise RuntimeError('tests present')
tp.write_text(ts.rstrip()+add+'\n',encoding='utf-8')
print('Applied V11.9 B1 authority split')
