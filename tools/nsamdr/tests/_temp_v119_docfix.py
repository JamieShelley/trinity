from pathlib import Path
ROOT=Path.cwd()
def r(path, old, new):
    p=ROOT/path; s=p.read_text(encoding='utf-8'); n=s.count(old)
    if n!=1: raise RuntimeError(f'{path}: doc anchor count {n}')
    p.write_text(s.replace(old,new,1),encoding='utf-8')
S='tools/nsamdr/neural/v9/application/baseline_relative_smoke.py'
P='tools/nsamdr/neural/v9/application/pipeline.py'
T='tools/nsamdr/tests/test_v117_baseline_relative_contract.py'
r(S,'        """Allow B1b only when topology-only B1a preserves baseline safety."""','''        """Allow B1b only when topology-only B1a preserves baseline safety.

        Purpose:
            Accept intentional C == B identity after B1a while rejecting unsafe
            held-out Raven regressions before continuous refinement.
        Called by:
            PassDrivenPipeline._run_quick_b1a_smoke().
        Calls:
            BaselineRelativeSmokeService.metrics().
        """''')
r(S,'        """Require B1b C to beat B without exceeding the regression budget."""','''        """Require B1b C to beat B without exceeding the regression budget.

        Purpose:
            Enforce strict positive baseline-relative improvement once B1b can
            train continuous geometry and structural residual authority.
        Called by:
            PassDrivenPipeline._run_quick_b1b_smoke().
        Calls:
            BaselineRelativeSmokeService.metrics().
        """''')
r(P,'        """Require bounded Quick B1b refinement to produce strict real C > B."""','''        """Require bounded Quick B1b refinement to produce strict real C > B.

        Purpose:
            Move strict baseline improvement to the first phase with continuous
            spline geometry and residual-gain training authority.
        Called by:
            PassDrivenPipeline._run_stage().
        Calls:
            BaselineRelativeSmokeService.metrics(), BaselineRelativeSmokeService.passed(),
            TrainingStateService.latest_phase_validation(), TrainingStateService.snapshot(),
            ExperimentService.reject().
        """''')
test_path = ROOT / T
test_path.write_text(test_path.read_text(encoding='utf-8').rstrip() + '\n', encoding='utf-8')
print('Applied V11.9 documentation/readability fix')
