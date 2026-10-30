"""V17.6 GUI command routing tests (headless; no Tk display needed)."""
import tempfile
from pathlib import Path
import unittest

from tools.nsamdr.gui.nsamdr_v176_workflow_gui import (
    PHASES, latest_report, raven_command, training_command,
)


class V176OperatorGuiTests(unittest.TestCase):
    def test_direct_script_mode_resolves_golden_editor_package(self):
        """Real GUI launcher is a file, not python -m: isolate its sys.path."""
        import subprocess
        import sys
        gui = Path(__file__).resolve().parent / "nsamdr_v176_workflow_gui.py"
        editor = gui.with_name("golden_trace_editor.py").resolve()
        probe = (
            "import importlib.util,runpy,sys,pathlib; "
            "runpy.run_path(sys.argv[1],run_name='nsamdr_gui_import_probe'); "
            "spec=importlib.util.find_spec('tools.nsamdr.gui.golden_trace_editor'); "
            "assert spec is not None, 'Golden Patch package unavailable'; "
            "assert pathlib.Path(spec.origin).resolve()==pathlib.Path(sys.argv[2]).resolve()"
        )
        with tempfile.TemporaryDirectory() as other_cwd:
            process = subprocess.run(
                [sys.executable, "-I", "-c", probe, str(gui), str(editor)],
                cwd=other_cwd, capture_output=True, text=True, timeout=30,
                check=False,
            )
        self.assertEqual(
            process.returncode, 0,
            f"direct GUI script cannot import Golden Patch editor:\n"
            f"{process.stdout}\n{process.stderr}"
        )

    def test_golden_reference_status_distinguishes_missing_valid_invalid_trace(self):
        import sys
        import types
        from unittest.mock import patch
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import golden_reference_status

        self.assertEqual(golden_reference_status(None), "prepare required")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertEqual(golden_reference_status(folder), "prepare required")
            (folder / "evidence.json").write_text("{}")
            self.assertEqual(golden_reference_status(folder), "trace required")
            (folder / "trace.json").write_text("{}")

            stub = types.ModuleType("tools.nsamdr.neural.v178.golden_patch")
            stub.validate_trace = lambda chosen: (chosen == folder) or None
            with patch.dict(sys.modules, {"tools.nsamdr.neural.v178.golden_patch": stub}):
                self.assertEqual(golden_reference_status(folder), "trace ready")
                def reject(_folder):
                    raise ValueError("trace source identity mismatch")
                stub.validate_trace = reject
                self.assertEqual(golden_reference_status(folder), "trace invalid")

    def test_train_comparison_is_real_optimisation(self):
        cmd = training_command("compare-features", "fresh", False, "auto")
        self.assertEqual(cmd[:3], ["v176-capacity-proof", "--phase", "compare-features"])
        self.assertNotIn("--audit-only", cmd)

    def test_attainable_is_separate_diagnostic(self):
        self.assertIn("attainable", PHASES)
        self.assertIn("attainable", training_command("attainable", "fresh", False, "auto"))

    def test_raven_selects_current_specialist(self):
        cmd = raven_command(r"C:\\CCP\\EVE", 10, "output10")
        self.assertEqual(cmd[0], "v176-raven-preview")
        self.assertEqual(cmd[-3:], ["10", "--mode", "output10"])
        self.assertNotIn("--target-size", cmd)

    def test_bad_selection_rejected(self):
        with self.assertRaises(ValueError):
            raven_command(r"C:\\CCP\\EVE", 11, "output10")
        with self.assertRaises(ValueError):
            training_command("v16-main", "fresh", False, "auto")

    def test_latest_report_ignores_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "proof_1"
            folder.mkdir()
            (folder / "report.json").write_text("{bad")
            self.assertIsNone(latest_report(Path(tmp)))

    def test_full_operator_stage_tree_restored(self):
        import inspect
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import App, STAGES
        self.assertEqual([s.id for s in STAGES], [
            "setup", "train", "compare", "attainable", "enhance",
            "golden", "contour", "goldenfit", "preview", "evidence", "tests", "quadtree",
        ])
        source = inspect.getsource(App._build)
        for item in ("ttk.Treeview", "ttk.Panedwindow", "controls_canvas",
                     "Command preview", "progress_value", "Stop current process",
                     "Detect artifacts", "preview_combo", "Runtime log"):
            self.assertIn(item, source)
        selection_source = inspect.getsource(App._selected)
        self.assertIn("Audit native texture channels", selection_source)
        self.assertIn("Open latest channel audit", selection_source)
        self.assertIn("Open latest golden evidence folder", selection_source)
        self.assertNotIn("Open latest Golden Patch / trace editor", selection_source)
        self.assertIn("Automatic contour inference", selection_source)

    def test_current_stage_commands_are_wired_without_tk_instance(self):
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import App
        class Var:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
        app = App.__new__(App)
        app.vars = {
            "phase": Var("compare-features"), "resume": Var("fresh"),
            "amp": Var("auto"), "device": Var("cuda"), "audit": Var(False),
            "authority": Var("10"), "mode": Var("output10"),
            "cache": Var(r"C:\CCP\EVE"),
        }
        actual = app._command_for_stage("compare")
        self.assertIn("v176-capacity-proof", actual)
        self.assertIn("compare-features", actual)
        self.assertNotIn("attainable", actual)
        actual = app._command_for_stage("preview")
        self.assertIn("v176-raven-preview", actual)
        self.assertIn("output10", actual)
        self.assertIsNone(app._command_for_stage("quadtree"))
        self.assertIn("v177-enhancement-train",app._command_for_stage("enhance"))
        self.assertIn("v178-golden-patch",app._command_for_stage("golden"))
        contour=app._command_for_stage("contour")
        self.assertIn("v178-contour-train",contour)
        self.assertEqual(contour[contour.index("--seed")+1],"17842")
        app.vars["folder"] = Var(r"D:\\REPOS\\trinity\\artifacts\\golden")
        self.assertIn("v178-golden-auto",app._command_for_stage("goldenfit"))
        self.assertNotIn("v178-golden-fit",app._command_for_stage("goldenfit"))
        automatic=app._command_for_stage("goldenfit")
        self.assertEqual(Path(automatic[automatic.index("--checkpoint")+1]).parent.name,
                         "seed_17842_patch_32_batch_4")
        app.vars["candidate_source"] = Var("v177")
        app.vars["enhancer_checkpoint"] = Var(r"C:\\candidate.pt")
        preview = app._command_for_stage("preview")
        self.assertIn("--enhancer-checkpoint",preview)
        self.assertIn(r"C:\\candidate.pt",preview)
        self.assertNotIn("channel-audit", " ".join(actual))

    def test_footer_source_matches_renderer_folder(self):
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import (
            PREVIEW_CHOICES, preview_label, App,
        )
        self.assertEqual(set(PREVIEW_CHOICES), {
            "TRAIN10_OUTPUT10", "TRAIN10_HR_FEATURES58",
            "TRAIN01_OUTPUT10", "TRAIN01_HR_FEATURES58",
        })
        for value in PREVIEW_CHOICES:
            authority = int(value[5:7])
            mode = value.split("_", 1)[1].lower()
            self.assertTrue(preview_label(authority, mode).endswith(value))

    def test_native_raven_geometry_executed_without_torch(self):
        """CI checks exact 4x geometry even without GPU/PyTorch installed."""
        import ast
        from types import SimpleNamespace
        source = (Path(__file__).resolve().parents[1] /
                  "neural/v176/raven_renderer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        funcs = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
        output = funcs["_output_size"]
        isolated = ast.Module(body=[
            ast.ImportFrom(module="__future__",
                           names=[ast.alias(name="annotations")], level=0),
            output], type_ignores=[])
        isolated = ast.fix_missing_locations(isolated)
        scope = {}
        exec(compile(isolated, "<native-geometry>", "exec"), scope)
        img = SimpleNamespace(ndim=3, shape=(1024, 2048, 4))
        self.assertEqual(scope["_output_size"](img, 4), (8192, 4096))
        img.shape = (4096, 4096, 4)
        self.assertEqual(scope["_output_size"](img, 4), (16384, 16384))
        img.shape = (4097, 4096, 4)
        with self.assertRaisesRegex(RuntimeError, "DX11"):
            scope["_output_size"](img, 4)
        # The source evidence path must never globally resize the native albedo.
        model_inputs = ast.get_source_segment(source, funcs["_model_inputs"])
        self.assertNotIn("cv2.resize(", model_inputs)
        self.assertIn("would resize original source", model_inputs)
        # Rendering many native maps must not allocate all of them on the GPU.
        tiled = ast.get_source_segment(source, funcs["_tiled_maps"])
        self.assertIn("np.memmap(", tiled)


if __name__ == "__main__":
    unittest.main()
