from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TileContextPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[3]

    def read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_public_registry_contains_only_modes_1_2_3(self) -> None:
        registry = self.read("trinityal/tests/nsamdr/NSAMDRStrategyModes.h") + self.read(
            "trinityal/tests/nsamdr/NSAMDRStrategyModes.cpp")
        self.assertIn("OriginalBaseline = 1", registry)
        self.assertIn("UvStretchDiagnostic = 2", registry)
        self.assertIn("NeuralReconstruction = 3", registry)
        self.assertIn('"1 - Original source (no cleanup)"', registry)
        self.assertIn('"2 - UV/stretch diagnostics"', registry)
        self.assertIn('"3 - NSAMDR cleanup"', registry)
        self.assertIn("kModeCount = 4", self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h"))
        for removed in ("Mode6", "Mode7", "SamplingMipCorrection", "FidelitySuperResolution4K"):
            self.assertNotIn(removed, registry)


    def test_eve_texture_v_correction_is_baked_into_obj_asset(self) -> None:
        converter = self.read("tools/nsamdr/gr2_converter/convert_eve_asset.mjs")
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        panel = self.read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        self.assertIn('schema: "NSAMDR_GR2_CONVERSION_V6_EVE_DIRECTX_LH_HANDEDNESS"', converter)
        self.assertIn("const sourceV = texcoords[i * 2 + 1]", converter)
        self.assertIn("const outputU = sourceU", converter)
        self.assertIn("const bakedTextureV = sourceV", converter)
        self.assertIn('textureVTransform: "v_out = v_gr2"', converter)
        self.assertIn("runtimeTextureVFlipRequired: false", converter)
        self.assertIn("function mirrorGr2X(value) { return -value; }", converter)
        self.assertIn("function previewTriangleWinding(a, b, c) { return [a, c, b]; }", converter)
        self.assertIn("const outputX = mirrorGr2X(sourceX)", converter)
        self.assertIn("const outputNormalX = mirrorGr2X(sourceNormalX)", converter)
        self.assertIn("validTriangles.push(previewTriangleWinding(a, b, c))", converter)
        self.assertIn('mirrorAxis: GR2_PREVIEW_MIRROR_AXIS', converter)
        self.assertIn('triangleWindingTransform: "a,b,c -> a,c,b"', converter)
        self.assertIn('uvTransform: "u_out = u_gr2; v_out = v_gr2"', converter)
        self.assertNotIn("${1 - texcoords[i * 2 + 1]}", converter)
        self.assertIn("bool flipV = false", types)
        self.assertIn('Debug invert baked texture V (V)', panel)

    def test_gr2_x_reflection_and_reversed_winding_preserve_transformed_face_normal(self) -> None:
        def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
            return tuple(a[index] - b[index] for index in range(3))

        def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def mirror_x(value: tuple[float, float, float]) -> tuple[float, float, float]:
            return (-value[0], value[1], value[2])

        a = (1.0, 0.0, 0.0)
        b = (2.0, 1.0, 0.0)
        c = (1.5, 0.25, 2.0)
        source_normal = cross(subtract(b, a), subtract(c, a))
        transformed_normal = mirror_x(source_normal)
        output_a, output_b, output_c = mirror_x(a), mirror_x(c), mirror_x(b)
        output_face_normal = cross(subtract(output_b, output_a), subtract(output_c, output_a))
        self.assertEqual(output_face_normal, transformed_normal)

    def test_same_renderer_granny_free_ab_contract(self) -> None:
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        cmake = self.read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")
        launcher = self.read("scripts/build/run_nsamdr_obj_preview_dx11.bat")
        self.assertIn("A) untouched source with the SAME high-quality sampler as the candidate", render)
        self.assertIn("Reuse the exact", render)
        self.assertIn("source geometry and index order on both sides", render)
        self.assertIn("baselineAsset.vertexBuffer", render)
        self.assertIn("baselineAsset.indexBuffer", render)
        self.assertIn("context->VSSetShader(resources.vertexShader.Get()", render)
        self.assertIn("context->PSSetShader(resources.pixelShader.Get()", render)
        self.assertIn("selectedAlbedoView = material.albedoView.Get()", render)
        self.assertNotIn("baselineVertexShader", types + render)
        self.assertNotIn("baselinePixelShader", types + render)
        self.assertNotIn("VSBaseline", shader)
        self.assertNotIn("PSBaseline", shader)
        self.assertIn("-DWITH_GRANNY=OFF", launcher)
        self.assertNotIn("trinity_dx11", cmake)
        self.assertNotIn("NSAMDRNativeEveRenderer", cmake)


    def test_mode1_is_default_and_mode3_is_fixed_visible_without_fallback(self) -> None:
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        modes = self.read("trinityal/tests/nsamdr/NSAMDRStrategyModes.cpp")
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        panel = self.read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        self.assertIn("int mode = 1", types)
        self.assertIn("bool splitCompare = true", types)
        self.assertIn('ImGui::BeginChild("NSAMDRScrollableControls"', panel)
        self.assertIn("for (const StrategyDescriptor& descriptor : m_strategyModes.Registry())", panel)
        self.assertNotIn("if (!m_strategyModes.IsVisible(candidates, mode)) continue", panel)
        self.assertIn("MODE 3 UNAVAILABLE - no source fallback", panel)
        self.assertIn("return AssetBinding{nullptr, nullptr, 0U, nullptr, false, false}", render)
        self.assertNotIn("baseline fallback shown", panel)
        self.assertIn('"3 - NSAMDR cleanup"', modes)

    def test_v9_metric_sdf_architecture_uses_the_neutral_authored_dataset(self) -> None:
        model = self.read("tools/nsamdr/neural/v9/model.py")
        dataset = self.read("tools/nsamdr/neural/v9/dataset.py")
        authored_dataset = self.read("tools/nsamdr/neural/authored_texture_dataset.py")
        losses = self.read("tools/nsamdr/neural/v9/losses.py")
        training = self.read("tools/nsamdr/neural/v9/training.py")
        config = json.loads(self.read("tools/nsamdr/neural/configs/v9_fidelity_full.json"))

        self.assertIn('MODEL_SCHEMA = "NSAMDR_SIGN_GAUGE_METRIC_SDF_RENDERER_4X_V9_8_3"', model)
        for component in (
            "class FidelityResidualNetV9",
            "class GeometryNet",
            "class BoundaryRenderer",
            "class ImplicitSDFResidualHead",
            "class ResizeDecoderStage",
        ):
            self.assertIn(component, model)
        self.assertIn('"geometryOutputs": ("sdf", "edge", "orientation", "hardness", "boundary_gate")', model)
        self.assertIn('"sharedAcrossPhysicalMaps": True', model)
        self.assertIn('"reconstructionPrimitive": "sign-gauge-metric-coarse-sdf', model)
        self.assertNotIn("nn.PixelShuffle", model)

        self.assertIn("authored_texture_dataset import prepare_dataset", dataset)
        self.assertIn("return _prepare_crop_bundles(", dataset)
        self.assertIn("def discover_shared_cache_families", authored_dataset)
        self.assertIn("def prepare_dataset", authored_dataset)
        self.assertIn("def load_normal_training_rgb", authored_dataset)
        self.assertIn("_d.dds", authored_dataset)
        self.assertIn("_n.dds", authored_dataset)
        self.assertIn("_pgs.dds", authored_dataset)

        self.assertIn("_sdf_global_polarity", losses)
        self.assertIn("_balanced_metric_band_mean", losses)
        self.assertIn("if bool(config.sdf_sign_gauge_invariant)", losses)
        self.assertIn("NSAMDR V9.8.3 GEOMETRY CHECKPOINT READY", training)
        self.assertEqual(config["targetScale"], 4)
        self.assertEqual(config["tileSize"], 128)
        self.assertEqual(config["inferenceTileSize"], 128)
        self.assertEqual(config["inferenceOverlap"], 24)
        self.assertTrue(config["sdfSignGaugeInvariant"])
        self.assertFalse(config["appearanceEnabled"])
        self.assertEqual(config["checkpointName"], "nsamdr_v9_fidelity.pt")
        self.assertEqual(config["device"], "cuda")

    def test_v9_metric_sdf_losses_keep_sensitive_reductions_in_fp32(self) -> None:
        losses = self.read("tools/nsamdr/neural/v9/losses.py")
        model = self.read("tools/nsamdr/neural/v9/model.py")
        self.assertIn("def _mean_fp32", losses)
        self.assertIn("return value.float().mean()", losses)
        self.assertIn("def _sum_fp32", losses)
        self.assertIn("return value.float().sum()", losses)
        self.assertIn('outputs["boundary_gate"].float()', losses)
        self.assertIn('"confidence_logits": confidence_logits', model)
        self.assertIn('"plateau_confidence": boundary["plateau_confidence"]', model)

    def test_v9_model_is_specialised_but_compact(self) -> None:
        from tools.nsamdr.neural.v9.config import V9Config
        from tools.nsamdr.neural.v9.model import FidelityResidualNetV9, parameter_count

        model = FidelityResidualNetV9(V9Config())
        contract = model.architecture_contract()
        self.assertEqual(contract["geometryModel"], "GeometryNet")
        self.assertEqual(contract["renderer"], "BoundaryRenderer")
        self.assertFalse(contract["geometryCanPaintRgb"])
        self.assertTrue(contract["sharedAcrossPhysicalMaps"])
        self.assertGreaterEqual(parameter_count(model), 6_000_000)
        self.assertLessEqual(parameter_count(model), 11_000_000)
        self.assertNotIn("PixelShuffle", {type(module).__name__ for module in model.modules()})

    def test_runtime_per_pixel_compute_is_removed(self) -> None:
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        application = self.read("trinityal/tests/nsamdr/NSAMDRPreviewApplication.h")
        cmake = self.read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")
        for relative in (
            "trinityal/tests/nsamdr/NSAMDRNeuralRuntime.h",
            "trinityal/tests/nsamdr/NSAMDRNeuralRuntime.cpp",
            "trinityal/tests/nsamdr/NSAMDRNeuralWeights.hlsli",
        ):
            self.assertFalse((self.root / relative).exists(), relative)
        combined = shader + render + types + application + cmake
        self.assertNotIn("CSNSAMDRNeuralReconstruction", combined)
        self.assertNotIn("NSAMDRNeuralRuntime", combined)
        self.assertNotIn("NSAMDRNeuralWeights", combined)
        self.assertNotIn("neuralAlbedoView", combined)
        self.assertNotIn("neuralAlbedoReady", combined)
        self.assertIn("already reconstructed by the offline V9 CUDA fidelity 4x", render)

    def test_candidate_generation_bakes_overlapping_tile_inference(self) -> None:
        generator = self.read("tools/nsamdr/generate_strategy_candidates.py")
        self.assertIn('REPORT_SCHEMA = "NSAMDR_THREE_MODE_PIPELINE_V9_6_SCIENTIFIC_CONTROL"', generator)
        self.assertIn("nsamdr_v9_fidelity.pt", generator)
        self.assertIn("_apply_tile_context_model", generator)
        self.assertIn("tile_model.infer_tiled", generator)
        self.assertIn('"runtimeComputeKernelRequired": False', generator)
        self.assertIn('"fp16OverlappingInferenceBaked": True', generator)
        self.assertIn('"bootstrapCandidateAvailable": False', generator)
        self.assertIn('"offlineCudaNeuralInference": True', generator)
        self.assertIn('"materialMapPassthrough": False', generator)
        self.assertIn('"mapsReconstructed": ["albedo", "normalXY", "materialRGB", "roughness", "emissive"]', generator)
        self.assertIn('"bootstrapCandidate": False', generator)
        self.assertIn("deterministic bootstrap is disabled", generator)
        self.assertIn("sign-gauge metric-SDF geometry-convergence renderer 4x", generator)
        self.assertNotIn("NSAMDR_THREE_MODE_PIPELINE_V1", generator)

    def test_mode3_ui_describes_offline_baked_reconstruction(self) -> None:
        panel = self.read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        mode3 = self.read("trinityal/tests/nsamdr/NSAMDRMode3Pipeline.cpp")
        app = self.read("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
        self.assertIn("Mode 3 CUDA neural reconstruction", panel)
        self.assertIn("Offline retraining", panel)
        self.assertIn("shared BoundaryRenderer", panel)
        self.assertIn("continuous 4x signed-distance field", mode3)
        self.assertIn("FP16 overlapping CUDA inference bakes reconstructed resources", mode3)
        self.assertIn("trained V9 fidelity-first checkpoint", app)
        self.assertNotIn("Mode 3 runtime correction", panel)
        self.assertNotIn("Redispatch loaded model", panel)

    def test_training_controller_writes_v9_profile_and_uses_dispatcher(self) -> None:
        controller = self.read("trinityal/tests/nsamdr/NSAMDRTrainingController.cpp")
        for key in (
            'datasetManifest', 'datasetRoot', 'maxFamilies', 'cropsPerFamily',
            'sourceCropSize', 'identityEpochs', 'residualEpochs', 'boundaryEpochs',
            'detailEpochs', 'physicalFinetuneEpochs', 'tilesPerEpoch',
            'validationTiles', 'tileSize', 'targetScale', 'widths',
            'blocksPerLevel', 'decoderBlocks', 'inferenceTileSize',
            'inferenceOverlap',
        ):
            self.assertIn(key, controller)
        self.assertIn("nsamdr_v9_fidelity.pt", controller)
        self.assertIn("nsamdr_v9_fidelity.json", controller)
        self.assertIn(r"\\scripts\\build\\nsamdr.bat", controller)
        self.assertIn('L" retrain-preview --config "', controller)
        self.assertIn("--shared-cache", controller)
        self.assertIn("--wait-pid", controller)

    def test_cuda_setup_and_v9_workflows_are_exposed_by_the_dispatcher(self) -> None:
        setup = self.read("scripts/build/setup_nsamdr_cuda.bat")
        launcher = self.read("scripts/build/nsamdr.bat")
        dispatcher = self.read("tools/nsamdr/nsamdr_cli.py")
        self.assertIn("torch==2.11.0", setup)
        self.assertIn("/whl/cu128", setup)
        self.assertIn("--require-arch sm_120", setup)
        self.assertIn(r"%ROOT%\tools\nsamdr\nsamdr_cli.py", launcher)
        self.assertIn(r"artifacts\nsamdr\python-env", launcher)
        self.assertIn(r"artifacts\nsamdr\python-env-cpu", launcher)
        for command in (
            "gui", "setup", "tune", "index", "train", "preview", "candidate",
            "compare", "promote", "validate", "test", "cleanup", "integrate",
            "run", "retrain-preview", "native",
        ):
            self.assertIn(f'add_parser("{command}"', dispatcher)
        for backend in (
            "run_nsamdr_v9_raven_tune_preview.py",
            "index_eve_texture_dataset_v9.py",
            "train_nsamdr_v9_preview_experiment.py",
            "train_nsamdr_v9.py",
            "preview_nsamdr_v9_experiment.py",
            "compare_nsamdr_v9_experiments.py",
            "promote_nsamdr_v9_experiment.py",
        ):
            self.assertIn(backend, dispatcher)

    def test_dispatcher_layout_validation_and_cleanup_are_scoped(self) -> None:
        dispatcher = self.read("tools/nsamdr/nsamdr_cli.py")
        for retained_batch in (
            "scripts/build/nsamdr.bat",
            "scripts/build/run_nsamdr_v9_gui.bat",
            "scripts/build/setup_nsamdr_cuda.bat",
            "scripts/build/setup_nsamdr_cpu.bat",
            "scripts/build/run_nsamdr_obj_preview_dx11.bat",
        ):
            self.assertIn(retained_batch, dispatcher)
        self.assertIn("REQUIRED_LAYOUT = (", dispatcher)
        self.assertIn("missing = [relative for relative in REQUIRED_LAYOUT", dispatcher)
        self.assertIn("def _safe_resolved_target(target: Path) -> Path:", dispatcher)
        self.assertIn("def _safe_target(relative: str) -> Path:", dispatcher)
        self.assertIn("resolved.relative_to(root)", dispatcher)
        self.assertIn('raise ValueError("refusing to clean the repository root")', dispatcher)
        self.assertIn('cleanup.add_argument("--dry-run"', dispatcher)
        for artifact_root in (
            "artifacts/nsamdr/training_v9_preview_raven",
            "artifacts/nsamdr/experiments",
            "artifacts/nsamdr/promoted",
            "artifacts/nsamdr/training_v9",
            "artifacts/nsamdr/neural_v9",
        ):
            self.assertIn(artifact_root, dispatcher)

    def test_cmake_registers_every_cpp_and_header(self) -> None:
        cmake = self.read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")
        source_dir = self.root / "trinityal/tests/nsamdr"
        for path in source_dir.iterdir():
            if path.suffix in {".cpp", ".h"}:
                self.assertIn(path.name, cmake, path.name)
        self.assertNotIn("NSAMDRNeuralRuntime", cmake)
        self.assertNotIn("NSAMDRNeuralWeights", cmake)

    def test_build_uses_cmake_incremental_state_without_private_overlay(self) -> None:
        launcher = self.read("scripts/build/run_nsamdr_obj_preview_dx11.bat")
        self.assertNotIn("SourceBuildState.ps1", launcher)
        self.assertIn("CMake will perform its normal incremental configure/build", launcher)
        self.assertNotIn("vcpkg-overlay-ports", launcher)
        self.assertIn('set "VCPKG_OVERLAY_PORTS="', launcher)
        self.assertIn("-DVCPKG_OVERLAY_PORTS=", launcher)
        self.assertIn("-DVCPKG_MANIFEST_INSTALL=OFF", launcher)
        self.assertIn("-DFXC_TOOL:FILEPATH=", launcher)
        self.assertIn(":resolve_fxc", launcher)
        self.assertIn("Windows Kits\\10\\bin", launcher)
        self.assertIn("NSAMDR will not download CCP's private fxc package", launcher)
        self.assertIn("cmake --preset x64-windows-trinitydev", launcher)
        self.assertIn("Windows SDK x64 fxc.exe", launcher)
        self.assertIn("Refusing non-x64 FXC compiler", launcher)
        self.assertNotIn("where.exe /r", launcher[launcher.index(":resolve_fxc"):launcher.index(":resolve_model")])

    def test_gr2_converter_source_is_packaged_and_dependency_complete(self) -> None:
        converter_dir = self.root / "tools/nsamdr/gr2_converter"
        package = json.loads((converter_dir / "package.json").read_text(encoding="utf-8"))
        converter = (converter_dir / "convert_eve_asset.mjs").read_text(encoding="utf-8")
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        launcher = self.read("scripts/build/run_nsamdr_obj_preview_dx11.bat")
        dispatcher = self.read("tools/nsamdr/nsamdr_cli.py")
        shim_dir = converter_dir / "vendor/core-math-compat"
        self.assertTrue((converter_dir / "README.md").is_file())
        self.assertIn("github.com/carbonenginejs/format-gr2/archive/", package["dependencies"]["@carbonenginejs/format-gr2"])
        self.assertIn("github.com/carbonenginejs/runtime-resource/archive/", package["dependencies"]["@carbonenginejs/runtime-resource"])
        self.assertIn("github.com/carbonenginejs/runtime-utils/archive/", package["dependencies"]["@carbonenginejs/runtime-utils"])
        self.assertIn("github.com/rawrafox/black-reader-js/archive/", package["dependencies"]["black-reader"])
        self.assertEqual(package["dependencies"]["@carbonenginejs/core-math"], "file:vendor/core-math-compat")
        shim_package = json.loads((shim_dir / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(shim_package["name"], "@carbonenginejs/core-math")
        self.assertEqual(shim_package["version"], "0.1.5")
        for module in ("mesh", "num", "tangent", "vec3"):
            self.assertIn(f'@carbonenginejs/runtime-utils/{module}', (shim_dir / f"{module}.js").read_text(encoding="utf-8"))
        self.assertIn('from "@carbonenginejs/format-gr2"', converter)
        self.assertIn('from "@carbonenginejs/runtime-resource/formats/dds"', converter)
        self.assertIn('from "black-reader"', converter)
        self.assertIn('from "black-reader/black-classes.js"', converter)
        self.assertIn('command === "gr2-to-obj"', converter)
        self.assertIn('command === "dds-to-png"', converter)
        self.assertIn('command === "dds-to-environment-png"', converter)
        self.assertIn('command === "sof-to-json"', converter)
        self.assertIn("selectModelMeshes", converter)
        self.assertIn("collapseLodAlternatives", converter)
        self.assertIn("highest-detail-per-mesh-family", converter)
        self.assertIn("NSAMDR_GR2_CONVERSION_V6_EVE_DIRECTX_LH_HANDEDNESS", converter)
        self.assertIn("NSAMDR_SOF_VISUALS_V2", converter)
        self.assertIn("materialIndex", converter)
        self.assertNotIn("sofUnsupported", converter)
        self.assertIn("package-lock.json", eve_test)
        self.assertIn("shutil.rmtree(node_modules)", eve_test)
        self.assertIn("--package-lock=false", eve_test)
        self.assertIn("public GitHub source archives", eve_test)
        self.assertIn("probe_converter_modules", eve_test)
        self.assertIn("CONVERTER_MODULE_PROBE", eve_test)
        self.assertIn("await import('@carbonenginejs/format-gr2')", eve_test)
        self.assertIn("existing_groups", eve_test)
        self.assertIn("for group_index in range(first_group, first_group + group_count)", eve_test)
        self.assertIn("--input-type=module", eve_test)
        self.assertIn("Node could not import the converter's actual entry points", eve_test)
        self.assertNotIn("installed_core_math.is_dir()", eve_test)
        self.assertNotIn("installed_runtime_utils.is_dir()", eve_test)
        self.assertIn("--input-type=module", launcher)
        self.assertIn("await import('@carbonenginejs/format-gr2')", launcher)
        self.assertIn("Node could not import the converter entry points", launcher)
        self.assertNotIn("CONVERTER_CORE_MATH", launcher)
        self.assertNotIn("CONVERTER_RUNTIME_UTILS", launcher)
        self.assertIn("--package-lock=false", launcher)
        self.assertIn('native_eve = native_commands.add_parser("eve"', dispatcher)
        self.assertIn('"tools/nsamdr/eve_asset_test.py"', dispatcher)


    def test_direct_raven_path_uses_deterministic_model_identity_when_sde_has_no_match(self) -> None:
        import zipfile
        from unittest.mock import patch
        from tools.nsamdr.eve_asset_test import ResourceRow, _resolve_sof_identity

        model = ResourceRow(
            "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2",
            "hashed/cb1_t1.gr2",
            "resfileindex.txt",
        )
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / "sde.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("graphics.jsonl", "")
                archive.writestr("types.jsonl", "")
            with patch("tools.nsamdr.eve_asset_test._ensure_sde_archive", return_value=archive_path):
                identity = _resolve_sof_identity([model], self.root, model, "")

        self.assertEqual(identity["hull"], "cb1_t1")
        self.assertEqual(identity["faction"], "caldaribase")
        self.assertEqual(identity["race"], "caldari")
        self.assertEqual(identity["raceSource"], "modelPath")

    def test_base_texture_resolution_preserves_authored_source_before_faction_insert(self) -> None:
        from tools.nsamdr.eve_asset_test import ResourceRow, _resolve_sof_texture

        logical = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1_d.dds"
        modified = "res:/dx9/model/ship/caldari/battleship/cb1/navy/cb1_navy_t1_d.dds"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "base.dds").write_bytes(b"base")
            (root / "navy.dds").write_bytes(b"navy")
            rows = {
                logical.lower(): ResourceRow(logical, "base.dds", "index"),
                modified.lower(): ResourceRow(modified, "navy.dds", "index"),
            }
            faction = _resolve_sof_texture(rows, root, logical, "navy")
            (root / "navy.dds").unlink()
            source = _resolve_sof_texture(rows, root, logical, "navy")

        self.assertIsNotNone(source)
        self.assertIsNotNone(faction)
        self.assertEqual(source.logical, logical)
        self.assertEqual(faction.logical, modified)

    def test_sof_failure_writes_an_explicitly_incomplete_tint_only_manifest(self) -> None:
        import csv
        from tools.nsamdr.eve_asset_test import _write_tint_only_material_manifest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conversion = root / "ship.conversion.json"
            conversion.write_text(json.dumps({
                "drawRanges": [
                    {"groupIndex": 0, "indexCount": 3},
                    {"groupIndex": 1, "indexCount": 6},
                ]
            }), encoding="utf-8")
            manifest = _write_tint_only_material_manifest(
                root, conversion, "caldari", "SOF unavailable")
            lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
            rows = list(csv.DictReader(lines, delimiter="\t"))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(not row["albedo"] and not row["normal"] and not row["material"] for row in rows))
            self.assertTrue(all(row["semantic_complete"] == "0" for row in rows))
            self.assertTrue(all(row["baseline_complete"] == "0" for row in rows))
            self.assertTrue(all(row["unresolved_semantics"] == "sof_visual_manifest" for row in rows))

            report = json.loads((root / "ship.materials.report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["unresolvedCount"], 2)
            self.assertEqual(report["reason"], "SOF unavailable")


    def test_legacy_pgs_keeps_authored_albedo_identity(self) -> None:
        from tools.nsamdr.eve_asset_test import _semantic_texture_layout

        pgs = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1_pgs.dds"
        layout = _semantic_texture_layout({
            "shader": "res:/graphics/effect/managed/space/ship/quad.fx",
        }, {
            "DiffuseMap": "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1_d.dds",
            "NormalMap": "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1_n.dds",
            "PgsMap": pgs,
        })

        self.assertEqual(layout["shaderFamily"], "legacy_pgs")
        self.assertEqual(layout["material"], pgs)
        self.assertEqual(layout["glow"], "")
        self.assertTrue(layout["semanticComplete"])
        self.assertEqual(layout["channels"]["normalX"], 3)
        self.assertEqual(layout["channels"]["roughness"], 1)
        self.assertEqual(layout["channels"]["material"], 2)

        source = self.read("tools/nsamdr/eve_asset_test.py")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        self.assertIn("Pre-PBR PGS: R=sub-mask, G=specular, B=mask, A=glow/opacity", source)
        self.assertIn("The renderer keeps this family explicit", source)
        self.assertIn("shaderFamily == 1", shader)
        self.assertIn("_d maps are already authored colour textures", shader)
        self.assertNotIn("Legacy EVE PGS stores glow/opacity in alpha", shader)

    def test_sof_area_indices_map_to_render_lod_draws_only(self) -> None:
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        converter = self.read("tools/nsamdr/gr2_converter/convert_eve_asset.mjs")
        self.assertIn("selectedMeshCount", converter)
        self.assertIn("selectedMeshIndices", converter)
        self.assertIn("rejectedLodMeshIndices", converter)
        self.assertIn("collapseLodAlternatives", converter)
        self.assertIn("Prefer the unsuffixed production/high-detail mesh", converter)
        self.assertIn("materialIndexFromGroup", converter)
        self.assertIn("groupIndex,", converter)
        self.assertIn('existing_groups = {int(draw.get("groupIndex", -1)) for draw in draw_ranges}', eve_test)
        self.assertIn("for group_index in range(first_group, first_group + group_count)", eve_test)
        self.assertIn("if group_index not in existing_groups", eve_test)
        self.assertIn('assigned_groups = {int(record["group"]) for record in records}', eve_test)

    def test_sof_hull_resolver_uses_exact_model_geometry_not_directory_guess(self) -> None:
        converter = self.read("tools/nsamdr/gr2_converter/convert_eve_asset.mjs")
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        self.assertIn("function resolveSofHull", converter)
        self.assertIn('"exact-geometry-path"', converter)
        self.assertIn('"geometry-basename"', converter)
        self.assertIn('"geometry-family"', converter)
        self.assertIn("SOF hull resolution is ambiguous", converter)
        self.assertIn("Nearby candidates", converter)
        self.assertIn("requestedModelPath", converter)
        self.assertIn("hullResolution", converter)
        self.assertIn("model.logical", eve_test)
        self.assertIn("_resolve_sof_identity(rows, repo_root, model, selection_key)", eve_test)
        self.assertIn('str(sof_identity["hull"])', eve_test)
        self.assertIn("raceSource", eve_test)

    def test_current_sof_schema_includes_shield_ellipsoid_flag(self) -> None:
        converter = self.read("tools/nsamdr/gr2_converter/convert_eve_asset.mjs")
        self.assertIn('ensureBlackClass("EveSOFDataHullExtensionPlacement"', converter)
        self.assertIn("extendsShieldEllipsoid: r.boolean", converter)

    def test_incomplete_sof_materials_are_explicit_and_conversion_schema_fails_closed(self) -> None:
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        processor = self.read("trinityal/tests/nsamdr/NSAMDRAssetProcessor.cpp")
        self.assertIn("_write_tint_only_material_manifest", eve_test)
        self.assertIn("incomplete tint-only fallback", eve_test)
        self.assertIn('"complete": False', eve_test)
        self.assertIn('value("semantic_complete")', processor)
        self.assertIn('value("baseline_complete")', processor)
        self.assertIn('value("unresolved_semantics")', processor)
        self.assertIn('EXPECTED_GR2_CONVERSION_SCHEMA = "NSAMDR_GR2_CONVERSION_V6_EVE_DIRECTX_LH_HANDEDNESS"', eve_test)
        self.assertIn('conversion_record.get("schema") != EXPECTED_GR2_CONVERSION_SCHEMA', eve_test)

    def test_readme_documents_the_active_v9_operator_workflow(self) -> None:
        readme = self.read("trinityal/tests/nsamdr/README.md")
        for phrase in (
            "# NSAMDR V9",
            "Operator guide",
            "Raven tuning",
            "all-assets production run",
            "same production architecture",
            "Configuration promotion",
            "Full production training",
            "Full production preview",
            "nsamdr_v9_fidelity.pt",
        ):
            self.assertIn(phrase, readme)

    def test_manifest_contains_only_mode3_candidate(self) -> None:
        from tools.nsamdr.strategy_pipeline.model import CandidateArtifact, StrategyManifest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            obj = root / "ship.obj"
            materials = root / "ship.materials.tsv"
            obj.write_text("", encoding="utf-8")
            materials.write_text("", encoding="utf-8")
            manifest = StrategyManifest(
                "NSAMDR_THREE_MODE_PIPELINE_V9_6_SCIENTIFIC_CONTROL", 4096, obj, materials)
            manifest.add(CandidateArtifact(3, "NSAMDR cleanup", obj, materials, metadata={"offlineCudaNeuralInference": True}))
            report = manifest.to_report()
            self.assertEqual(report["schema"], "NSAMDR_THREE_MODE_PIPELINE_V9_6_SCIENTIFIC_CONTROL")
            self.assertEqual(set(report["strategies"]), {"3"})
            self.assertTrue(report["strategies"]["3"]["offlineCudaNeuralInference"])

    def test_no_inl_and_window_icon_is_integrated(self) -> None:
        source_dir = self.root / "trinityal/tests/nsamdr"
        self.assertFalse(list(source_dir.glob("*.inl")))
        window_icon = self.read("trinityal/tests/nsamdr/NSAMDRWindowIcon.cpp")
        cmake = self.read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")
        application = self.read("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
        self.assertTrue((source_dir / "NSAMDRPreviewIcon.ico").is_file())
        self.assertTrue((source_dir / "NSAMDRPreviewIcon.png").is_file())
        self.assertIn("WM_SETICON", window_icon)
        self.assertIn("SetClassLongPtrW", window_icon)
        self.assertIn("GCLP_HICON", window_icon)
        self.assertIn("GCLP_HICONSM", window_icon)
        self.assertIn("IDI_NSAMDR_PREVIEW", window_icon)
        self.assertIn("1 ICON", cmake)
        self.assertIn("101 ICON", cmake)
        self.assertIn("NSAMDRPreviewIcon.generated.rc", cmake)
        self.assertLess(application.index("m_host.resize(1440U, 900U)"), application.index("windowIcon.Apply"))

    def test_only_test_fixture_uses_inheritance(self) -> None:
        cpp = self.read("trinityal/tests/nsamdr/NSAMDRShipPreview.cpp")
        self.assertIn("public WithValidRenderContext", cpp)
        for path in (self.root / "trinityal/tests/nsamdr").glob("*.h"):
            if path.name == "NSAMDRShipPreview.cpp":
                continue
            self.assertNotIn(" : public ", path.read_text(encoding="utf-8", errors="ignore"), path.name)


if __name__ == "__main__":
    unittest.main()
