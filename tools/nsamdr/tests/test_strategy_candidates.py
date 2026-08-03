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
        self.assertIn('schema: "NSAMDR_GR2_CONVERSION_V5_BAKED_EVE_TEXTURE_V"', converter)
        self.assertIn("const sourceV = texcoords[i * 2 + 1]", converter)
        self.assertIn("const bakedTextureV = sourceV", converter)
        self.assertIn('textureVTransform: "v_out = v_gr2"', converter)
        self.assertIn("runtimeTextureVFlipRequired: false", converter)
        self.assertNotIn("${1 - texcoords[i * 2 + 1]}", converter)
        self.assertIn("bool flipV = false", types)
        self.assertIn('Debug invert baked texture V (V)', panel)

    def test_same_renderer_granny_free_ab_contract(self) -> None:
        render = self.read("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        cmake = self.read("scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake")
        launcher = self.read("scripts/build/run_nsamdr_obj_preview_dx11.bat")
        self.assertIn("strict same-renderer A/B comparison", render)
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
        self.assertIn("MODE 3 UNAVAILABLE - no Mode 1 fallback", panel)
        self.assertIn("return AssetBinding{nullptr, nullptr, 0U, nullptr, false, false}", render)
        self.assertNotIn("baseline fallback shown", panel)
        self.assertIn('"3 - NSAMDR cleanup"', modes)

    def test_v4_tile_context_architecture(self) -> None:
        trainer = self.read("tools/nsamdr/neural/train_nsamdr_kernel.py")
        config = json.loads(self.read("tools/nsamdr/neural/default_training_config.json"))
        self.assertIn('MODEL_SCHEMA = "NSAMDR_TILE_CONTEXT_MATERIAL_V4"', trainer)
        self.assertIn("class MaterialTileContextNet", trainer)
        self.assertIn("class DilatedResidualBlock", trainer)
        self.assertIn("DILATION_PATTERN = (1, 2, 4, 8, 8, 4, 2, 1)", trainer)
        self.assertIn("def receptive_field_pixels", trainer)
        self.assertIn("def infer_tiled", trainer)
        self.assertIn("F.grid_sample", trainer)
        self.assertIn('"flow": flow', trainer)
        self.assertIn('"residual": residual', trainer)
        self.assertIn('"confidence": confidence', trainer)
        self.assertEqual(config["batchSize"], 8)
        self.assertEqual(config["baseChannels"], 32)
        self.assertEqual(config["residualBlocks"], 8)
        self.assertEqual(config["tileSize"], 96)
        self.assertEqual(config["inferenceTileSize"], 512)
        self.assertEqual(config["inferenceOverlap"], 64)
        self.assertEqual(config["device"], "cuda")

    def test_v4_model_is_contextual_but_compact(self) -> None:
        from tools.nsamdr.neural.train_nsamdr_kernel import (
            MaterialTileContextNet,
            parameter_count,
            receptive_field_pixels,
        )

        model = MaterialTileContextNet()
        self.assertGreaterEqual(parameter_count(model), 100_000)
        self.assertLessEqual(parameter_count(model), 500_000)
        self.assertEqual(receptive_field_pixels(8), 125)

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
        self.assertIn("already reconstructed by the offline V4 tile-context", render)

    def test_candidate_generation_bakes_overlapping_tile_inference(self) -> None:
        generator = self.read("tools/nsamdr/generate_strategy_candidates.py")
        self.assertIn('REPORT_SCHEMA = "NSAMDR_THREE_MODE_PIPELINE_V2_TILE_CONTEXT"', generator)
        self.assertIn("nsamdr_tile_context.pt", generator)
        self.assertIn("_apply_tile_context_model", generator)
        self.assertIn("tile_model.infer_tiled", generator)
        self.assertIn('"runtimeComputeKernelRequired": False', generator)
        self.assertIn('"overlappingTileInferenceBaked": tile_runtime is not None', generator)
        self.assertIn('"bootstrapCandidateAvailable": tile_runtime is None', generator)
        self.assertIn('"offlineTileContext": tile_runtime is not None', generator)
        self.assertIn('"bootstrapCandidate": tile_runtime is None', generator)
        self.assertIn("previous V3 per-pixel checkpoint is intentionally incompatible", generator)
        self.assertNotIn("NSAMDR_THREE_MODE_PIPELINE_V1", generator)

    def test_mode3_ui_describes_offline_baked_reconstruction(self) -> None:
        panel = self.read("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
        mode3 = self.read("trinityal/tests/nsamdr/NSAMDRMode3Pipeline.cpp")
        app = self.read("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
        self.assertIn("Mode 3 baked tile reconstruction", panel)
        self.assertIn("Offline retraining", panel)
        self.assertIn("continuous source transport", panel)
        self.assertIn("125-pixel receptive field", panel + mode3)
        self.assertIn("baked reconstructed material textures", mode3)
        self.assertIn("trained V4 tile-context inference or the deterministic bootstrap", app)
        self.assertNotIn("Mode 3 runtime correction", panel)
        self.assertNotIn("Redispatch loaded model", panel)

    def test_training_controller_writes_v4_profile(self) -> None:
        controller = self.read("trinityal/tests/nsamdr/NSAMDRTrainingController.cpp")
        types = self.read("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
        for key in (
            'tilesPerEpoch', 'baseChannels', 'residualBlocks', 'tileSize',
            'reconstructionWeight', 'edgeWeight', 'confidenceWeight',
            'flowSmoothnessWeight', 'inferenceTileSize', 'inferenceOverlap',
        ):
            self.assertIn(key, controller)
        self.assertIn("nsamdr_tile_context.pt", controller)
        self.assertIn("nsamdr_tile_context.json", controller)
        for field in ("tilesPerEpoch", "baseChannels", "residualBlocks", "tileSize"):
            self.assertIn(field, types)
        for old in ("samplesPerEpoch", "hiddenChannels", "transportWeight", "requestNeuralRedispatch"):
            self.assertNotIn(old, controller + types)

    def test_cuda_training_and_prompt_are_retained(self) -> None:
        setup = self.read("scripts/build/setup_nsamdr_cuda.bat")
        train_bat = self.read("scripts/build/train_nsamdr.bat")
        self.assertIn("torch==2.11.0", setup)
        self.assertIn("/whl/cu128", setup)
        self.assertIn("--require-arch sm_120", setup)
        self.assertIn("SELECT NSAMDR TRAINING DEVICE", train_bat)
        self.assertIn("Select 1 or 2 [1]", train_bat)
        self.assertIn("artifacts\\nsamdr\\python-env", train_bat)
        self.assertIn("python-env-cpu", train_bat)

    def test_layout_verifier_is_strict_and_removes_v3_runtime(self) -> None:
        verifier = self.read("scripts/build/verify_and_clean_nsamdr_layout.bat")
        required_start = verifier.index('for %%F in (\n    "trinityal\\CMakeLists.txt"')
        required_block = verifier[required_start:verifier.index(") do call :RequireFile", required_start)]
        for stale in (
            "NSAMDRNeuralRuntime.h",
            "NSAMDRNeuralRuntime.cpp",
            "NSAMDRNeuralWeights.hlsli",
        ):
            self.assertIn(f'call :RemoveFile "trinityal\\tests\\nsamdr\\{stale}"', verifier)
            self.assertNotIn(stale, required_block)
        for upstream in ("trinityal\\scripts", "trinityal\\tools", "trinityal\\trinityal"):
            self.assertNotIn(f'call :RemoveDirectory "{upstream}"', verifier)
        self.assertIn("Preserve all upstream TrinityAL directories", verifier)
        self.assertIn("RepairMissingTrinityALMarker.ps1", verifier)
        self.assertIn('call :RemoveFile "scripts\\build\\nsamdr\\SourceBuildState.ps1"', verifier)
        self.assertIn(":CleanNSAMDRSourceDirectory", verifier)
        self.assertNotIn('rmdir /s /q "scripts"', verifier)
        self.assertNotIn('rmdir /s /q "tools"', verifier)
        self.assertNotIn('rmdir /s /q "trinityal"', verifier)

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
        repair = self.read("scripts/build/nsamdr/RepairMissingTrinityALMarker.ps1")
        verifier = self.read("scripts/build/verify_and_clean_nsamdr_layout.bat")
        self.assertNotIn("SourceBuildState.ps1", launcher)
        self.assertNotIn("Get-SourceStateSignature", launcher + repair)
        self.assertNotIn("Get-FileHash", launcher + repair)
        self.assertIn("CMake will perform its normal incremental configure/build", launcher)
        self.assertNotIn("vcpkg-overlay-ports", launcher)
        self.assertIn('set "VCPKG_OVERLAY_PORTS="', launcher)
        self.assertIn("-DVCPKG_OVERLAY_PORTS=", launcher)
        self.assertIn("-DVCPKG_MANIFEST_INSTALL=OFF", launcher)
        self.assertIn("-DFXC_TOOL:FILEPATH=", launcher)
        self.assertIn(":resolve_fxc", launcher)
        self.assertIn("Windows Kits\\10\\bin", launcher)
        self.assertIn("NSAMDR will not download CCP's private fxc package", launcher)
        self.assertIn("ls-files --deleted -- trinityal", repair)
        self.assertIn("'restore', '--source=HEAD', '--worktree'", repair)
        self.assertIn("trinityal/tests/nsamdr/", repair)
        self.assertIn("Existing files were not overwritten", repair)
        self.assertIn("trinityal\\ALLog.h", verifier)
        self.assertIn("trinityal\\tests\\ALResultTest.cpp", verifier)
        self.assertIn("No commit was created", repair)
        self.assertIn("TRINITYAL_REPAIR", verifier)
        self.assertIn("Windows SDK x64 fxc.exe", launcher)
        self.assertIn("Refusing non-x64 FXC compiler", launcher)
        self.assertNotIn("where.exe /r", launcher[launcher.index(":resolve_fxc"):launcher.index(":resolve_model")])

    def test_gr2_converter_source_is_packaged_and_dependency_complete(self) -> None:
        converter_dir = self.root / "tools/nsamdr/gr2_converter"
        package = json.loads((converter_dir / "package.json").read_text(encoding="utf-8"))
        converter = (converter_dir / "convert_eve_asset.mjs").read_text(encoding="utf-8")
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        launcher = self.read("scripts/build/run_nsamdr_obj_preview_dx11.bat")
        verifier = self.read("scripts/build/verify_and_clean_nsamdr_layout.bat")
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
        self.assertIn("NSAMDR_GR2_CONVERSION_V5_BAKED_EVE_TEXTURE_V", converter)
        self.assertIn("NSAMDR_SOF_VISUALS_V2", converter)
        self.assertIn("materialIndex", converter)
        self.assertNotIn("sofUnsupported", converter)
        self.assertIn("package-lock.json", eve_test)
        self.assertIn("shutil.rmtree(node_modules)", eve_test)
        self.assertIn("--package-lock=false", eve_test)
        self.assertIn("public GitHub source archives", eve_test)
        self.assertIn("probe_converter_modules", eve_test)
        self.assertIn("CONVERTER_MODULE_PROBE", eve_test)
        self.assertIn("await import('black-reader')", eve_test)
        self.assertIn("draws_by_material_index", eve_test)
        self.assertIn('draw.get("materialIndex", group_index)', eve_test)
        self.assertIn("--input-type=module", eve_test)
        self.assertIn("Node could not import the GR2, DDS or EVE Black reader entry points", eve_test)
        self.assertNotIn("installed_core_math.is_dir()", eve_test)
        self.assertNotIn("installed_runtime_utils.is_dir()", eve_test)
        self.assertIn("--input-type=module", launcher)
        self.assertIn("await import('@carbonenginejs/format-gr2')", launcher)
        self.assertIn("Node could not import the converter entry points", launcher)
        self.assertNotIn("CONVERTER_CORE_MATH", launcher)
        self.assertNotIn("CONVERTER_RUNTIME_UTILS", launcher)
        self.assertIn("--package-lock=false", launcher)
        self.assertIn(r'tools\nsamdr\gr2_converter\convert_eve_asset.mjs', verifier)
        self.assertIn(r'tools\nsamdr\gr2_converter\vendor\core-math-compat\package.json', verifier)


    def test_direct_raven_path_uses_deterministic_base_sof_identity(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.eve_asset_test import ResourceRow, _resolve_sof_identity

        model = ResourceRow(
            "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2",
            "hashed/cb1_t1.gr2",
            "resfileindex.txt",
        )
        with patch(
            "tools.nsamdr.eve_asset_test._ensure_sde_archive",
            side_effect=AssertionError("direct model identity must not scan shared SDE graphics"),
        ):
            identity = _resolve_sof_identity([model], self.root, model, "")

        self.assertEqual(identity["hull"], "cb1")
        self.assertEqual(identity["faction"], "caldaribase")
        self.assertEqual(identity["race"], "caldari")
        self.assertEqual(identity["identitySource"], "direct-model-base")
        self.assertFalse(identity["preferFactionTextures"])

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
            source = _resolve_sof_texture(rows, root, logical, "navy", False)
            faction = _resolve_sof_texture(rows, root, logical, "navy", True)

        self.assertIsNotNone(source)
        self.assertIsNotNone(faction)
        self.assertEqual(source.logical, logical)
        self.assertEqual(faction.logical, modified)

    def test_sof_failure_uses_real_extracted_textures_not_neutral_1x1_fallback(self) -> None:
        import csv
        from tools.nsamdr.eve_asset_test import _write_extracted_texture_material_manifest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conversion = root / "ship.conversion.json"
            conversion.write_text(json.dumps({
                "drawRanges": [
                    {"groupIndex": 0, "indexCount": 3},
                    {"groupIndex": 1, "indexCount": 6},
                ]
            }), encoding="utf-8")
            albedo = root / "ship_albedo.png"
            normal = root / "ship_normal.png"
            albedo.write_bytes(b"real-albedo-placeholder")
            normal.write_bytes(b"real-normal-placeholder")

            manifest = _write_extracted_texture_material_manifest(
                root, conversion, "caldari", "SOF unavailable", albedo, normal, None)
            lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
            rows = list(csv.DictReader(lines, delimiter="\t"))

            self.assertEqual(len(rows), 2)
            self.assertTrue(all(Path(row["albedo"]) == albedo.resolve() for row in rows))
            self.assertTrue(all(Path(row["normal"]) == normal.resolve() for row in rows))
            self.assertTrue(all(row["normal_x_channel"] == "3" for row in rows))
            self.assertTrue(all(row["normal_y_channel"] == "1" for row in rows))
            self.assertTrue(all(row["mtl1_r"] == "1" and row["mtl1_g"] == "1" and row["mtl1_b"] == "1" for row in rows))
            self.assertTrue(all(row["baseline_complete"] == "0" for row in rows))

            report = json.loads((root / "ship.materials.report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fallback"], "real-extracted-textures")
            self.assertEqual(report["textures"]["albedo"], str(albedo.resolve()))
            self.assertEqual(report["textures"]["normal"], str(normal.resolve()))


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

        source = self.read("tools/nsamdr/eve_asset_test.py")
        shader = self.read("trinityal/tests/nsamdr/NSAMDRPreview.hlsl")
        self.assertIn("exact-selected-legacy", source)
        self.assertIn('slots = [{**slot, "color": (1.0, 1.0, 1.0)} for slot in slots]', source)
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
        self.assertIn("draws_by_material_index.setdefault(material_index, []).append(draw)", eve_test)
        self.assertIn("matching_draws.extend(draws_by_material_index.get(material_index, []))", eve_test)

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
        self.assertIn("hull, faction, race, model_path", eve_test)

    def test_current_sof_schema_includes_shield_ellipsoid_flag(self) -> None:
        converter = self.read("tools/nsamdr/gr2_converter/convert_eve_asset.mjs")
        self.assertIn('ensureBlackClass("EveSOFDataHullExtensionPlacement"', converter)
        self.assertIn("extendsShieldEllipsoid: r.boolean", converter)

    def test_preview_fails_closed_on_incomplete_sof_materials(self) -> None:
        eve_test = self.read("tools/nsamdr/eve_asset_test.py")
        self.assertIn("refusing to render an invented material fallback", eve_test)
        self.assertIn("preview launch is blocked rather than", eve_test)
        self.assertIn("SOF material baseline remains incomplete", eve_test)
        self.assertIn('conversion_record.get("schema") != "NSAMDR_GR2_CONVERSION_V5_BAKED_EVE_TEXTURE_V"', eve_test)

    def test_readme_documents_v4_and_mandatory_retraining(self) -> None:
        readme = self.read("trinityal/tests/nsamdr/README.md")
        self.assertTrue(readme.startswith("# NSAMDR V5.34"))
        for phrase in (
            "## Quick start",
            "mandatory",
            "NSAMDR_TILE_CONTEXT_MATERIAL_V4",
            "125-pixel receptive field",
            "overlapping 512×512 tiles",
            "64-pixel overlap",
            "old V3 checkpoint is incompatible",
            "Mode 1 — original source, no cleanup",
            "Mode 3 — tile-context cleanup",
            "same mesh, camera, lighting, environment and shader",
            "No runtime neural compute shader",
            "window and taskbar icon",
        ):
            self.assertIn(phrase, readme)
        self.assertNotIn("NSAMDRNeuralWeights.hlsli", readme)
        self.assertNotIn("NSAMDRNeuralRuntime", readme)
        self.assertNotIn("13×13", readme)

    def test_manifest_contains_only_mode3_candidate(self) -> None:
        from tools.nsamdr.strategy_pipeline.model import CandidateArtifact, StrategyManifest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            obj = root / "ship.obj"
            materials = root / "ship.materials.tsv"
            obj.write_text("", encoding="utf-8")
            materials.write_text("", encoding="utf-8")
            manifest = StrategyManifest(
                "NSAMDR_THREE_MODE_PIPELINE_V2_TILE_CONTEXT", 4096, obj, materials)
            manifest.add(CandidateArtifact(3, "NSAMDR cleanup", obj, materials, metadata={"offlineTileContext": True}))
            report = manifest.to_report()
            self.assertEqual(report["schema"], "NSAMDR_THREE_MODE_PIPELINE_V2_TILE_CONTEXT")
            self.assertEqual(set(report["strategies"]), {"3"})
            self.assertTrue(report["strategies"]["3"]["offlineTileContext"])

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
