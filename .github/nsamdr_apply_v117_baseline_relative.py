from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one exact match, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str) -> None:
    text = read(path)
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"{path}: regex expected one match, found {count}: {pattern[:120]!r}")
    write(path, updated)


# ---------------------------------------------------------------------------
# 1) Inference exposes baseline and stage-relative production outputs.
# ---------------------------------------------------------------------------
path = "tools/nsamdr/neural/v9/inference.py"
replace_once(
    path,
    '''        return_diagnostics: bool = False,\n        return_all_maps: bool = False,\n    ):''',
    '''        return_diagnostics: bool = False,\n        return_all_maps: bool = False,\n        output_variant: str = "final",\n    ):''',
)
replace_once(
    path,
    '''        tensor = tensor.to(device=device, dtype=torch.float32).unsqueeze(0)\n        if device.type == "cuda" and model.config.channels_last:\n            tensor = tensor.contiguous(memory_format=torch.channels_last)\n        _, _, input_height, input_width = tensor.shape\n        tile_size = max(64, min(int(tile_size), max(input_height, input_width)))''',
    '''        tensor = tensor.to(device=device, dtype=torch.float32).unsqueeze(0)\n        if device.type == "cuda" and model.config.channels_last:\n            tensor = tensor.contiguous(memory_format=torch.channels_last)\n        _, _, input_height, input_width = tensor.shape\n        variant = str(output_variant).strip().casefold()\n        allowed_variants = {"final", "baseline", "structural", "seam", "detail"}\n        if variant not in allowed_variants:\n            raise ValueError(\n                f"unsupported NSAMDR output_variant={output_variant!r}; "\n                f"expected one of {sorted(allowed_variants)}"\n            )\n\n        # Baseline is a first-class production comparator, not a neural output.\n        # Keep this branch independent of model.forward() so live A/B/C preview can\n        # render the exact deterministic 4x control without stealing CUDA time from\n        # training or accidentally exposing an untrained downstream head.\n        if variant == "baseline":\n            source_albedo = tensor[:, 0:3].clamp(0.0, 1.0)\n            source_normal = tensor[:, 3:5].clamp(-1.0, 1.0)\n            source_material = tensor[:, 5:8].clamp(0.0, 1.0)\n            baseline_albedo = F.interpolate(\n                source_albedo, scale_factor=UPSCALE_FACTOR, mode="bicubic",\n                align_corners=False, antialias=True,\n            ).clamp(0.0, 1.0)\n            baseline_normal = F.interpolate(\n                source_normal, scale_factor=UPSCALE_FACTOR, mode="bilinear",\n                align_corners=False,\n            )\n            normal_length = torch.sqrt(\n                baseline_normal.square().sum(dim=1, keepdim=True) + 1.0e-8\n            )\n            baseline_normal = baseline_normal / torch.maximum(\n                torch.ones_like(normal_length), normal_length / 0.999\n            )\n            baseline_material = F.interpolate(\n                source_material, scale_factor=UPSCALE_FACTOR, mode="nearest"\n            ).clamp(0.0, 1.0)\n            h_hr, w_hr = baseline_albedo.shape[-2:]\n            zeros1 = torch.zeros((1, 1, h_hr, w_hr), device=device, dtype=torch.float32)\n            zeros2 = torch.zeros((1, 2, h_hr, w_hr), device=device, dtype=torch.float32)\n            maps = {\n                "albedo": baseline_albedo[0].permute(1, 2, 0).cpu().numpy(),\n                "normal_xy": baseline_normal[0].permute(1, 2, 0).cpu().numpy(),\n                "roughness": baseline_material[:, 2:3][0].permute(1, 2, 0).cpu().numpy(),\n                "emissive": baseline_material[:, 1:2][0].permute(1, 2, 0).cpu().numpy(),\n                "material": baseline_material[0].permute(1, 2, 0).cpu().numpy(),\n                "sdf": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "orientation": zeros2[0].permute(1, 2, 0).cpu().numpy(),\n                "confidence": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "edge": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "hardness": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "boundary_gate": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "coarse_sdf": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n                "contour_normal_offset_pixels": zeros1[0].permute(1, 2, 0).cpu().numpy(),\n            }\n            maps["sdf_residual_pixels"] = np.zeros_like(\n                maps["contour_normal_offset_pixels"], dtype=np.float32\n            )\n            maps["displacement"] = np.zeros((h_hr, w_hr, 2), dtype=np.float32)\n            maps["displacement_gate"] = maps["boundary_gate"]\n            diagnostics = {\n                "schema": MODEL_SCHEMA,\n                "upscaleFactor": UPSCALE_FACTOR,\n                "inputSize": [input_width, input_height],\n                "outputSize": [w_hr, h_hr],\n                "tileSize": 0,\n                "overlap": 0,\n                "tileCount": 0,\n                "inferenceMilliseconds": 0.0,\n                "device": str(device),\n                "precision": "fp32-baseline",\n                "outputVariant": "baseline",\n                "candidateAuthority": "deterministic-4x-baseline",\n                "productionForward": "baseline interpolation only; model.forward not called",\n            }\n            if not return_all_maps:\n                maps = {key: maps[key] for key in ("albedo", "normal_xy", "material")}\n            if return_diagnostics:\n                return maps, diagnostics\n            return maps\n\n        tile_size = max(64, min(int(tile_size), max(input_height, input_width)))''',
)
replace_once(
    path,
    '''                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):\n                    output = model(tile)\n                packed = torch.cat((\n                    output["albedo"], output["normal_xy"], output["roughness"], output["emissive"],\n                    output["material"], output["sdf"], output["orientation"], output["confidence"],''',
    '''                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):\n                    output = model(tile)\n                if variant == "structural":\n                    selected_albedo = output["boundary_pre_seam_albedo"]\n                    selected_normal = output["boundary_pre_seam_normal"]\n                    selected_material = output["boundary_pre_seam_material"]\n                elif variant == "seam":\n                    selected_albedo = output["boundary_reconstructed_albedo"]\n                    selected_normal = output["boundary_reconstructed_normal"]\n                    selected_material = output["boundary_reconstructed_material"]\n                elif variant == "detail":\n                    selected_albedo = output["detail_candidate_albedo"]\n                    selected_normal = output["detail_candidate_normal"]\n                    selected_material = output["detail_candidate_material"]\n                else:\n                    selected_albedo = output["albedo"]\n                    selected_normal = output["normal_xy"]\n                    selected_material = output["material"]\n                packed = torch.cat((\n                    selected_albedo, selected_normal,\n                    selected_material[:, 2:3], selected_material[:, 1:2],\n                    selected_material, output["sdf"], output["orientation"], output["confidence"],''',
)
replace_once(
    path,
    '''            "productionForward": "FidelityResidualNetV9.forward(inputs)",\n            "externalGeometryAuthority": False,''',
    '''            "productionForward": "FidelityResidualNetV9.forward(inputs)",\n            "outputVariant": variant,\n            "externalGeometryAuthority": False,''',
)

# Candidate generator passes stage selection through without changing final defaults.
path = "tools/nsamdr/generate_strategy_candidates.py"
replace_once(
    path,
    '''        out_width: int,\n        out_height: int,\n    ) -> tuple[dict[str, Any], dict[str, Any]]:''',
    '''        out_width: int,\n        out_height: int,\n        output_variant: str = "final",\n    ) -> tuple[dict[str, Any], dict[str, Any]]:''',
)
replace_once(
    path,
    '''            return_diagnostics=True,\n            return_all_maps=True,\n        )''',
    '''            return_diagnostics=True,\n            return_all_maps=True,\n            output_variant=output_variant,\n        )''',
)
replace_once(
    path,
    '''                "candidateAuthority": "direct-production-forward",''',
    '''                "candidateAuthority": f"direct-production-forward:{output_variant}",''',
)

# ---------------------------------------------------------------------------
# 2) Live worker publishes baseline and actual current-stage candidates together.
# ---------------------------------------------------------------------------
path = "tools/nsamdr/neural/live_preview_nsamdr_v9_training.py"
pattern = r'    def _generate_candidate\(.*?\n    def _candidate_pointer_text\('
replacement = r'''    def _stage_output_variant(self, phase: str) -> str:
        phase = str(phase).strip().casefold()
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            return "structural"
        if phase in {"seam-proof", "seam-authority", "gate-proof"}:
            return "seam"
        if phase == "detail-reconstruction":
            return "detail"
        return "final"

    def _generate_candidate(
        self,
        *,
        root: Path,
        directory: Path,
        checkpoint: Path,
        epoch: int,
        phase: str,
        target_size: int,
        requested_device: str,
        obj_path: Path,
        materials: Path,
        asset_manifest: Path,
    ) -> dict[str, Any]:
        checkpoint = checkpoint.resolve()
        checkpoint_sha = self._sha256(checkpoint)
        config = load_resolved_config(root, directory.name)
        device = resolve_device(config, requested_device)
        model = self._load_epoch_model(checkpoint, config, device, epoch)
        helper = StrategyCandidateGenerator()
        fields, rows, comments = helper._read_tsv(materials)
        usages = helper._collect_usages(rows, materials.parent)
        contexts = helper._collect_contexts(rows, materials.parent)
        if not contexts:
            raise RuntimeError("live EVE material manifest contains no albedo contexts")
        source_before = helper._source_snapshot(usages.keys())
        stage_variant = self._stage_output_variant(phase)

        output_root = directory / "previews" / "live" / "candidates" / f"epoch_{epoch:04d}"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        stage_canvases: dict[Path, Any] = {}
        baseline_canvases: dict[Path, Any] = {}
        stage_semantics: dict[Path, set[str]] = {}
        baseline_semantics: dict[Path, set[str]] = {}
        inference_records: list[dict[str, Any]] = []
        for index, (albedo, context) in enumerate(sorted(contexts.items(), key=lambda item: str(item[0]).casefold()), start=1):
            source = helper._read_bgra(albedo)
            width, height = helper._output_size(source, target_size)
            baseline_maps, baseline_diagnostics = helper._direct_maps(
                albedo=albedo,
                context=context,
                model=model,
                config=config,
                device=device,
                out_width=width,
                out_height=height,
                output_variant="baseline",
            )
            stage_maps, stage_diagnostics = helper._direct_maps(
                albedo=albedo,
                context=context,
                model=model,
                config=config,
                device=device,
                out_width=width,
                out_height=height,
                output_variant=stage_variant,
            )
            helper._apply_direct_maps(
                albedo=albedo,
                context=context,
                maps=baseline_maps,
                width=width,
                height=height,
                canvases=baseline_canvases,
                semantics=baseline_semantics,
            )
            helper._apply_direct_maps(
                albedo=albedo,
                context=context,
                maps=stage_maps,
                width=width,
                height=height,
                canvases=stage_canvases,
                semantics=stage_semantics,
            )
            inference_records.append({
                "source": str(albedo),
                "outputSize": [int(width), int(height)],
                "baselineDiagnostics": baseline_diagnostics,
                "stageVariant": stage_variant,
                "stageDiagnostics": stage_diagnostics,
            })
            print(
                f"[live-preview] epoch {epoch} [{index}/{len(contexts)}] {albedo.name} "
                f"-> {width}x{height} baseline + {stage_variant}",
                flush=True,
            )

        def write_canvases(canvases, folder: str, suffix: str) -> dict[Path, Path]:
            texture_dir = output_root / folder
            replacements: dict[Path, Path] = {}
            for source, canvas in sorted(canvases.items(), key=lambda item: str(item[0]).casefold()):
                token = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:10]
                destination = texture_dir / f"{source.stem}_{token}_{suffix}.png"
                helper._write_png(destination, canvas)
                replacements[source] = destination.resolve()
            return replacements

        baseline_replacements = write_canvases(
            baseline_canvases, "baseline_4x", "baseline_4x"
        )
        stage_replacements = write_canvases(
            stage_canvases, "live_nsamdr", f"nsamdr_{stage_variant}"
        )

        baseline_materials = output_root / "baseline.materials.tsv"
        self._write_material_manifest(
            helper, baseline_materials, fields, rows, comments,
            baseline_replacements, materials.parent,
        )
        candidate_materials = output_root / "live.materials.tsv"
        self._write_material_manifest(
            helper, candidate_materials, fields, rows, comments,
            stage_replacements, materials.parent,
        )
        candidate_obj = output_root / obj_path.name
        shutil.copy2(obj_path, candidate_obj)
        provenance = helper._provenance(
            source_before=source_before,
            replacements=stage_replacements,
            usages=usages,
            material_manifest=materials,
            asset_manifest=asset_manifest,
        )
        provenance_path = output_root / "live_control_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        analysis_path = output_root / "live_candidate_analysis.json"
        analysis_path.write_text(
            json.dumps({
                "schema": "NSAMDR_LIVE_TRAINING_CANDIDATE_ANALYSIS_V2",
                "authority": "training-intermediate",
                "qualified": False,
                "epoch": int(epoch),
                "phase": phase,
                "stageVariant": stage_variant,
                "baselineVariant": "baseline",
                "checkpoint": str(checkpoint),
                "checkpointSha256": checkpoint_sha,
                "modelSchema": MODEL_SCHEMA,
                "productionForward": "FidelityResidualNetV9.forward(inputs)",
                "baselineForward": "deterministic production 4x baseline; no model.forward",
                "inference": inference_records,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema": "NSAMDR_LIVE_TRAINING_CANDIDATE_V2",
            "status": "live-preview-ready",
            "authority": "training-intermediate",
            "qualified": False,
            "epoch": int(epoch),
            "phase": phase,
            "stageVariant": stage_variant,
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_sha,
            "candidateObj": str(candidate_obj.resolve()),
            "baselineObj": str(candidate_obj.resolve()),
            "baselineMaterials": str(baseline_materials.resolve()),
            "candidateMaterials": str(candidate_materials.resolve()),
            "candidateAnalysis": str(analysis_path.resolve()),
            "controlProvenance": provenance,
            "controlProvenancePath": str(provenance_path.resolve()),
            "targetSize": int(target_size),
            "inferenceDevice": str(device),
        }
        report_path = output_root / "candidate_manifest.json"
        report["reportPath"] = str(report_path.resolve())
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self._sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("completed live checkpoint changed during candidate generation")
        return report

    def _candidate_pointer_text('''
regex_once(path, pattern, replacement)
replace_once(
    path,
    '''            f"candidateObj={report['candidateObj']}",\n            f"candidateMaterials={report['candidateMaterials']}",''',
    '''            f"stageVariant={report['stageVariant']}",\n            f"baselineObj={report['baselineObj']}",\n            f"baselineMaterials={report['baselineMaterials']}",\n            f"candidateObj={report['candidateObj']}",\n            f"candidateMaterials={report['candidateMaterials']}",''',
)

# ---------------------------------------------------------------------------
# 3) Native live preview becomes A authored / B deterministic baseline / C stage.
# ---------------------------------------------------------------------------
path = "trinityal/tests/nsamdr/NSAMDRPreviewTypes.h"
replace_once(
    path,
    '''struct FinalCandidateSet\n{\n    CandidateAssetGpu candidate;\n};''',
    '''struct FinalCandidateSet\n{\n    CandidateAssetGpu baseline;\n    CandidateAssetGpu candidate;\n};''',
)

path = "trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp"
regex_once(
    path,
    r'struct LiveCandidatePointer\n\{.*?\n\}\n\nbool ReadLiveCandidatePointer\(.*?\n\}\n\nbool ValidationPassed',
    '''struct LiveCandidatePointer
{
    std::string token;
    std::string epoch;
    std::string phase;
    std::string stageVariant;
    std::string checkpointSha;
    std::string baselineObj;
    std::string baselineMaterials;
    std::string candidateObj;
    std::string candidateMaterials;
};

bool ReadLiveCandidatePointer(const std::string& path, LiveCandidatePointer& pointer)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line)) return false;
    if (!line.empty() && line.back() == '\\r') line.pop_back();
    if (line != "NSAMDR_LIVE_CANDIDATE_POINTER_V1") return false;
    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\\r') line.pop_back();
        const size_t split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1U);
        if (key == "token") pointer.token = value;
        else if (key == "epoch") pointer.epoch = value;
        else if (key == "phase") pointer.phase = value;
        else if (key == "stageVariant") pointer.stageVariant = value;
        else if (key == "checkpointSha256") pointer.checkpointSha = value;
        else if (key == "baselineObj") pointer.baselineObj = value;
        else if (key == "baselineMaterials") pointer.baselineMaterials = value;
        else if (key == "candidateObj") pointer.candidateObj = value;
        else if (key == "candidateMaterials") pointer.candidateMaterials = value;
    }
    return !pointer.token.empty() && !pointer.baselineObj.empty() &&
        !pointer.baselineMaterials.empty() && !pointer.candidateObj.empty() &&
        !pointer.candidateMaterials.empty();
}

bool ValidationPassed''',
)
regex_once(
    path,
    r'bool PreviewProcessing::RefreshLiveCandidate\(.*?\n\}\n\nbool PreviewProcessing::InitializeState',
    '''bool PreviewProcessing::RefreshLiveCandidate(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const PreviewResources& resources,
    const std::string& rawAlbedoPath,
    FinalCandidateSet& candidates)
{
    (void)rawAlbedoPath;
    if (ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) != "training-intermediate")
        return true;
    const std::string pointerPath = GetEnvironmentString("NSAMDR_LIVE_CANDIDATE_POINTER");
    if (pointerPath.empty()) return true;

    LiveCandidatePointer pointer;
    if (!ReadLiveCandidatePointer(pointerPath, pointer)) return true;
    if (pointer.token == m_liveCandidateToken) return true;

    CandidateAssetGpu nextBaseline;
    if (!m_assetProcessor.LoadCandidateAsset(
            device,
            context,
            "4X DETERMINISTIC BASELINE",
            pointer.baselineObj,
            pointer.baselineMaterials,
            nextBaseline))
    {
        std::printf("NSAMDR live preview: baseline GPU load failed for token %s\\n", pointer.token.c_str());
        return true;
    }
    if (!nextBaseline.available || !CandidateUsesSourceDrawRanges(resources, nextBaseline))
    {
        std::printf("NSAMDR live preview: rejected baseline for token %s\\n", pointer.token.c_str());
        return true;
    }

    CandidateAssetGpu nextCandidate;
    const std::string stage = pointer.stageVariant.empty() ? pointer.phase : pointer.stageVariant;
    const std::string label = "NSAMDR LIVE epoch " + pointer.epoch + " | " + stage;
    if (!m_assetProcessor.LoadCandidateAsset(
            device,
            context,
            label,
            pointer.candidateObj,
            pointer.candidateMaterials,
            nextCandidate))
    {
        std::printf("NSAMDR live preview: candidate GPU load failed for token %s\\n", pointer.token.c_str());
        return true;
    }
    if (!nextCandidate.available)
    {
        std::printf("NSAMDR live preview: candidate unavailable for token %s: %s\\n",
            pointer.token.c_str(), nextCandidate.status.c_str());
        return true;
    }
    if (!CandidateUsesSourceDrawRanges(resources, nextCandidate))
    {
        std::printf("NSAMDR live preview: rejected token %s because draw ranges differ from source\\n",
            pointer.token.c_str());
        return true;
    }

    nextBaseline.status = "DETERMINISTIC 4X BASELINE | same degraded LR evidence";
    nextCandidate.status = "UNQUALIFIED INTERMEDIATE | epoch=" + pointer.epoch +
        " | phase=" + pointer.phase + " | variant=" + stage + " | checkpoint=" +
        (pointer.checkpointSha.empty() ? std::string("unknown") : pointer.checkpointSha.substr(0U, 12U));
    candidates.baseline = std::move(nextBaseline);
    candidates.candidate = std::move(nextCandidate);
    m_liveCandidateToken = pointer.token;
    std::printf("NSAMDR live preview: hot-reloaded baseline + %s\\n", candidates.candidate.status.c_str());
    return true;
}

bool PreviewProcessing::InitializeState''',
)

path = "trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp"
regex_once(
    path,
    r'    const AssetBinding baselineAsset\{.*?    const float blendFactor\[4\] = \{0\.0f, 0\.0f, 0\.0f, 0\.0f\};',
    '''    const AssetBinding baselineAsset{
        resources.vertexBuffer.Get(),
        resources.indexBuffer.Get(),
        resources.indexCount,
        &resources.areaMaterials,
        true,
        false,
        true,
    };

    const CandidateAssetGpu& deterministicBaseline = candidates.baseline;
    const AssetBinding deterministicBaselineAsset = deterministicBaseline.available
        ? AssetBinding{
            baselineAsset.vertexBuffer,
            baselineAsset.indexBuffer,
            baselineAsset.indexCount,
            &deterministicBaseline.areaMaterials,
            false,
            true,
            true,
        }
        : AssetBinding{nullptr, nullptr, 0U, nullptr, false, false, false};

    const CandidateAssetGpu& finalCandidate = candidates.candidate;
    const AssetBinding finalAsset = finalCandidate.available
        ? AssetBinding{
            baselineAsset.vertexBuffer,
            baselineAsset.indexBuffer,
            baselineAsset.indexCount,
            &finalCandidate.areaMaterials,
            false,
            true,
            true,
        }
        : AssetBinding{nullptr, nullptr, 0U, nullptr, false, false, false};

    const bool threeWay = deterministicBaseline.available;
    PaneRect rawControlPane{sceneX, 0U, sceneWidth, sceneHeight};
    PaneRect deterministicBaselinePane = rawControlPane;
    PaneRect candidatePane = rawControlPane;
    if (threeWay)
    {
        if (state.splitVertical)
        {
            const uint32_t firstWidth = std::max(1U, sceneWidth / 3U);
            const uint32_t secondWidth = std::max(1U, (sceneWidth - firstWidth) / 2U);
            rawControlPane = PaneRect{sceneX, 0U, firstWidth, sceneHeight};
            deterministicBaselinePane = PaneRect{sceneX + firstWidth, 0U, secondWidth, sceneHeight};
            candidatePane = PaneRect{
                sceneX + firstWidth + secondWidth, 0U,
                sceneWidth - firstWidth - secondWidth, sceneHeight};
        }
        else
        {
            const uint32_t firstHeight = std::max(1U, sceneHeight / 3U);
            const uint32_t secondHeight = std::max(1U, (sceneHeight - firstHeight) / 2U);
            rawControlPane = PaneRect{sceneX, 0U, sceneWidth, firstHeight};
            deterministicBaselinePane = PaneRect{sceneX, firstHeight, sceneWidth, secondHeight};
            candidatePane = PaneRect{
                sceneX, firstHeight + secondHeight, sceneWidth,
                sceneHeight - firstHeight - secondHeight};
        }
        if (state.swapSplitSides)
            std::swap(rawControlPane, candidatePane);
    }
    else
    {
        if (state.splitVertical)
        {
            const uint32_t firstWidth = std::max(1U, sceneWidth / 2U);
            rawControlPane = PaneRect{sceneX, 0U, firstWidth, sceneHeight};
            candidatePane = PaneRect{sceneX + firstWidth, 0U, sceneWidth - firstWidth, sceneHeight};
        }
        else
        {
            const uint32_t firstHeight = std::max(1U, sceneHeight / 2U);
            rawControlPane = PaneRect{sceneX, 0U, sceneWidth, firstHeight};
            candidatePane = PaneRect{sceneX, firstHeight, sceneWidth, sceneHeight - firstHeight};
        }
        if (state.swapSplitSides)
            std::swap(rawControlPane, candidatePane);
    }
    const float blendFactor[4] = {0.0f, 0.0f, 0.0f, 0.0f};''',
)
replace_once(
    path,
    '''    // B: immutable NSAMDR FINAL material resources on the source mesh.\n    drawBackgroundPane(candidatePane);\n    drawPane(candidatePane, finalAsset);\n    context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);\n\n    // A: authoritative raw source under the exact same draw path.\n    drawBackgroundPane(rawControlPane);\n    drawPane(rawControlPane, baselineAsset);''',
    '''    // C: current learned stage (or immutable final outside live mode).\n    drawBackgroundPane(candidatePane);\n    drawPane(candidatePane, finalAsset);\n    context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);\n\n    // B: exact deterministic 4x reconstruction baseline from the same LR evidence.\n    if (threeWay)\n    {\n        drawBackgroundPane(deterministicBaselinePane);\n        drawPane(deterministicBaselinePane, deterministicBaselineAsset);\n        context->ClearDepthStencilView(resources.depthStencilView.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);\n    }\n\n    // A: authoritative authored source under the exact same draw path.\n    drawBackgroundPane(rawControlPane);\n    drawPane(rawControlPane, baselineAsset);''',
)

path = "trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp"
replace_once(
    path,
    '''            ? "Live training comparison: A RAW SOURCE and B CURRENT NSAMDR EPOCH. Both panes keep the same EVE mesh, camera, lighting, background, material shader and sampler while only B hot-reloads after a completed epoch."''',
    '''            ? "Live training comparison: A AUTHORED SOURCE, B DETERMINISTIC 4X BASELINE, and C CURRENT TRAINED STAGE. All panes keep the same EVE mesh, camera, lighting, background, material shader and sampler."''',
)
replace_once(
    path,
    '''            ? "A RAW SOURCE stays fixed while B CURRENT NSAMDR EPOCH hot-reloads. Camera, transform, lighting, EVE environment, shader and sampler remain identical, so visual changes on the right are training changes."''',
    '''            ? "A AUTHORED SOURCE stays fixed. B is the deterministic 4x baseline from the same degraded LR evidence. C is the current stage output and hot-reloads. Training is useful only when C improves on B while moving toward A."''',
)
regex_once(
    path,
    r'void PreviewPanel::DrawSplitCompareOverlay\(.*?\n\}\n\nstd::string PreviewPanel::BuildScreenshotPath\(\) const',
    '''void PreviewPanel::DrawSplitCompareOverlay(
    const PreviewState& state,
    const PreviewResources& resources,
    const FinalCandidateSet& candidates)
{
    const float sceneX = static_cast<float>(std::min(
        state.sceneViewportX,
        resources.width > 1U ? resources.width - 1U : 0U));
    const float sceneWidth = std::max(1.0f, static_cast<float>(resources.width) - sceneX);
    const float sceneHeight = std::max(1.0f, static_cast<float>(resources.height));
    ImDrawList* draw = ImGui::GetForegroundDrawList();
    const bool liveTrainingPreview =
        ReadEnvironmentVariable("NSAMDR_PREVIEW_AUTHORITY") == "training-intermediate";
    const bool threeWay = liveTrainingPreview && candidates.baseline.available;

    auto addLabel = [&](const ImVec2& paneMin, const std::string& label)
    {
        const ImVec2 textPos(paneMin.x + 12.0f, paneMin.y + 12.0f);
        const ImVec2 textSize = ImGui::CalcTextSize(label.c_str());
        draw->AddRectFilled(
            ImVec2(textPos.x - 6.0f, textPos.y - 4.0f),
            ImVec2(textPos.x + textSize.x + 6.0f, textPos.y + textSize.y + 4.0f),
            IM_COL32(0, 0, 0, 190), 4.0f);
        draw->AddText(textPos, IM_COL32(255, 255, 255, 255), label.c_str());
    };

    struct Pane { ImVec2 min; ImVec2 max; };
    Pane raw{{sceneX, 0.0f}, {sceneX + sceneWidth, sceneHeight}};
    Pane baseline = raw;
    Pane candidate = raw;
    if (threeWay)
    {
        if (state.splitVertical)
        {
            const float d1 = sceneX + std::floor(sceneWidth / 3.0f);
            const float d2 = sceneX + std::floor(sceneWidth * 2.0f / 3.0f);
            raw = {{sceneX, 0.0f}, {d1, sceneHeight}};
            baseline = {{d1, 0.0f}, {d2, sceneHeight}};
            candidate = {{d2, 0.0f}, {sceneX + sceneWidth, sceneHeight}};
            draw->AddLine({d1, 0.0f}, {d1, sceneHeight}, IM_COL32(255,255,255,210), 2.0f);
            draw->AddLine({d2, 0.0f}, {d2, sceneHeight}, IM_COL32(255,255,255,210), 2.0f);
        }
        else
        {
            const float d1 = std::floor(sceneHeight / 3.0f);
            const float d2 = std::floor(sceneHeight * 2.0f / 3.0f);
            raw = {{sceneX, 0.0f}, {sceneX + sceneWidth, d1}};
            baseline = {{sceneX, d1}, {sceneX + sceneWidth, d2}};
            candidate = {{sceneX, d2}, {sceneX + sceneWidth, sceneHeight}};
            draw->AddLine({sceneX, d1}, {sceneX + sceneWidth, d1}, IM_COL32(255,255,255,210), 2.0f);
            draw->AddLine({sceneX, d2}, {sceneX + sceneWidth, d2}, IM_COL32(255,255,255,210), 2.0f);
        }
        if (state.swapSplitSides) std::swap(raw, candidate);
        addLabel(raw.min, "A AUTHORED SOURCE - 16x AF / LOD 0");
        addLabel(baseline.min, "B 4X BASELINE - SAME LR EVIDENCE");
        addLabel(candidate.min, "C NSAMDR LIVE STAGE - UNQUALIFIED");
    }
    else
    {
        if (state.splitVertical)
        {
            const float divider = sceneX + std::floor(sceneWidth * 0.5f);
            raw = {{sceneX, 0.0f}, {divider, sceneHeight}};
            candidate = {{divider, 0.0f}, {sceneX + sceneWidth, sceneHeight}};
            draw->AddLine({divider, 0.0f}, {divider, sceneHeight}, IM_COL32(255,255,255,210), 2.0f);
        }
        else
        {
            const float divider = std::floor(sceneHeight * 0.5f);
            raw = {{sceneX, 0.0f}, {sceneX + sceneWidth, divider}};
            candidate = {{sceneX, divider}, {sceneX + sceneWidth, sceneHeight}};
            draw->AddLine({sceneX, divider}, {sceneX + sceneWidth, divider}, IM_COL32(255,255,255,210), 2.0f);
        }
        if (state.swapSplitSides) std::swap(raw, candidate);
        addLabel(raw.min, liveTrainingPreview
            ? "A AUTHORED SOURCE - 16x AF / LOD 0"
            : "A RAW SOURCE - 16x AF / LOD 0");
        addLabel(candidate.min, liveTrainingPreview
            ? "C NSAMDR LIVE STAGE - UNQUALIFIED"
            : "B NSAMDR FINAL - 16x AF / LOD 0");
    }

    const CandidateAssetGpu& finalCandidate = candidates.candidate;
    if (!finalCandidate.available)
    {
        const char* warning = liveTrainingPreview
            ? "C WAITING - no completed stage candidate"
            : "B BLOCKED - immutable final provenance not verified";
        const ImVec2 warningPos(candidate.min.x + 12.0f, candidate.min.y + 48.0f);
        const ImVec2 warningSize = ImGui::CalcTextSize(warning);
        draw->AddRectFilled(candidate.min, candidate.max, IM_COL32(80, 0, 0, 78));
        draw->AddRectFilled(
            ImVec2(warningPos.x - 6.0f, warningPos.y - 4.0f),
            ImVec2(warningPos.x + warningSize.x + 6.0f, warningPos.y + warningSize.y + 4.0f),
            IM_COL32(130, 0, 0, 230), 4.0f);
        draw->AddText(warningPos, IM_COL32(255, 235, 235, 255), warning);
    }
}

std::string PreviewPanel::BuildScreenshotPath() const''',
)

# ---------------------------------------------------------------------------
# 4) Quick gets a small first B1b smoke pass before paying for the full bank.
# ---------------------------------------------------------------------------
path = "tools/nsamdr/neural/v9/training.py"
replace_once(
    path,
    '''            epoch_batch_size = max(1, int(epoch_loader.batch_size or 1))\n            epoch_batch_count = len(epoch_loader) * seam_proof_passes\n            epoch_tile_count = len(epoch_loader.dataset) * seam_proof_passes\n            epoch_workers = workers''',
    '''            epoch_batch_size = max(1, int(epoch_loader.batch_size or 1))\n            epoch_batch_count = len(epoch_loader) * seam_proof_passes\n            epoch_tile_count = len(epoch_loader.dataset) * seam_proof_passes\n            # Quick's first connected-spline geometry epoch is a complete-class\n            # smoke proof, not a promotion epoch. Two examples per primitive family\n            # give an early A/B/C result before paying for the full 70-tile bank.\n            # Full runs and every later B1b epoch retain the complete training bank.\n            structural_smoke_batch_limit = None\n            if (\n                local_structure_phase\n                and b1b_stage_epoch == 1\n                and int(config.tiles_per_epoch) <= 64\n            ):\n                structural_smoke_batch_limit = min(\n                    epoch_batch_count, max(PRIMITIVE_COUNT * 2, PRIMITIVE_COUNT)\n                )\n                epoch_batch_count = structural_smoke_batch_limit\n                epoch_tile_count = structural_smoke_batch_limit * epoch_batch_size\n                self._status(\n                    f"  B1b QUICK SMOKE: {structural_smoke_batch_limit} batch(es) "\n                    "(2/class) before the full connected-spline bank."\n                )\n            epoch_workers = workers''',
)
replace_once(
    path,
    '''            for batch_index, batch in enumerate(train_batches, 1):\n                step_completed = False''',
    '''            for batch_index, batch in enumerate(train_batches, 1):\n                if (\n                    structural_smoke_batch_limit is not None\n                    and batch_index > structural_smoke_batch_limit\n                ):\n                    break\n                step_completed = False''',
)
replace_once(
    path,
    '''                "batches": len(epoch_loader),''',
    '''                "batches": int(epoch_batch_count),''',
)

# ---------------------------------------------------------------------------
# 5) Update/add regression contracts.
# ---------------------------------------------------------------------------
path = "tools/nsamdr/tests/test_live_eve_training_preview_contract.py"
text = read(path)
text = text.replace(
    'assert "A RAW SOURCE stays fixed while B CURRENT NSAMDR EPOCH hot-reloads" in panel',
    'assert "A AUTHORED SOURCE stays fixed" in panel\n    assert "B DETERMINISTIC 4X BASELINE" in panel\n    assert "C CURRENT TRAINED STAGE" in panel',
)
text = text.replace(
    'assert "NSAMDR_LIVE_CANDIDATE_POINTER_V1" in live',
    'assert "NSAMDR_LIVE_CANDIDATE_POINTER_V1" in live\n    assert "baselineMaterials" in live\n    assert "stageVariant" in live\n    assert "_stage_output_variant" in live',
)
write(path, text)

new_test = ROOT / "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"
new_test.write_text(r'''from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_baseline_variant_is_deterministic_and_does_not_call_model_forward():
    from v9.inference import infer_tiled

    class NeverForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                channels_last=False,
                amp_dtype="auto",
                appearance_enabled=True,
                detail_reconstruction_enabled=True,
            )

        def forward(self, _value):  # pragma: no cover - must never execute
            raise AssertionError("baseline variant called model.forward")

    value = np.zeros((17, 12, 10), dtype=np.float32)
    value[0:3] = 0.37
    value[3] = 0.25
    value[4] = -0.15
    value[5] = 0.2
    value[6] = 0.4
    value[7] = 0.7
    maps, diagnostics = infer_tiled(
        NeverForward(), value, "cpu", return_diagnostics=True,
        return_all_maps=True, output_variant="baseline",
    )
    assert maps["albedo"].shape == (48, 40, 3)
    assert np.allclose(maps["albedo"], 0.37, atol=1.0e-5)
    assert np.allclose(maps["material"][..., 0], 0.2, atol=1.0e-6)
    assert diagnostics["outputVariant"] == "baseline"
    assert diagnostics["tileCount"] == 0


def test_stage_variants_are_selected_before_final_selector():
    source = inspect.getsource(__import__("v9.inference", fromlist=["InferenceService"]).InferenceService.infer_tiled)
    assert 'variant == "structural"' in source
    assert 'output["boundary_pre_seam_albedo"]' in source
    assert 'variant == "seam"' in source
    assert 'output["boundary_reconstructed_albedo"]' in source
    assert 'variant == "detail"' in source
    assert 'output["detail_candidate_albedo"]' in source


def test_live_preview_is_authored_baseline_stage_not_raw_vs_final():
    live = text("tools/nsamdr/neural/live_preview_nsamdr_v9_training.py")
    types = text("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    processing = text("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")
    render = text("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    panel = text("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    assert 'return "structural"' in live
    assert 'output_variant="baseline"' in live
    assert 'output_variant=stage_variant' in live
    assert 'CandidateAssetGpu baseline;' in types
    assert 'pointer.baselineMaterials' in processing
    assert 'candidates.baseline = std::move(nextBaseline)' in processing
    assert 'const bool threeWay = deterministicBaseline.available;' in render
    assert 'A AUTHORED SOURCE' in panel
    assert 'B 4X BASELINE' in panel
    assert 'C NSAMDR LIVE STAGE' in panel


def test_quick_first_b1b_is_small_complete_class_smoke_only():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'PRIMITIVE_COUNT * 2' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'structural_smoke_batch_limit' in source
    # Later epochs/full runs still use the canonical complete loader.
    assert 'local_structure_train_loader' in source
''', encoding="utf-8")

# Short design note retained with the implementation so future revisions do not
# regress to full-frame replacement or confuse authored source with LR baseline.
doc = ROOT / "tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md"
doc.write_text('''# NSAMDR baseline-relative reconstruction contract\n\n## Non-negotiable comparison\n\nEvery learned stage is judged against the deterministic reconstruction available from the same degraded LR evidence.\n\n- **A — Authored source:** held-out HR target / real EVE authored texture.\n- **B — Deterministic 4x baseline:** bicubic albedo, normalized bilinear normal XY, nearest physical material channels.\n- **C — Current learned stage:** the stage actually being trained, before downstream selectors can hide it.\n\nTraining is useful only when C improves on B while moving toward A. Production-final remains a separate fail-closed authority.\n\n## Literature corrections carried into V11.7\n\n1. Residual SR systems (VDSR, LapSRN, SwinIR) preserve a low-frequency/interpolated path and learn the missing correction rather than forcing the network to repaint the full image. NSAMDR already had an internal baseline, but its proof/preview did not expose it as a first-class control.\n2. Deep Vectorization of Technical Drawings uses neural estimates as an initializer and then refines explicit geometric parameters. This remains the next structural escalation if the connected-spline learned proposal cannot beat B reliably.\n3. End-to-End Line Drawing Vectorization supports hard ordered connectivity: connectivity should be represented, not merely penalized.\n4. DiffVG supplies differentiable anti-aliased rasterization but does not solve discrete topology changes; topology remains an explicit NSAMDR responsibility.\n5. LIVE reinforces that low raster error alone is not a topology guarantee.\n6. Long smoothing B-splines support smoothing the parameterized curve itself, with corners/junctions exempted structurally rather than blurring output pixels.\n\n## Quick feedback contract\n\nThe first Quick B1b epoch is a two-examples-per-primitive smoke pass. It is not a promotion proof. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete training bank and all existing hard qualification gates.\n''', encoding="utf-8")

print("V11.7 baseline-relative patch applied")
