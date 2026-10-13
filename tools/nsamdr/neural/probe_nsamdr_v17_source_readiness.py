#!/usr/bin/env python3
"""V17.5 Stage 7: read-only A-C source, corpus and conditioning feasibility proof.

Never trains, rewrites source textures, alters an old checkpoint, or opens the
eight sealed V17.3 final-test authority pixels. Normal XY ordering is not
guessed from channel variance: orientation stays UNVERIFIED pending evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (HERE, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import eve_asset_test as eve
from authored_texture_dataset import AuthoredTextureDatasetRepository, TextureSource
from probe_nsamdr_v17_comprehensive import (
    _antialias_integral, _batch_config, _load_v3_origin, _origin, _records,
)
from probe_nsamdr_v17_sibling_architecture import _batch_for_record
from v14.qualification import sample_metrics
from v17.spatial_sampling import NativeAuthoritySampler, _id, _score_box

SCHEMA = "NSAMDR_V17_5_SOURCE_READINESS_V1"
DESTINATION = "artifacts/nsamdr/diagnostics/v17_source_readiness"
# Classification is for discovery only. It is NOT permission to reinterpret
# or mix maps: version-specific semantic profiles require external verification.
FORMATS = {
    "legacy": {"albedo": ("_d.dds", "_d.png"),
               "normal": ("_n.dds", "_n.png"),
               "material": ("_pgs.dds", "_pgs.png")},
    "modern": {"albedo": ("_ar.dds", "_ar.png"),
               "normal": ("_no.dds", "_no.png"),
               "material": ("_pmdg.dds", "_pmdg.png")},
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(logical: str) -> tuple[str, str, str] | None:
    name = str(logical).lower().replace("\\", "/")
    for generation, roles in FORMATS.items():
        for role, suffixes in roles.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    return generation, role, name[:-len(suffix)]
    return None


def _dds_signature(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"DDS "


def _stats(rgba: np.ndarray) -> dict[str, Any]:
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError("RGBA array required")
    return {key: {
        "min": int(rgba[..., i].min()), "max": int(rgba[..., i].max()),
        "std": float(rgba[..., i].astype(np.float32).std()),
        "distinct": int(np.unique(rgba[..., i]).size),
    } for i, key in enumerate(("red", "green", "blue", "alpha"))}


def _normal_hypotheses(rgba: np.ndarray) -> dict[str, Any]:
    """Observable geometry statistics; no channel layout is inferred."""
    planes = {name: rgba[..., channel].astype(np.float32)/127.5-1.0
              for channel, name in enumerate(("red", "green", "blue", "alpha"))}
    out = {}
    for label, (xname, yname) in {
        "legacy-green-alpha": ("green", "alpha"),
        "legacy-alpha-green": ("alpha", "green"),
        "rgb-red-green": ("red", "green"),
    }.items():
        xy2 = planes[xname]**2 + planes[yname]**2
        out[label] = {
            "outOfUnitDiskFraction": float(np.mean(xy2 > 1.0001)),
            "meanXYMagnitude": float(np.sqrt(xy2).mean()),
            "xChannel": xname, "yChannel": yname,
        }
    return out


def _rgba_native_if_supported(path: Path) -> tuple[np.ndarray | None, str]:
    try:
        with Image.open(path) as im:
            # Do not claim the native decoder independent if it happens to be
            # backed by the same library; source verification records methods.
            return np.asarray(im.convert("RGBA"), dtype=np.uint8), "Pillow-DDS"
    except (OSError, ValueError, EOFError, NotImplementedError) as exc:
        return None, f"unavailable: {type(exc).__name__}"


def _mip_evidence(source: Path, decoded_rgba: np.ndarray, mip_count: int) -> dict[str, Any]:
    """Check genuine hashed DDS mip level 2, if actually exposed by Pillow."""
    result: dict[str, Any] = {
        "headerMipCount": int(mip_count), "status": "UNAVAILABLE",
        "reason": "Native decoder does not expose aligned mip level 2",
        "actualMip2VsAreaMAE": None,
    }
    try:
        with Image.open(source) as image:
            result["decoderFrameCount"] = int(getattr(image, "n_frames", 1))
            if result["decoderFrameCount"] < 3 or mip_count < 3:
                return result
            image.seek(2)
            target_h, target_w = decoded_rgba.shape[0]//4, decoded_rgba.shape[1]//4
            candidate=np.asarray(image.convert("RGBA"),dtype=np.uint8)
            if candidate.shape[:2] != (target_h,target_w):
                result["reason"] = "Native mip dimensions do not match 4x"
                return result
            synthetic=cv2.resize(decoded_rgba, (target_w,target_h),
                                 interpolation=cv2.INTER_AREA)
            result.update({
                "status":"PASS", "reason":"Decoded DDS mip2 compared with area filter",
                "actualMip2VsAreaMAE":float(np.abs(
                    candidate.astype(np.int16)-synthetic.astype(np.int16)
                ).mean()/255),
            })
    except (OSError, ValueError, EOFError, NotImplementedError) as exc:
        result["reason"] = f"Native DDS mip decoder unavailable: {type(exc).__name__}"
    return result


def _audit_a(repo: Path, manifest: dict[str, Any],
             parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    indexed = {str(f.get("familyId") or f.get("family_id")): f
               for f in manifest.get("families", [])}
    loader = AuthoredTextureDatasetRepository()
    results, errors = [], []
    converter = None
    with tempfile.TemporaryDirectory(prefix="nsamdr_v175_decode_") as temporary:
        scratch = Path(temporary)
        for row in parts["training"]:
            authority = _id(row)
            family = indexed.get(authority)
            if not family:
                errors.append(f"{authority}: metadata unavailable")
                continue
            record: dict[str, Any] = {"authorityId": authority, "maps": {}}
            for role in ("albedo", "normal", "material"):
                meta = family.get(role)
                if not meta:
                    record["maps"][role] = {"status": "ABSENT"}
                    continue
                source = Path(str(meta["path"]))
                channel = {"logical": meta.get("logical"),
                           "filenameExtension": source.suffix.lower()}
                try:
                    if not source.is_file() or not _dds_signature(source):
                        raise RuntimeError("source missing or non-DDS signature")
                    width, height, mips, fmt = loader.parse_dds_header(source)
                    source_type = TextureSource(**meta)
                    png = scratch / f"{authority}_{role}.png"
                    converted = loader._decode_source(source_type, png, repo, converter)
                    with Image.open(converted) as im:
                        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
                    if rgba.shape[:2] != (height, width):
                        raise RuntimeError("decoded dimensions disagree with raw DDS header")
                    native, method = _rgba_native_if_supported(source)
                    mismatch = None
                    if native is not None and native.shape == rgba.shape:
                        mismatch = float(np.abs(
                            rgba.astype(np.int16)-native.astype(np.int16)
                        ).mean()/255.0)
                    channel.update({
                        "status": "PASS" if mismatch is None or mismatch <= (2/255)
                                  else "WARN",
                        "signature": "DDS", "headerFormat": fmt,
                        "headerMipCount": mips, "dimensions": [width, height],
                        "rawDDSsha256": _digest(source),
                        "convertedChannels": _stats(rgba),
                        "independentDecoder": method,
                        "converterVsPillowMAE": mismatch,
                        "sourceMipEvidence": _mip_evidence(source, rgba, mips),
                        "decoderEquivalence": "UNAVAILABLE" if mismatch is None
                                              else "PASS" if mismatch <= 2/255 else "WARN",
                    })
                    if role == "normal":
                        channel.update({
                            "channelHypotheses": _normal_hypotheses(rgba),
                            "normalOrientationVerified": False,
                            "semanticStatus": "UNVERIFIED",
                            "alphaVarying": bool(rgba[..., 3].std() > .75),
                        })
                    png.unlink(missing_ok=True)
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                    channel["status"] = "FAIL"
                    channel["error"] = f"{type(exc).__name__}: {exc}"
                    if role in ("albedo", "normal"):
                        errors.append(f"{authority}/{role}: {channel['error']}")
                record["maps"][role] = channel
            results.append(record)
    return {
        "status": "FAIL" if errors else "WARN",
        "hardFailures": errors,
        "normalOrientation": "UNVERIFIED",
        "orientationPolicy": "No heuristic channel swap or semantic declaration",
        "rolePixelsOpened": ["training"],
        "validationPixelsOpened": 0, "sealedFinalPixelsOpened": 0,
        "trainingAuthorities": len(results),
        "perAuthority": results,
    }


def _census_rows(rows: list[Any], resfiles: Path,
                 loader: AuthoredTextureDatasetRepository) -> dict[str, Any]:
    candidates = defaultdict(lambda: defaultdict(dict))
    indexed = Counter()
    local = Counter()
    missing = Counter()
    corrupt = Counter()
    for row in rows:
        classified = _classify(row.logical)
        if not classified:
            continue
        generation, role, stem = classified
        # Discovery is deliberately broad across EVE assets, not only ships;
        # no claim of an authored, aligned pair is made without both resources.
        indexed[(generation, role)] += 1
        path = resfiles / Path(row.hashed)
        if not path.is_file():
            missing[(generation, role)] += 1
            continue
        try:
            w, h, mips, fmt = loader.parse_dds_header(path)
        except (OSError, ValueError):
            corrupt[(generation, role)] += 1
            continue
        local[(generation, role)] += 1
        key = (generation, stem)
        record = {
            "logical": row.logical, "width": w, "height": h,
            "mipCount": mips, "format": fmt,
            "hashed": row.hashed,
        }
        existing = candidates[key][role].get("record")
        if existing is None or min(w, h) > min(existing["width"], existing["height"]):
            candidates[key][role]["record"] = record
    generations = {}
    for generation in FORMATS:
        cohort = [(k, v) for k, v in candidates.items() if k[0] == generation]
        pairs = [(stem, value["albedo"]["record"], value["normal"]["record"])
                 for (_gen, stem), value in cohort
                 if "albedo" in value and "normal" in value]
        eligible = [pair for pair in pairs
                    if min(pair[1]["width"], pair[1]["height"],
                           pair[2]["width"], pair[2]["height"]) >= 1024]
        with_material = sum(
            1 for (gen, stem), value in cohort
            if "albedo" in value and "normal" in value and "material" in value
        )
        generations[generation] = {
            "indexed": {role:indexed[(generation,role)] for role in FORMATS[generation]},
            "locallyAvailable":{role:local[(generation,role)] for role in FORMATS[generation]},
            "notLocal":{role:missing[(generation,role)] for role in FORMATS[generation]},
            "badHeader":{role:corrupt[(generation,role)] for role in FORMATS[generation]},
            "alignedAlbedoNormalStems": len(pairs),
            "pairedMin1024":len(eligible),
            "pairedWithDirectMaterialResource":with_material,
            "potentialPairIds": [
                hashlib.sha256((generation+"|"+stem).encode()).hexdigest()[:16]
                for stem,*_ in eligible
            ],
            "eligiblePairExamples": [
                {
                    "stem":stem,
                    "albedoLogical":albedo["logical"],
                    "normalLogical":normal["logical"],
                    "albedoResolution":[albedo["width"],albedo["height"]],
                    "normalResolution":[normal["width"],normal["height"]],
                }
                for stem,albedo,normal in eligible[:64]
            ],
            "eligiblePairExamplesTruncated":len(eligible)>64,
            "formatSemanticsVerified":False,
            "meaning": "Local/index/header availability only, not training eligibility or PBR supervision",
        }
    return {"status":"PASS", "source":"EVE SharedCache indexed resources",
            "sourceRowsScanned":len(rows), "generations":generations,
            "trainingCorpusModified":False, "semanticPBRCompletenessVerified":False}


def _audit_b(cache: str) -> dict[str, Any]:
    try:
        _cache_root, indexes, resfiles = eve.resolve_layout(cache, allow_prompt=False)
        rows = eve.read_rows(indexes)
        result = _census_rows(rows, resfiles, AuthoredTextureDatasetRepository())
        result["indexProvenance"] = [{
            "path":str(p), "size":p.stat().st_size,
            "mtimeNS":p.stat().st_mtime_ns,
        } for p in indexes]
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status":"UNAVAILABLE", "reason":f"{type(exc).__name__}: {exc}",
                "trainingCorpusModified":False}


def _conditioning(model: torch.nn.Module, batches: list[dict[str,torch.Tensor]],
                  config: Any) -> dict[str, Any]:
    model.eval()
    records = []
    with torch.inference_mode():
        for batch in batches:
            outputs = {}
            for scenario in ("original", "neutral-normals"):
                normal = (batch["lr_normal"] if scenario == "original"
                          else torch.zeros_like(batch["lr_normal"]))
                prediction = model(batch["lr_albedo"], normal, batch["lr_material"])
                # The canonical albedo target and LR albedo baseline are identical
                # under both treatments; geometric metrics would be confounded.
                m = sample_metrics(prediction, batch, final=False)
                outputs[scenario] = {k:m[k] for k in (
                    "global_recovery", "edge_recovery", "gradient_recovery",
                    "detail_recovery_1px", "lattice_cell_excess",
                )}
            records.append(outputs)
    return {
        "status": "PASS",
        "perDevelopmentCrop": records,
        "albedoOnlyMeanDeltaNeutralMinusOriginal": {
            k:float(np.mean([v["neutral-normals"][k]-v["original"][k]
                             for v in records]))
            for k in ("global_recovery","edge_recovery","gradient_recovery",
                      "detail_recovery_1px","lattice_cell_excess")
        },
        "qualificationDecisionMade":False,
        "caution":"Frozen model under out-of-training-distribution neutral normals; not proof corrected XY improves training",
    }


def _tile_ambiguity(sampler: NativeAuthoritySampler, *, max_tiles: int = 24) -> dict[str, Any]:
    """Training-only LR similarity test; near duplicates are candidate ambiguity."""
    rng = random.Random(17501)
    features, targets = [], []
    ids = []
    for authority, family in sampler.families.items():
        height, width = family.albedo.shape[:2]
        for _ in range(max_tiles):
            x = rng.randrange(0, width-64+1, 4)
            y = rng.randrange(0, height-64+1, 4)
            hr = family.albedo[y:y+64,x:x+64].astype(np.float32)/255
            lr = cv2.resize(hr,(16,16),interpolation=cv2.INTER_AREA)
            features.append(lr.reshape(-1))
            targets.append(hr)
            ids.append((authority,x,y))
    matches, conflicts = 0, 0
    closest = []
    if len(features) > 1:
        values = np.asarray(features, dtype=np.float32)
        for i in range(len(values)):
            distances = np.abs(values[i+1:]-values[i]).mean(axis=1)
            if not len(distances):
                continue
            j = i+1+int(np.argmin(distances))
            low = float(np.min(distances))
            # Not duplicate locations in the same source.
            if ids[i][0]==ids[j][0] and abs(ids[i][1]-ids[j][1])<64 and abs(ids[i][2]-ids[j][2])<64:
                continue
            if low < .015:
                matches += 1
                hr_delta = float(np.abs(targets[i]-targets[j]).mean())
                if hr_delta > .04:
                    conflicts += 1
                    if len(closest) < 12:
                        closest.append({
                            "firstAuthority":ids[i][0],"secondAuthority":ids[j][0],
                            "LRmae":low,"HRmae":hr_delta,
                        })
    return {
        "status":"WARN" if conflicts else "UNDETERMINED",
        "trainingSourceTiles":len(features),"verySimilarLRTilePairs":matches,
        "discordantHighResolutionPairs":conflicts,
        "examples":closest,
        "method":"Approximate 16x16 LR input to 64x64 HR authored albedo; similarity threshold .015, discordance .04",
        "limitation":"Negative or sparse matches do not establish recoverability; positives may reflect translation or photometric ambiguity",
        "validationTilesOpened":0,"sealedFinalTilesOpened":0,
    }


def _alignment_evidence(albedo: np.ndarray, normal: np.ndarray) -> dict[str, Any]:
    """Compare spatial boundaries, not component orientation or normal semantics."""
    side=256
    a=cv2.resize(albedo.astype(np.float32)/255,(side,side),
                 interpolation=cv2.INTER_AREA)
    n=cv2.resize(normal.astype(np.float32)/127.5-1,(side,side),
                 interpolation=cv2.INTER_AREA)
    light=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    def edge(value:np.ndarray)->np.ndarray:
        dx=cv2.Sobel(value,cv2.CV_32F,1,0,ksize=3)
        dy=cv2.Sobel(value,cv2.CV_32F,0,1,ksize=3)
        return cv2.magnitude(dx,dy)
    albedo_edges=edge(light)
    normal_edges=edge(n[...,0])+edge(n[...,1])
    if albedo_edges.std()<1.e-5 or normal_edges.std()<1.e-5:
        return {"status":"UNDETERMINED","reason":"Insufficient source edge variation"}
    # Crop with 4px boundary margin to avoid wrap and keep matching support.
    ae=albedo_edges[4:-4,4:-4]
    ae=(ae-ae.mean())/(ae.std()+1.e-6)
    scores={}
    for dy in range(-4,5):
        for dx in range(-4,5):
            ne=normal_edges[4+dy:side-4+dy,4+dx:side-4+dx]
            ne=(ne-ne.mean())/(ne.std()+1.e-6)
            scores[(dx,dy)]=float(np.mean(ae*ne))
    best=max(scores,key=scores.get)
    return {
        "status":"PASS",
        "peakOffsetAt256":[best[0],best[1]],
        "zeroOffsetCorrelation":scores[(0,0)],
        "peakCorrelation":scores[best],
        "peakImprovement":scores[best]-scores[(0,0)],
        "warning":"Albedo and normal edge maps need not coincide, so an offset is evidence to inspect, not proof of misregistration",
    }


def _audit_c(repo: Path, source: dict[str, Any],
             manifest: dict[str,Any], parts: dict[str,list[dict[str,Any]]],
             *, device: str, skip_conditioning: bool) -> dict[str,Any]:
    with_native = NativeAuthoritySampler(repo, manifest, parts["training"],
                                        hr_size=512, seed=17301)
    try:
        sampler = []
        for authority, fam in with_native.families.items():
            corrected = _antialias_integral(fam)
            h,w = fam.albedo.shape[:2]
            rng = random.Random(17501+int(authority[:8],16))
            candidates=[(rng.randint(0,w-512),rng.randint(0,h-512))
                        for _ in range(24)]
            legacy=max(candidates,key=lambda p:_score_box(fam.detail_integral,*p,512,w,h))
            fixed=max(candidates,key=lambda p:_score_box(corrected,*p,512,w,h))
            sampler.append({"authorityId":authority,"legacyOrigin":list(legacy),
                            "antiAliasedOrigin":list(fixed),"selectionChanged":legacy!=fixed})
        ambiguity=_tile_ambiguity(with_native)
        alignments={
            authority:_alignment_evidence(fam.albedo,fam.normal)
            for authority,fam in with_native.families.items()
        }
    finally:
        with_native.close()
    result={
        "status":"PASS",
        "sampling":{"perAuthority":sampler,
                    "changedAuthorities":sum(x["selectionChanged"] for x in sampler),
                    "proposedSamplerAppliedToTraining":False},
        "reconstructionAmbiguity":ambiguity,
        "sourceAlignment":alignments,
        "trainingFitDiagnostics":{},
    }
    root=repo/"artifacts/nsamdr/diagnostics/v17_comprehensive_proof"
    prior=sorted(root.glob("proof_*/report.json"),key=lambda p:p.stat().st_mtime,
                 reverse=True)
    for path in prior:
        try:
            report=json.loads(path.read_text(encoding="utf-8"))
            d=report.get("armResults",{}).get("D_continuation",{})
            if d.get("bestValidationUpdate")==86016 and report.get("sourceV3Report")==source["v3Report"]:
                selected=d.get("selectedTrained",{})
                result["trainingFitDiagnostics"]={
                    "referenceReport":str(path),
                    "fixedTrainProbeAllPass":selected.get("allPass"),
                    "fixedTrainProbeMeans":selected.get("mean"),
                    "probeCount":len(selected.get("perCrop") or []),
                    "interpretation":"Probe-only fit: not a direct measurement of all moving sampled regions",
                }
                break
        except (OSError, ValueError, TypeError):
            continue
    if skip_conditioning:
        result["conditioning"]={"status":"SKIPPED","reason":"Operator selected sampling-only diagnostics"}
    else:
        try:
            from probe_nsamdr_v17_comprehensive import _batch_config
            config=_batch_config(source)
            actual=torch.device(device if device!="auto" else
                                ("cuda" if torch.cuda.is_available() else "cpu"))
            if actual.type=="cpu":
                torch.set_num_threads(max(1,min(torch.get_num_threads(),4)))
            model, _optimizer=_load_v3_origin(source, config, actual)
            batches=[_batch_for_record(row,config,seed=17301+1001+i,device=actual)[0]
                     for i,row in enumerate(parts["validation"])]
            result["conditioning"]=_conditioning(model,batches,config)
            result["conditioning"]["developmentAuthorityCount"]=len(batches)
            result["conditioning"]["sealedFinalOpened"]=False
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
            result["conditioning"]={"status":"UNAVAILABLE",
                                    "reason":f"{type(exc).__name__}: {exc}"}
    if result["conditioning"]["status"] == "UNAVAILABLE":
        result["status"] = "WARN"
        result["reason"] = "Frozen-model conditioning unavailable; sampling and ambiguity diagnostics completed"
    return result


def _markdown(report:dict[str,Any]) -> str:
    lines=[
        "# NSAMDR V17.5 — Verified Source Readiness (A–C)",
        "",
        "Research-only, no training, no checkpoint/model/corpus mutation.",
        "",
        "| Phase | Result | Observation |",
        "|---|---|---|",
    ]
    for key in ("A_ddsIntegrity","B_generationCensus","C_reconstructionFeasibility"):
        phase=report.get("phases",{}).get(key,{})
        note=phase.get("reason") or (
            "Normal semantics UNVERIFIED" if key.startswith("A") else
            "Indexed presence only" if key.startswith("B") else
            "Frozen conditioning + sampling + LR ambiguity"
        )
        lines.append(f"| {key} | {phase.get('status','UNKNOWN')} | {str(note).replace('|','/')} |")
    lines.extend([
        "", "## Safety",
        "- Eight independent final-test authorities: untouched.",
        "- V17.3/V17.4 checkpoints and proof reports: unchanged.",
        "- No normal X/Y channel order inferred from channel statistics.",
        "- New texture generations are censused, NOT added to training.",
        "- CPU inference on four development validation crops (unless skipped) is diagnostic only.",
        "", f"Report: {report['reportPath']}", "",
    ])
    return "\n".join(lines)


def run(args:argparse.Namespace)->tuple[int,Path]:
    repo=args.repo_root.resolve()
    source=_origin(repo,args.v3_report)
    manifest,parts=_records(source)
    now=datetime.now().strftime("%Y%m%d-%H%M%S")
    folder=repo/DESTINATION/f"proof_{now}"
    folder.mkdir(parents=True,exist_ok=False)
    proof=folder/"report.json"
    report={
        "schema":SCHEMA,"researchOnly":True,"cpuTrainingUpdates":0,
        "productionPromotionAllowed":False,
        "finalTestEvaluated":False,"finalTestEvaluationCount":0,
        "sourceV3Report":source["v3Report"],"sourceV3ReportSha256":source["v3ReportSha256"],
        "sourceManifest":source["manifest"],
        "sourceCheckpointSha256":source["checkpointSha256"],
        "authoritiesByRole":{k:[_id(r) for r in v] for k,v in parts.items()},
        "phases":{},"reportPath":str(proof),
    }
    for name,action in (
        ("A_ddsIntegrity",lambda:_audit_a(repo,manifest,parts)),
        ("B_generationCensus",lambda:_audit_b(args.shared_cache)),
        ("C_reconstructionFeasibility",lambda:_audit_c(
            repo,source,manifest,parts,device=args.device,
            skip_conditioning=args.skip_conditioning)),
    ):
        print(f"[v17-source] {name} starting",flush=True)
        try:
            phase=action()
        except (OSError,RuntimeError,ValueError,KeyError,TypeError) as exc:
            phase={"status":"FAIL","reason":f"{type(exc).__name__}: {exc}"}
        report["phases"][name]=phase
        _atomic_json(folder/f"{name}.json",phase)
        _atomic_json(proof,report)
        print(f"[v17-source] {name} status={phase['status']}",flush=True)
    a=report["phases"]["A_ddsIntegrity"]
    report["decision"]=(
        "BLOCKED-source-integrity" if a["status"]=="FAIL"
        else "requires-verification-and-controlled-training"
    )
    _atomic_json(proof,report)
    (folder/"report.md").write_text(_markdown(report),encoding="utf-8")
    print(f"[v17-source] decision={report['decision']} finalTestOpened=False report={proof}",
          flush=True)
    return 0,proof


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--shared-cache",default=r"C:\CCP\EVE")
    p.add_argument("--v3-report",default="")
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    p.add_argument("--skip-conditioning",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
