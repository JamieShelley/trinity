"""V17.5 whole-authority BC4/BC5 source verification and frozen 12/64 selection."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import urllib.request
from ..authored_texture_dataset import AuthoredTextureDatasetRepository
from .spatial_sampling import _id, _crop_id, _family_ids

SELECTION_SCHEMA = "NSAMDR_V17_5_FROZEN_DIVERSITY_SELECTION_V1"
CONTROL_N, EXPANDED_N, ADDITIONAL_N = 12, 64, 52
SELECTION_SEED = 17301
MIN_NATIVE_SIDE = 1024
MIN_TARGET_RESIDUAL = .01
RESOURCE_CDN_BASE = "https://resources.eveonline.com"
ALBEDO_DDS_FORMATS = frozenset(("ATI1", "DXGI_80"))
NORMAL_DDS_FORMATS = frozenset(("ATI2", "DXGI_83"))

def canonical_logical(s: str) -> str:
    s = str(s).strip().replace("\\", "/").casefold()
    if not s:
        raise ValueError("empty logical resource")
    return s

def authority_id(row: dict) -> str:
    value = _id(row)
    if not value:
        raise ValueError("empty authority ID")
    return value

def crop_id(row: dict) -> str:
    value = _crop_id(row)
    if not value.endswith("_000"):
        raise ValueError(f"not a frozen _000 crop: {value}")
    return value

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()

def ship_stratum(logical: str) -> str:
    name = canonical_logical(logical)
    if "/model/ship/" not in name:
        raise ValueError("not-ship-classified")
    tokens = name.split("/model/ship/",1)[1].split("/")
    if len(tokens)<3 or not all(tokens[:2]):
        raise ValueError("not-ship-classified")
    return tokens[0]+"/"+tokens[1]

def _source(meta: dict, role: str, reader: AuthoredTextureDatasetRepository) -> dict:
    if not isinstance(meta,dict):
        raise ValueError(f"missing-{role}")
    path = Path(str(meta.get("path") or ""))
    if not path.is_file():
        raise ValueError(f"missing-local-{role}")
    if min(int(meta.get("width") or 0),int(meta.get("height") or 0))<MIN_NATIVE_SIDE:
        raise ValueError(f"small-{role}")
    with path.open("rb") as f:
        if f.read(4) != b"DDS ":
            raise ValueError(f"not-dds-{role}")
    w,h,_mips,fmt=reader.parse_dds_header(path)
    if (w,h)!=(int(meta["width"]),int(meta["height"])):
        raise ValueError(f"dimension-mismatch-{role}")
    formats=ALBEDO_DDS_FORMATS if role=="albedo" else NORMAL_DDS_FORMATS
    if fmt not in formats:
        raise ValueError(f"wrong-format-{role}:{fmt}")
    return {"logical":canonical_logical(meta.get("logical") or ""),
            "path":str(path.resolve()),"sha256":sha256_file(path),
            "format":fmt,"width":w,"height":h}

def _pair(family: dict, reader: AuthoredTextureDatasetRepository) -> dict:
    albedo=_source(family.get("albedo"),"albedo",reader)
    normal=_source(family.get("normal"),"normal",reader)
    return {"albedo":albedo,"normal":normal,
            "stratum":ship_stratum(albedo["logical"])}


def _verify_source_identity(crop: str, role: str, source: dict, frozen: dict) -> None:
    """Check the original identity rules and identify the failing condition.

    A missing cache file, a changed logical resource and a changed resolved
    path require different repairs. Do not combine them into a claim that
    the authored bytes changed: byte verification follows this check.
    """
    logical=canonical_logical(source.get("logical") or "")
    path=str(Path(str(source.get("path") or "")).resolve())
    frozen_exists=Path(frozen["path"]).is_file()
    failures=[]
    if logical!=frozen["logical"]:
        failures.append("logical_resource_mismatch")
    if path!=frozen["path"]:
        failures.append("resolved_path_mismatch")
    if not frozen_exists:
        failures.append("frozen_file_missing")
    if failures:
        diagnostic={
            "cropId":crop,"role":role,"failedChecks":failures,
            "frozenLogical":frozen["logical"],"currentLogical":logical,
            "frozenPath":frozen["path"],"currentResolvedPath":path,
            "frozenFileExists":frozen_exists,"currentFileExists":Path(path).is_file(),
            "expectedSha256":frozen.get("sha256"),
            "sourceBytesVerified":False,
        }
        raise RuntimeError(f"source path/role changed: {crop}/{role}\n"+
                           json.dumps(diagnostic,indent=2,sort_keys=True))



# Purpose: Derive the content-addressed EVE CDN suffix from a frozen ResFiles path.
# Called by: _restore_missing_frozen_source.
# Calls: Path.parts.
def _cdn_resource_suffix(path: Path) -> str:
    parts=list(path.parts)
    for index,part in enumerate(parts):
        if str(part).casefold()=="resfiles":
            tail=[str(value) for value in parts[index+1:]]
            if len(tail)>=2 and all(value not in ("",".","..") for value in tail):
                return "/".join(tail)
            return ""
    return ""


# Purpose: Rehydrate one evicted EVE cache object without changing frozen source identity.
# Called by: verify_frozen_selection.
# Calls: _cdn_resource_suffix, canonical_logical, sha256_file, urllib.request.urlopen.
def _restore_missing_frozen_source(crop: str, role: str, source: dict, frozen: dict) -> bool:
    logical=canonical_logical(source.get("logical") or "")
    current=Path(str(source.get("path") or ""))
    target=Path(str(frozen.get("path") or ""))
    if (logical!=frozen.get("logical") or
        str(current.resolve())!=str(target.resolve()) or target.is_file()):
        return False
    suffix=_cdn_resource_suffix(target)
    if not suffix:
        return False
    expected=str(frozen.get("sha256") or "").casefold()
    if len(expected)!=64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError(f"invalid frozen source SHA: {crop}/{role}")
    target.parent.mkdir(parents=True,exist_ok=True)
    fd,temp_name=tempfile.mkstemp(prefix=f".{target.name}.",suffix=".part",
                                  dir=str(target.parent))
    os.close(fd)
    temp=Path(temp_name)
    url=f"{RESOURCE_CDN_BASE}/{suffix}"
    try:
        request=urllib.request.Request(url,headers={"User-Agent":"NSAMDR-frozen-source-recovery/1"})
        with urllib.request.urlopen(request,timeout=120) as response, temp.open("wb") as output:
            for chunk in iter(lambda: response.read(1024*1024),b""):
                output.write(chunk)
        actual=sha256_file(temp)
        if actual.casefold()!=expected:
            raise RuntimeError(
                f"rehydrated source SHA mismatch: {crop}/{role}\n"
                f"expected={expected}\nactual={actual}\nurl={url}"
            )
        if target.is_file():
            existing=sha256_file(target)
            if existing.casefold()!=expected:
                raise RuntimeError(
                    f"frozen cache path changed during recovery: {crop}/{role}\n"
                    f"expected={expected}\nactual={existing}\npath={target}"
                )
            return True
        os.replace(temp,target)
        return True
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"failed to restore frozen EVE source: {crop}/{role}\n"
            f"url={url}\npath={target}\n{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _collision(candidate: dict, existing: list[dict]) -> bool:
    a,n=candidate["albedo"],candidate["normal"]
    for old in existing:
        oa,on=old["albedo"],old["normal"]
        if ((a["logical"],n["logical"])==(oa["logical"],on["logical"])
            or (a["sha256"],n["sha256"])==(oa["sha256"],on["sha256"])
            or a["path"] in (oa["path"],on["path"])
            or n["path"] in (oa["path"],on["path"])):
            return True
    return False

def freeze_authorities(repo: Path, source: dict, manifest: dict,
                       parts: dict, v2: dict, *,
                       selection_seed: int=SELECTION_SEED,
                       target_count: int=EXPANDED_N,
                       suitability: Callable[[dict],float]|None=None,
                       loader: AuthoredTextureDatasetRepository|None=None
                       ) -> tuple[dict,dict]:
    if selection_seed!=17301 or target_count!=64:
        raise ValueError("locked V17.5 selection: seed 17301; exactly 64")
    if v2.get("schema")!="NSAMDR_V17_GENERALIZATION_V2_PROOF_V1":
        raise RuntimeError("V2 proof provenance required")
    control,dev,sealed=(list(parts[key]) for key in ("training","validation","final_test"))
    if tuple(map(len,(control,dev,sealed)))!=(12,4,8):
        raise RuntimeError("frozen 12/4/8 partitions changed")
    ids=[[authority_id(row) for row in group] for group in (control,dev,sealed)]
    if len(set(sum(ids,[])))!=24:
        raise RuntimeError("V17.3 whole-authority overlap")
    protected_v2=_family_ids([*(v2.get("trainingCropIds") or []),
                             *(v2.get("validationCropIds") or []),
                             *(v2.get("finalTestCropIds") or [])])
    if not protected_v2:
        raise RuntimeError("V2 exclusions missing")
    v16_val={authority_id(row) for row in manifest.get("crops",[])
             if row.get("split")=="validation"}
    protected=set(sum(ids,[]))|protected_v2|v16_val
    by_family={str(f.get("familyId") or f.get("family_id") or ""):f
               for f in manifest.get("families",[])}
    reader=loader or AuthoredTextureDatasetRepository()
    reasons=Counter()
    controls=[]
    for row in control:
        fid=authority_id(row)
        if row.get("split")!="train" or not Path(str(row.get("path") or "")).is_file():
            raise RuntimeError(f"missing frozen control: {fid}")
        entry=_pair(by_family.get(fid,{}),reader)
        entry.update({"familyId":fid,"cropId":crop_id(row),
                      "cropBundleSha256":sha256_file(Path(row["path"]))})
        if _collision(entry,controls):
            raise RuntimeError(f"duplicate frozen control: {fid}")
        if suitability and float(suitability(row))<MIN_TARGET_RESIDUAL:
            raise RuntimeError(f"weak frozen control: {fid}")
        controls.append(entry)
    choices={}
    for row in manifest.get("crops",[]):
        if row.get("split")!="train" or not str(row.get("crop_id") or row.get("cropId") or "").endswith("_000"):
            continue
        fid=authority_id(row)
        if fid in protected:
            reasons["protected-or-original"]+=1
            continue
        if fid in choices:
            reasons["duplicate-family"]+=1
            continue
        if not Path(str(row.get("path") or "")).is_file():
            reasons["missing-train-crop"]+=1
            continue
        try:
            item=_pair(by_family.get(fid,{}),reader)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            reasons[str(exc).split(":")[0]]+=1
            continue
        item.update({"familyId":fid,"cropId":crop_id(row),
                     "cropBundleSha256":sha256_file(Path(row["path"]))})
        if _collision(item,controls) or _collision(item,list(choices.values())):
            reasons["duplicate-source"]+=1
            continue
        if suitability:
            try:
                residual=float(suitability(row))
            except (OSError,ValueError,RuntimeError):
                reasons["suitability-error"]+=1
                continue
            if residual<MIN_TARGET_RESIDUAL:
                reasons["weak-training-residual"]+=1
                continue
            item["baselineTargetResidual"]=residual
        choices[fid]=item
    buckets=defaultdict(list)
    for item in choices.values():
        buckets[item["stratum"]].append(item)
    for key in buckets:
        buckets[key].sort(key=lambda item: (
            hashlib.sha256(f"{selection_seed}|{item['familyId']}".encode()).hexdigest(),
            item["familyId"]))
    selected=[]
    for index in range(max(map(len,buckets.values()),default=0)):
        for key in sorted(buckets):
            if index<len(buckets[key]) and len(selected)<52:
                selected.append(buckets[key][index])
    audit={"schema":SELECTION_SCHEMA,"eligibleAdditional":len(choices),
           "selectedAdditional":len(selected),"rejectionsByReason":dict(sorted(reasons.items())),
           "selectedStratumCounts":dict(sorted(Counter(x["stratum"] for x in selected).items())),
           "developmentPixelsOpened":0,"sealedPixelsOpened":0,
           "trainingPixelsScreenedOnly":True}
    if len(selected)!=52:
        raise RuntimeError(f"BLOCKED_INSUFFICIENT_ELIGIBLE_TRAIN_AUTHORITIES: "
                           f"eligible={len(choices)} selected={len(selected)}/52; "
                           f"rejections={audit['rejectionsByReason']}")
    expanded=controls+selected
    if len({x["familyId"] for x in expanded})!=64:
        raise RuntimeError("duplicate selected family")
    selection={"schema":SELECTION_SCHEMA,
      "selectionPolicy":"A12-frozen-plus-52-ship-only-BC4-BC5-training-authorities",
      "selectionSeed":17301,
      "manifestPath":source["manifest"],"manifestSha256":sha256_file(Path(source["manifest"])),
      "v3ReportPath":source["v3Report"],"v3ReportSha256":source["v3ReportSha256"],
      "v2ReportPath":source["v2Report"],"v2ReportSha256":sha256_file(Path(source["v2Report"])),
      "stage7ReportSha256":None,
      "controlTrainingCropIds":[crop_id(x) for x in control],
      "expandedTrainingCropIds":[x["cropId"] for x in expanded],
      "developmentCropIds":[crop_id(x) for x in dev],
      "sealedFinalCropIds":[crop_id(x) for x in sealed],
      "protection":{"v2FamilyIds":sorted(protected_v2),
                    "v16ValidationFamilyIds":sorted(v16_val),
                    "frozenDevFamilyIds":ids[1],
                    "frozenFinalFamilyIds":ids[2]},
      "selectedAuthoritySources":expanded,
      "rejectionsByReason":audit["rejectionsByReason"],
      "selectedStratumCounts":audit["selectedStratumCounts"],
      "sealedPixelsOpened":0}
    return selection,audit

def verify_frozen_selection(selection: dict, source: dict, manifest: dict,
                            parts: dict, *, verify_bytes: bool=True) -> None:
    if selection.get("schema")!=SELECTION_SCHEMA or selection.get("selectionSeed")!=17301:
        raise RuntimeError("frozen selection version or seed differs")
    keys=("controlTrainingCropIds","expandedTrainingCropIds","developmentCropIds",
          "sealedFinalCropIds")
    if tuple(len(selection.get(k,[])) for k in keys)!=(12,64,4,8):
        raise RuntimeError("frozen selection counts differ")
    if any(selection[k] != [crop_id(r) for r in parts[role]]
           for k,role in (("controlTrainingCropIds","training"),
                          ("developmentCropIds","validation"),
                          ("sealedFinalCropIds","final_test"))):
        raise RuntimeError("historical frozen V17.3 identities differ")
    if selection["expandedTrainingCropIds"][:12]!=selection["controlTrainingCropIds"]:
        raise RuntimeError("A12 is not the prefix of B64")
    if (source["manifest"]!=selection["manifestPath"] or
        sha256_file(Path(source["manifest"]))!=selection["manifestSha256"] or
        source["v3ReportSha256"]!=selection["v3ReportSha256"] or
        sha256_file(Path(source["v2Report"]))!=selection["v2ReportSha256"]):
        raise RuntimeError("historical V17.3/V2/manifest SHA or path mismatch")
    crops={str(x.get("crop_id") or x.get("cropId")):x for x in manifest.get("crops",[])}
    families={str(x.get("familyId") or x.get("family_id")):x for x in manifest.get("families",[])}
    v2=json.loads(Path(source["v2Report"]).read_text(encoding="utf-8"))
    if v2.get("schema")!="NSAMDR_V17_GENERALIZATION_V2_PROOF_V1":
        raise RuntimeError("invalid original V2 proof")
    expected_v2=_family_ids([*(v2.get("trainingCropIds") or []),
                            *(v2.get("validationCropIds") or []),
                            *(v2.get("finalTestCropIds") or [])])
    expected_v16={authority_id(row) for row in manifest.get("crops",[])
                  if row.get("split")=="validation"}
    if (set(selection["protection"]["v2FamilyIds"])!=expected_v2 or
        set(selection["protection"]["v16ValidationFamilyIds"])!=expected_v16 or
        selection["protection"]["frozenDevFamilyIds"]!=[
            authority_id(row) for row in parts["validation"]] or
        selection["protection"]["frozenFinalFamilyIds"]!=[
            authority_id(row) for row in parts["final_test"]]):
        raise RuntimeError("frozen protection ledger differs from authoritative sources")
    protected=expected_v2|expected_v16
    if selection.get("selectionPolicy")!="A12-frozen-plus-52-ship-only-BC4-BC5-training-authorities":
        raise RuntimeError("frozen source-selection policy changed")
    seen=[]
    for i,(name,entry) in enumerate(zip(selection["expandedTrainingCropIds"],
                                        selection["selectedAuthoritySources"])):
        row=crops.get(name)
        if (row is None or row.get("split")!="train" or authority_id(row)!=entry["familyId"]
            or entry["cropId"]!=name):
            raise RuntimeError(f"selected source crop changed: {name}")
        if i>=12 and entry["familyId"] in protected:
            raise RuntimeError(f"protected source selected: {name}")
        if not Path(str(row.get("path") or "")).is_file():
            raise RuntimeError(f"frozen crop bundle missing: {name}")
        if verify_bytes and sha256_file(Path(row["path"]))!=entry.get("cropBundleSha256"):
            raise RuntimeError(f"frozen authored crop bundle SHA changed: {name}")
        family=families.get(entry["familyId"]) or {}
        reader=AuthoredTextureDatasetRepository()
        for role in ("albedo","normal"):
            src=family.get(role) or {}
            stored=entry[role]
            if not Path(stored["path"]).is_file():
                _restore_missing_frozen_source(name,role,src,stored)
            _verify_source_identity(name,role,src,stored)
            if verify_bytes and sha256_file(Path(stored["path"]))!=stored["sha256"]:
                raise RuntimeError(f"source SHA changed: {name}/{role}")
            verified=_source(src,role,reader)
            if (verified["format"]!=stored["format"] or
                verified["width"]!=stored["width"] or
                verified["height"]!=stored["height"]):
                raise RuntimeError(f"DDS physical-source format or dimensions changed: {name}/{role}")
        if _collision(entry,seen):
            raise RuntimeError("frozen source duplicate")
        seen.append(entry)
    if len(seen)!=64 or len(set(selection["expandedTrainingCropIds"]))!=64:
        raise RuntimeError("frozen selection roster entries do not match")

def selected_rows(selection: dict, manifest: dict, arm: str) -> list[dict]:
    if arm not in ("A12","B64"):
        raise ValueError("unknown diversity arm")
    rows={str(x.get("crop_id") or x.get("cropId")):x for x in manifest.get("crops",[])}
    ids=selection["controlTrainingCropIds"] if arm=="A12" else selection["expandedTrainingCropIds"]
    return [dict(rows[name]) for name in ids]
