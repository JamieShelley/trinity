"""Read-only Stage 7 V17.5 proof contract regression tests (CPU-only)."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.nsamdr.neural import probe_nsamdr_v17_source_readiness as proof
from tools.nsamdr.neural.authored_texture_dataset import AuthoredTextureDatasetRepository
from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication


def _dds_header(path: Path, *, width: int=1024, height: int=1024) -> None:
    # Valid DDS header/signature for index inspection; no pixel decoding.
    buf=bytearray(148)
    buf[:4]=b"DDS "
    struct.pack_into("<I",buf,4,124)
    struct.pack_into("<II",buf,12,height,width)
    struct.pack_into("<I",buf,28,11)
    buf[84:88]=b"DXT1"
    path.write_bytes(buf)


class SourceReadinessTests(unittest.TestCase):
    def test_hashed_dds_files_are_detected_from_bytes_not_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"d4f5072345"
            _dds_header(path,width=2048,height=1024)
            self.assertEqual(path.suffix,"")
            self.assertTrue(proof._dds_signature(path))
            size=AuthoredTextureDatasetRepository().parse_dds_header(path)
            self.assertEqual(size[:3],(2048,1024,11))
            path.write_bytes(b"\x89PNGnotdds")
            self.assertFalse(proof._dds_signature(path))

    def test_legacy_and_modern_suffixes_are_separate_and_case_insensitive(self):
        self.assertEqual(proof._classify("res:/a/ship_d.dds"),
                         ("legacy","albedo","res:/a/ship"))
        self.assertEqual(proof._classify("res:/a/ship_N.dds"),
                         ("legacy","normal","res:/a/ship"))
        self.assertEqual(proof._classify("res:/a/ship_PGS.dds"),
                         ("legacy","material","res:/a/ship"))
        self.assertEqual(proof._classify("res:/a/ship_AR.dds"),
                         ("modern","albedo","res:/a/ship"))
        self.assertEqual(proof._classify("res:/a/ship_NO.dds"),
                         ("modern","normal","res:/a/ship"))
        self.assertEqual(proof._classify("res:/a/ship_PMDG.dds"),
                         ("modern","material","res:/a/ship"))
        self.assertIsNone(proof._classify("res:/a/ship_black.dds"))
        self.assertIsNone(proof._classify("res:/a/ship_nothing.bin"))

    def test_census_keeps_generations_disjoint_and_requires_local_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for generation,suffixes in (
                ("legacy",("_d.dds","_n.dds")),
                ("modern",("_AR.dds","_NO.dds","_PMDG.dds")),
            ):
                for index,suffix in enumerate(suffixes):
                    hashed=f"{generation}_{index}_hash"
                    _dds_header(root/hashed)
                    rows.append(SimpleNamespace(
                        logical=f"res:/dx9/model/ship/example/ship{suffix}",
                        hashed=hashed))
            rows.append(SimpleNamespace(
                logical="res:/dx9/model/ship/missing/ship_NO.dds",
                hashed="not_downloaded"))
            out=proof._census_rows(
                rows,root,AuthoredTextureDatasetRepository())
            self.assertEqual(out["sourceRowsScanned"],6)
            legacy=out["generations"]["legacy"]
            modern=out["generations"]["modern"]
            self.assertEqual(legacy["pairedMin1024"],1)
            self.assertEqual(legacy["pairedWithDirectMaterialResource"],0)
            self.assertEqual(modern["pairedMin1024"],1)
            self.assertEqual(modern["pairedWithDirectMaterialResource"],1)
            self.assertEqual(modern["notLocal"]["normal"],1)
            self.assertFalse(modern["formatSemanticsVerified"])
            self.assertFalse(out["semanticPBRCompletenessVerified"])
            self.assertFalse(out["trainingCorpusModified"])

    def test_normal_hypotheses_do_not_infer_orientation(self):
        rgba=np.zeros((8,8,4),dtype=np.uint8)
        rgba[...,0]=255
        rgba[...,1]=100
        rgba[...,3]=180
        h=proof._normal_hypotheses(rgba)
        self.assertEqual(set(h),{
            "legacy-green-alpha","legacy-alpha-green","rgb-red-green"})
        self.assertEqual(h["legacy-green-alpha"]["xChannel"],"green")
        self.assertEqual(h["legacy-alpha-green"]["xChannel"],"alpha")
        self.assertEqual(h["rgb-red-green"]["yChannel"],"green")

    def test_raw_channel_statistics_preserve_alpha_variation(self):
        rgba=np.zeros((5,7,4),dtype=np.uint8)
        rgba[...,3]=np.arange(7,dtype=np.uint8)[None,:]*31
        stats=proof._stats(rgba)
        self.assertEqual(stats["alpha"]["min"],0)
        self.assertEqual(stats["alpha"]["max"],186)
        self.assertEqual(stats["alpha"]["distinct"],7)
        self.assertGreater(stats["alpha"]["std"],0)

    def test_a_reads_training_only_and_never_validation_or_sealed_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/"hashed"
            _dds_header(source)
            png=root/"mock_decode.png"
            pixels=np.full((8,8,4),127,dtype=np.uint8)
            pixels[...,3]=np.arange(8,dtype=np.uint8)[None,:]*20
            Image.fromarray(pixels,"RGBA").save(png)
            manifest={"families":[{
                "familyId":"abc0000000000000",
                "albedo":{
                    "role":"albedo","logical":"res:/one_d.dds",
                    "path":str(source),"width":8,"height":8,
                    "mip_count":1,"format":"DXT1",
                },
                "normal":None,
                "material":None,
            }]}
            parts={
                "training":[{"family_id":"abc0000000000000"}],
                "validation":[{"family_id":"def0000000000000"}],
                "final_test":[{"family_id":"fed0000000000000"}],
            }
            with patch.object(AuthoredTextureDatasetRepository,
                              "parse_dds_header",return_value=(8,8,1,"DXT1")):
                with patch.object(AuthoredTextureDatasetRepository,
                                  "_decode_source",side_effect=lambda *_:png):
                    with patch.object(proof,"_rgba_native_if_supported",
                                      return_value=(None,"unavailable")):
                        report=proof._audit_a(root,manifest,parts)
            self.assertEqual(len(report["perAuthority"]),1)
            self.assertEqual(report["perAuthority"][0]["authorityId"],
                             "abc0000000000000")
            self.assertEqual(report["validationPixelsOpened"],0)
            self.assertEqual(report["sealedFinalPixelsOpened"],0)
            self.assertEqual(report["normalOrientation"],"UNVERIFIED")
            self.assertEqual(report["status"],"WARN")

    def test_report_schema_and_cli_defaults(self):
        args=proof.parser().parse_args([])
        self.assertEqual(args.device,"cpu")
        self.assertFalse(args.skip_conditioning)
        app=NSAMDRCommandLineApplication()
        parsed=app.build_parser().parse_args([
            "v17-source-readiness","--skip-conditioning"])
        self.assertTrue(parsed.skip_conditioning)
        self.assertEqual(parsed.device,"cpu")

    def test_ambiguity_fails_to_claim_nonexistence_with_no_matches(self):
        # Sampling has no pixel access to development or final test; an
        # absence of matches cannot be interpreted as proof of recoverability.
        from tools.nsamdr.neural.v17.spatial_sampling import NativeFamily
        img=np.random.default_rng(7).integers(0,255,(128,128,3),dtype=np.uint8)
        family=NativeFamily("12345678abcdef01",img,np.zeros((128,128,2),
                   dtype=np.uint8),None,(0,0),np.ones((3,3)), "unknown",False)
        sampler=SimpleNamespace(families={"12345678abcdef01":family})
        result=proof._tile_ambiguity(sampler,max_tiles=2)
        self.assertEqual(result["validationTilesOpened"],0)
        self.assertEqual(result["sealedFinalTilesOpened"],0)
        self.assertIn(result["status"],("WARN","UNDETERMINED"))


if __name__=="__main__":
    unittest.main()
