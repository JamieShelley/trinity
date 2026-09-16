from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.nsamdr.neural.prepare_nsamdr_v16_authored_prior_corpus import (
    CensusAuthorityRepository,
    PriorCorpusConfig,
    _latest_census,
    _load_census,
)


class AuthoredPriorCorpusTests(unittest.TestCase):
    def test_config_requires_genuine_hr_crop_support(self) -> None:
        config = PriorCorpusConfig(source_crop_size=512, min_source_dimension=1024)
        config.validate()
        with self.assertRaises(ValueError):
            PriorCorpusConfig(source_crop_size=512, min_source_dimension=256).validate()

    def test_authority_split_is_deterministic_and_whole_authority(self) -> None:
        repository = CensusAuthorityRepository(Path("dummy-census.json"))
        first = repository._family_split("authority-123", 0.12)
        second = repository._family_split("authority-123", 0.12)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation"})

    def test_latest_census_pointer_resolves_report(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "artifacts/nsamdr/diagnostics/eve_census/census_20260917-000000"
            run.mkdir(parents=True)
            report = run / "eve_authored_corpus_census.json"
            report.write_text(json.dumps({"authorities": []}), encoding="utf-8")
            latest = run.parent / "LATEST.txt"
            latest.write_text(str(run), encoding="utf-8")
            self.assertEqual(_latest_census(root), report.resolve())
            self.assertEqual(_load_census(report)["authorities"], [])


if __name__ == "__main__":
    unittest.main()
