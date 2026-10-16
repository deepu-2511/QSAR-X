import sys
import unittest
from pathlib import Path

from fastapi import HTTPException

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import main  # noqa: E402


class TargetRegistryTests(unittest.TestCase):
    def test_registry_exposes_egfr_and_braf_with_distinct_approval_states(self):
        targets = {item["target_id"]: item for item in main.list_targets()["targets"]}
        self.assertTrue(targets["CHEMBL203"]["prediction_available"])
        self.assertFalse(targets["CHEMBL5145"]["prediction_available"])

    def test_candidate_target_is_blocked_until_approval(self):
        with self.assertRaises(HTTPException) as caught:
            main.require_predictable_target("CHEMBL5145")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["code"], "MODEL_NOT_APPROVED")

    def test_unknown_target_is_not_silently_routed(self):
        with self.assertRaises(HTTPException) as caught:
            main.require_predictable_target("CHEMBL0000000")
        self.assertEqual(caught.exception.status_code, 404)

    def test_egfr_prediction_still_uses_existing_model(self):
        request = main.PredictionRequest(
            target_id="CHEMBL203",
            smiles="CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        )
        response = main.predict_endpoint(request)
        self.assertTrue(response["success"])
        self.assertEqual(response["result"]["target"]["target_id"], "CHEMBL203")
        self.assertIn("pIC50", response["result"]["prediction"])


if __name__ == "__main__":
    unittest.main()
