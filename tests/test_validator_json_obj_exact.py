from __future__ import annotations

import unittest

from atos_core.validators import run_validator


class JsonObjExactValidatorTests(unittest.TestCase):
    def test_json_obj_exact_accepts_canonical_json_text(self) -> None:
        out = '{"a":1,"b":2}'
        exp = {"a": 1, "b": 2}
        vr = run_validator("json_obj_exact", out, exp)
        self.assertTrue(vr.passed)

    def test_json_obj_exact_rejects_non_canonical_text(self) -> None:
        out = '{ "b": 2, "a": 1 }'
        exp = {"a": 1, "b": 2}
        vr = run_validator("json_obj_exact", out, exp)
        self.assertFalse(vr.passed)

    def test_json_obj_exact_accepts_python_obj(self) -> None:
        out = {"a": 1, "b": 2}
        exp = {"a": 1, "b": 2}
        vr = run_validator("json_obj_exact", out, exp)
        self.assertTrue(vr.passed)

    def test_json_obj_exact_rejects_wrong_expected_type(self) -> None:
        vr = run_validator("json_obj_exact", '{"a":1}', "not a dict")
        self.assertFalse(vr.passed)


if __name__ == "__main__":
    unittest.main()

