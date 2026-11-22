import copy
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import retrieve_pa_opsd_eedi as retrieval
from pa_opsd_contract import fingerprint

REVISION = "a" * 40
METADATA = {"model": "fixture-only-not-a-real-model", "revision": REVISION, "source_sha256": "b" * 64}
CSV = (
    "MisconceptionId,MisconceptionName,ConstructName,SubTopicName,TopicName,SubjectName\n"
    "10,Place value error,Read digits,Place Value,Arithmetic,Number\n"
    "10,Place value error,Write digits,Place Value,Arithmetic,Number\n"
    "20,Adding error,Add integers,Addition,Arithmetic,Number\n"
    "30,Subtracting error,Subtract integers,Subtraction,Arithmetic,Number\n"
)


def seed(sample_id="fixture-1"):
    return {
        "sample_id": sample_id,
        "student_input": {
            "current_question": "[12]\n填空: 三百零五的十位数字是什么？\n知识点: 旧知识点\n时间: 2021-01-01\n结果: 错误",
            "history_info": ["private-history"],
        },
        "teacher_privileged": {
            "knowledge_concepts": ["位值", "整数"], "standard_answer": ["secret-answer"],
            "expert_analysis": "secret-analysis", "longtutor_weak_annotation": {"diagnosis": "secret-diagnosis"},
        },
        "target": {"diagnosis": "secret-target"},
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "map.csv"
        self.source.write_text(CSV, encoding="utf-8")
        self.index = retrieval.load_index(self.source)

    @staticmethod
    def encode(texts):
        # Deliberate toy vectors test arithmetic only, not cross-lingual quality.
        vectors = []
        for text in texts:
            if text.startswith("Place value"):
                vectors.append([4, 0])
            elif text.startswith("Adding"):
                vectors.append([3, 4])
            elif text.startswith("Subtracting"):
                vectors.append([-7, 0])
            else:
                vectors.append([2, 0])
        return vectors

    def rows(self, seeds=None, **kwargs):
        return list(retrieval.candidate_rows(seeds or [seed()], self.index, self.encode, METADATA, **kwargs))

    def test_query_allowlist_and_contract_fingerprint(self):
        original = seed()
        query = retrieval.retrieval_query(original)
        self.assertEqual(query, "填空: 三百零五的十位数字是什么？\n知识点: 位值, 整数")
        changed = copy.deepcopy(original)
        changed["teacher_privileged"]["standard_answer"] = ["different"]
        changed["teacher_privileged"]["longtutor_weak_annotation"] = {"diagnosis": "different"}
        changed["student_input"]["history_info"] = ["different"]
        changed["student_input"]["current_question"] = original["student_input"]["current_question"].replace("错误", "正确")
        self.assertEqual(retrieval.retrieval_query(changed), query)
        self.assertEqual(retrieval.fingerprint(query), fingerprint(query))
        self.assertEqual(self.rows()[0]["query_fingerprint"], fingerprint(query))
        changed["teacher_privileged"]["knowledge_concepts"].append("小数")
        self.assertNotEqual(fingerprint(retrieval.retrieval_query(changed)), fingerprint(query))

    def test_explicit_answer_and_diagnosis_suffix_is_not_queried(self):
        for label in ("答案", "标准答案", "解析", "诊断", "answer", "analysis", "diagnosis"):
            item = seed()
            item["student_input"]["current_question"] = "求十位数字\n" + label + ": SECRET"
            self.assertNotIn("SECRET", retrieval.retrieval_query(item))

    def test_invalid_query_schema(self):
        for question, concepts in (({}, []), ("", []), ("题目", "位值"), ("题目", [123]), ("答案: secret", [])):
            item = seed()
            item["student_input"]["current_question"] = question
            item["teacher_privileged"]["knowledge_concepts"] = concepts
            with self.assertRaises(ValueError):
                retrieval.retrieval_query(item)

    def test_real_schema_dedup_context_and_source_hash(self):
        self.assertEqual(len(self.index), 3)
        first = self.index[0]
        self.assertEqual(first["id"], "10")
        self.assertIn("Read digits", first["retrieval_text"])
        self.assertIn("Write digits", first["retrieval_text"])
        self.assertEqual(first["source"]["csv_lines"], [2, 3])
        self.assertEqual(first["source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.source.write_bytes(CSV.replace("Place value error", "Caf\u00e9 value error").encode("cp1252"))
        self.assertEqual(retrieval.load_index(self.source)[0]["source"]["encoding"], "cp1252")

    def test_invalid_index_and_conflicting_names(self):
        for text in ("id,text\n1,x\n", CSV.splitlines()[0] + "\n", CSV + "10,Conflicting,Read,Place,Arithmetic,Number\n", CSV + ",Empty,Read,Place,Arithmetic,Number\n"):
            self.source.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                retrieval.load_index(self.source)

    def test_cosine_top_k_scores_not_probabilities(self):
        row = self.rows(top_k=3, threshold=-1)[0]
        self.assertEqual([c["id"] for c in row["candidates"]], ["10", "20", "30"])
        self.assertEqual([c["score"] for c in row["candidates"]], [1.0, 0.6, -1.0])
        self.assertEqual(len(self.rows(top_k=1)[0]["candidates"]), 1)
        self.assertEqual(set(row), {"sample_id", "candidates", "embedding", "query_fingerprint"})
        self.assertEqual(set(row["candidates"][0]), {"id", "text", "score", "source"})
        self.assertNotIn("target", row)

    def test_threshold_can_return_no_match(self):
        def encode(texts):
            return [[1, 1] if "\n知识点:" in text else [1, 0] for text in texts]
        rows = list(retrieval.candidate_rows([seed()], self.index, encode, METADATA, threshold=0.9))
        self.assertEqual(rows[0]["candidates"], [])

    def test_seed_is_not_mutated(self):
        item = seed()
        saved = copy.deepcopy(item)
        self.rows([item])
        self.assertEqual(item, saved)

    def test_deterministic_ties_and_unique_sample_id(self):
        rows = list(retrieval.candidate_rows([seed()], self.index[::-1], lambda texts: [[1, 0]] * len(texts), METADATA))
        self.assertEqual([c["id"] for c in rows[0]["candidates"]], ["10", "20", "30"])
        for seeds in ([seed(), seed()], [dict(seed(), sample_id="")]):
            with self.assertRaises(ValueError):
                list(retrieval.candidate_rows(seeds, self.index, self.encode, METADATA))

    def test_invalid_vectors_and_configuration(self):
        for vectors in ([[0, 0]], [[math.nan, 1]], [[math.inf, 1]], [[]], [[1], [1, 2]], [["oops"]]):
            with self.assertRaises((ValueError, TypeError)):
                retrieval.normalized_vectors(vectors, len(vectors))
        with self.assertRaises(ValueError):
            retrieval.normalized_vectors([[1, 0]], 2)
        for kwargs in ({"top_k": 0}, {"threshold": 1.01}, {"threshold": math.nan}):
            with self.assertRaises(ValueError):
                self.rows(**kwargs)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            list(retrieval.candidate_rows([seed()], self.index, lambda texts: [[1]] if len(texts) == 1 else [[1, 0]] * len(texts), METADATA))

    def test_revision_must_be_immutable(self):
        for revision in (None, "", "main", "MAIN", "dev", "v1.0", "abc123"):
            with self.assertRaises(ValueError):
                retrieval.validate_revision(revision)
        with self.assertRaises(ValueError):
            list(retrieval.candidate_rows([seed()], self.index, self.encode, {"model": "x", "revision": REVISION}))

    def test_model_source_hash_is_content_based(self):
        model = self.directory / "model"
        model.mkdir()
        weights = model / "weights.bin"
        weights.write_bytes(b"weights")
        old_hash = retrieval.model_source_hash(model)
        self.assertEqual(retrieval.model_source_hash(model), old_hash)
        weights.write_bytes(b"changed")
        self.assertNotEqual(retrieval.model_source_hash(model), old_hash)

    def test_missing_optional_dependency_has_no_fallback(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}), self.assertRaisesRegex(RuntimeError, "no dependency installation"):
            retrieval.load_encoder(retrieval.DEFAULT_MODEL, REVISION)

    def test_cli_writes_only_candidate_sidecar(self):
        input_path = self.directory / "seed.jsonl"
        output = self.directory / "candidates.jsonl"
        input_path.write_text(json.dumps(seed(), ensure_ascii=False) + "\n", encoding="utf-8")
        before = input_path.read_bytes()
        args = ["--seed", str(input_path), "--output", str(output), "--source", str(self.source), "--revision", REVISION]
        with patch.object(retrieval, "load_encoder", return_value=(self.encode, METADATA)) as loader:
            self.assertEqual(retrieval.main(args), 0)
        loader.assert_called_once_with(retrieval.DEFAULT_MODEL, REVISION, 16, "cpu", True)
        row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["query_fingerprint"], fingerprint(retrieval.retrieval_query(seed())))
        self.assertNotEqual(row["embedding"]["revision"], "main")
        self.assertEqual(input_path.read_bytes(), before)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            retrieval.main(args)
        args[3] = str(input_path)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            retrieval.main(args)
        self.assertEqual(input_path.read_bytes(), before)

    def test_import_does_not_load_optional_embedding_stack(self):
        name = "retrieval_dependency_probe"
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / "retrieve_pa_opsd_eedi.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"sentence_transformers": None, "numpy": None}):
            spec.loader.exec_module(module)
            self.assertEqual(module.DEFAULT_MODEL, "BAAI/bge-m3")


if __name__ == "__main__":
    unittest.main()
