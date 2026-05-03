import unittest
from scripts.build_training_split import compute_hash, deduplicate_records, rebucket_records

class TestBuildTrainingSplit(unittest.TestCase):
    def test_compute_hash(self):
        record = {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"}
            ]
        }
        hash_val = compute_hash(record)
        self.assertTrue(isinstance(hash_val, str))
        self.assertTrue(len(hash_val) > 0)
        
        # Changing content should change hash
        record2 = {"messages": [{"role": "user", "content": "Question 2"}, {"role": "assistant", "content": "Answer 1"}]}
        self.assertNotEqual(hash_val, compute_hash(record2))

    def test_deduplicate_records(self):
        records = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}], "id": 1},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}], "id": 2},
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}], "id": 3}, # Duplicate of 1
        ]
        deduped = deduplicate_records(records)
        self.assertEqual(len(deduped), 2)
        self.assertEqual([r["id"] for r in deduped], [1, 2])

    def test_rebucket_records(self):
        records = [
            {"metadata": {"task": "task_a"}, "id": 1},
            {"metadata": {"task": "task_b"}, "id": 2},
            {"metadata": {"task": "task_a"}, "id": 3},
        ]
        buckets = rebucket_records(records)
        self.assertEqual(list(buckets.keys()), ["task_a", "task_b"])
        self.assertEqual(len(buckets["task_a"]), 2)
        self.assertEqual(len(buckets["task_b"]), 1)

if __name__ == '__main__':
    unittest.main()
