import hashlib
from collections import defaultdict

def compute_hash(record: dict) -> str:
    """Compute a SHA256 hash based on user and assistant messages."""
    content_str = ""
    for msg in record.get("messages", []):
        content_str += f"[{msg.get('role', '')}]:{msg.get('content', '')}|"
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

def deduplicate_records(records: list[dict]) -> list[dict]:
    """Remove records with duplicate prompt/response hashes. Keep first seen."""
    seen_hashes = set()
    deduped = []
    for record in records:
        h = compute_hash(record)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(record)
    return deduped

def rebucket_records(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by their metadata.task value."""
    buckets = defaultdict(list)
    for record in records:
        task = record.get("metadata", {}).get("task", "unknown")
        buckets[task].append(record)
    return buckets
