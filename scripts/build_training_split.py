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

import random

def split_group_a(bucket: list[dict], train_pct: float, val_pct: float, shadow_pct: float) -> tuple[list[dict], list[dict], list[dict]]:
    """Split benchmark proxy tasks into train, val, and shadow."""
    total = len(bucket)
    train_idx = int(total * train_pct)
    val_idx = train_idx + int(total * val_pct)
    
    train_slice = bucket[:train_idx]
    val_slice = bucket[train_idx:val_idx]
    shadow_slice = bucket[val_idx:]
    
    return train_slice, val_slice, shadow_slice

def split_group_b(bucket: list[dict], train_pct: float, probe_pct: float) -> tuple[list[dict], list[dict]]:
    """Split retention tasks into train and probe."""
    total = len(bucket)
    train_idx = int(total * train_pct)
    
    train_slice = bucket[:train_idx]
    probe_slice = bucket[train_idx:]
    
    return train_slice, probe_slice
