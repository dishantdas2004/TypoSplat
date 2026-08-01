import hashlib

def get_cache_tier(sample_id, disk_fraction=0.65):
    """
    Deterministically assigns each sample to 'disk' or 'ram' tier,
    based on a stable hash of its ID -- identical assignment every run,
    across both cache_features.py and train_stage1.py.
    """
    h = int(hashlib.md5(str(sample_id).encode()).hexdigest(), 16)
    return "disk" if (h % 100) < (disk_fraction * 100) else "ram"