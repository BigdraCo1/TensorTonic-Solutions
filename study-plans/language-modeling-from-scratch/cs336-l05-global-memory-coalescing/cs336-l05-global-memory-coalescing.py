import numpy as np

def coalescing_analysis(byte_addresses: np.ndarray, access_width_bytes: int = 4, cache_line_bytes: int = 128) -> dict:
    """
    Returns a dict: line_ids (int64 array), transaction_count (int), useful_byte_fraction (float).
    """
    dedup_byte = np.unique(byte_addresses)
    last_byte_addresses = dedup_byte + access_width_bytes - 1
    s = np.floor(dedup_byte / cache_line_bytes)
    e = np.floor(last_byte_addresses / cache_line_bytes)
    overlap = 0.0
    for i in range(dedup_byte.size - 1):
        if dedup_byte[i + 1] <= last_byte_addresses[i]:
            overlap += last_byte_addresses[i] - dedup_byte[i + 1] + 1
    line = np.union1d(s, e).astype(np.int64)
    byte_fraction = 0.0
    for val in line:
        valid_addr_s = val * cache_line_bytes
        valid_addr_e = valid_addr_s + cache_line_bytes - 1
        start_idx = np.searchsorted(dedup_byte, valid_addr_s, side='left')
        end_idx = np.searchsorted(dedup_byte, valid_addr_e, side='right')  
        sz = dedup_byte[start_idx:end_idx].size
        byte_fraction += sz * access_width_bytes
    byte_fraction -= overlap
    u_cl = byte_fraction / float(line.size * cache_line_bytes)
    return dict(line_ids=line, transaction_count=line.size, useful_byte_fraction=u_cl)
