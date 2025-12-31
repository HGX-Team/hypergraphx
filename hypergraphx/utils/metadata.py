def merge_metadata(existing, incoming):
    """Merge two metadata dicts into a single dict.

    If values conflict, accumulate them into lists while preserving order.
    """
    if existing is None:
        existing = {}
    if incoming is None:
        return dict(existing)
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        raise ValueError("Metadata must be a dict or None.")

    merged = dict(existing)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
            continue
        prev = merged[key]
        if prev == value:
            continue
        if isinstance(prev, list):
            prev_list = list(prev)
            if isinstance(value, list):
                for item in value:
                    if item not in prev_list:
                        prev_list.append(item)
            else:
                if value not in prev_list:
                    prev_list.append(value)
            merged[key] = prev_list
            continue
        if isinstance(value, list):
            new_list = [prev]
            for item in value:
                if item not in new_list:
                    new_list.append(item)
            merged[key] = new_list
            continue
        merged[key] = [prev, value]
    return merged
