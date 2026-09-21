def select_gradient_probe(named_tensors, event_index):
    """Select one shared tensor per diagnostic event to cap peak GPU memory."""
    if not named_tensors:
        return []
    index = int(event_index) % len(named_tensors)
    return [named_tensors[index]]
