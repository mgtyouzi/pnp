from collections import defaultdict
from typing import Callable, List, Sequence, TypeVar


T = TypeVar("T")


def stratified_round_robin_indices(
    items: Sequence[T], maximum: int, key: Callable[[T], str]
) -> List[int]:
    if maximum <= 0 or maximum >= len(items):
        return list(range(len(items)))
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[str(key(item))].append(index)
    names = sorted(groups)
    offsets = {name: 0 for name in names}
    selected = []
    while len(selected) < maximum:
        added = False
        for name in names:
            offset = offsets[name]
            if offset >= len(groups[name]):
                continue
            selected.append(groups[name][offset])
            offsets[name] += 1
            added = True
            if len(selected) == maximum:
                break
        if not added:
            break
    return selected
