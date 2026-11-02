"""Connected-component filtering on CPU arrays; no neural-network computation."""
import numpy as np


def keep_largest_component(labels, connectivity=8):
    """Keep the largest connected non-background region, preserving its class IDs.

    Area is pixel count. All nonzero classes form the foreground union.
    Equal-area ties select the first component in row-major scan order.
    """
    from scipy import ndimage

    labels = np.asarray(labels)
    if labels.ndim != 2 or not (np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_):
        raise ValueError("Expected a 2D integer/bool class mask")
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    components, count = ndimage.label(labels != 0, structure=structure)
    before = int(np.count_nonzero(labels))
    if count == 0:
        return labels.copy(), dict(components=0, foreground_before=0,
                                   foreground_after=0, removed_pixels=0)
    areas = np.bincount(components.ravel())
    areas[0] = 0
    largest = int(areas.argmax())
    keep = components == largest
    result = np.where(keep, labels, 0).astype(labels.dtype, copy=False)
    after = int(areas[largest])
    return result, dict(components=int(count), foreground_before=before,
                        foreground_after=after, removed_pixels=before-after)
