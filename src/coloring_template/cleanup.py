"""Morphological cleanup for line art masks.

Cleans up raw extraction masks by:
  1. Removing large solid fill regions that aren't outlines
  2. Closing small gaps in outlines (morphological CLOSE)
  3. Removing isolated noise blobs below a minimum area
  4. Smoothing jagged edges with a light Gaussian pass + re-threshold
"""

import cv2
import numpy as np


def _remove_fill_regions(mask: np.ndarray, max_fill_ratio: float = 0.01) -> np.ndarray:
    """Remove large solid regions that are fills, not outlines.

    Real outlines are thin lines. Large filled areas (like dark ears on a dog,
    or dark hair fills) should be reduced to just their boundary edges.

    Strategy: erode the mask aggressively. Anything that survives heavy erosion
    is a thick/solid region (a fill). Subtract those interiors but keep their
    edges by taking just the boundary contour.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        max_fill_ratio: Maximum ratio of image area a single component can occupy
                        before being treated as a fill region. Default 0.02 (2%).

    Returns:
        Mask with fill interiors replaced by boundary outlines.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    max_area = int(total_pixels * max_fill_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    result = mask.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_area:
            continue

        # This component is suspiciously large — check if it's a solid fill
        # by measuring its "compactness" (area / convex hull area).
        # Also check aspect ratio — long thin lines can be large but aren't fills.
        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        # Solidity: how much of the convex hull is filled
        solidity = area / hull_area

        # Bounding box aspect ratio
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = bw * bh
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        # A solid fill region has high solidity AND high bounding-box fill ratio
        # Thin outlines/lines have low fill ratio even if they span large areas
        if solidity > 0.5 and fill_ratio > 0.3:
            # Replace with just the boundary contour (3px thick)
            result[labels == i] = 0
            cv2.drawContours(result, [cnt], -1, 255, 3)

    return result


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Compute the morphological skeleton of a binary mask.

    Uses iterative erosion/opening: at each step, subtract the opening from
    the eroded image and accumulate into the skeleton. Repeats until the
    image is fully eroded away. Pure OpenCV, no contrib modules needed.

    Args:
        mask: Binary mask (uint8) where 255 = foreground.

    Returns:
        Skeleton mask (uint8) where 255 = skeleton pixel.
    """
    img = mask.copy()
    skeleton = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        # Pixels that disappear in opening but survive erosion = skeleton layer
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded

        if cv2.countNonZero(img) == 0:
            break

    return skeleton


def _find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Find endpoint pixels in a skeleton (pixels with exactly 1 neighbor).

    Args:
        skeleton: Binary skeleton mask (uint8, 255 = skeleton pixel).

    Returns:
        Nx2 array of (x, y) endpoint coordinates.
    """
    # Convert to 0/1 for convolution
    skel_01 = (skeleton > 0).astype(np.uint8)

    # Count 8-connected neighbors using a 3x3 kernel (center excluded)
    neighbor_kernel = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_01, cv2.CV_16U, neighbor_kernel)

    # Endpoints: skeleton pixel with exactly 1 neighbor
    endpoint_mask = (skel_01 == 1) & (neighbor_count == 1)
    ys, xs = np.where(endpoint_mask)

    return np.column_stack([xs, ys]) if len(xs) > 0 else np.empty((0, 2), dtype=int)


def _close_gaps_via_endpoints(
    skeleton: np.ndarray,
    max_gap_radius: int = 20,
) -> np.ndarray:
    """Connect nearby skeleton endpoints on different components to close gaps.

    For each endpoint, finds the nearest endpoint on a different connected
    component and draws a line between them if within max_gap_radius.

    Args:
        skeleton: Binary skeleton mask (uint8).
        max_gap_radius: Maximum distance (pixels) to connect endpoints.

    Returns:
        Skeleton with gaps closed by connecting lines.
    """
    endpoints = _find_endpoints(skeleton)
    if len(endpoints) < 2:
        return skeleton

    # Label connected components so we can avoid connecting endpoints on same component
    num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)

    # Get component label for each endpoint
    ep_labels = labels[endpoints[:, 1], endpoints[:, 0]]

    result = skeleton.copy()
    n = len(endpoints)

    # For large endpoint counts, batch process to stay efficient
    if n > 10000:
        # Too many endpoints — skip gap closing to avoid slowdown
        return skeleton

    # Compute pairwise distances using vectorized numpy
    # endpoints is Nx2, compute all-pairs L2 distance
    diffs = endpoints[:, np.newaxis, :] - endpoints[np.newaxis, :, :]  # NxNx2
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # NxN

    # Mask out same-component pairs (set to infinity)
    same_component = ep_labels[:, np.newaxis] == ep_labels[np.newaxis, :]
    dists[same_component] = np.inf
    # Mask out self-connections
    np.fill_diagonal(dists, np.inf)

    # Track which endpoints have been connected (each connects to at most 1 other)
    connected = np.zeros(n, dtype=bool)

    # Greedily connect closest pairs
    for _ in range(n // 2):
        # Find the globally closest unconnected cross-component pair
        min_idx = np.argmin(dists)
        i, j = divmod(min_idx, n)
        min_dist = dists[i, j]

        if min_dist > max_gap_radius:
            break  # No more pairs within radius

        if connected[i] or connected[j]:
            # One of the pair was already connected — invalidate and continue
            dists[i, j] = np.inf
            dists[j, i] = np.inf
            continue

        # Draw connecting line
        pt1 = tuple(endpoints[i])
        pt2 = tuple(endpoints[j])
        cv2.line(result, pt1, pt2, 255, 1)

        connected[i] = True
        connected[j] = True
        dists[i, :] = np.inf
        dists[:, i] = np.inf
        dists[j, :] = np.inf
        dists[:, j] = np.inf

    return result


def _fill_holes(mask: np.ndarray, max_hole_area: int | None = None) -> np.ndarray:
    """Fill small enclosed holes inside outline components.

    Highlights inside dark regions (e.g. sunglass reflections, coffee cups,
    text inside speech bubbles) become white holes in the extraction mask.
    This function fills holes smaller than max_hole_area so the region
    remains a single closed colorable area.
    """
    if max_hole_area is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        max_hole_area = max(500, int(total_pixels * 0.001))

    # Invert so holes become foreground blobs
    inv = cv2.bitwise_not(mask)

    # Flood-fill from the border to mark all background-connected pixels.
    # Any remaining foreground pixels in `inv` are true holes.
    h, w = inv.shape
    flood_canvas = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_canvas, flood_mask, (0, 0), 128)

    # Pixels still 255 in flood_canvas are holes (not reachable from border)
    hole_mask = (flood_canvas == 255).astype(np.uint8) * 255

    # Label holes and filter by area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hole_mask, connectivity=8
    )

    result = mask.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
            result[labels == i] = 255

    return result


def clean(
    mask: np.ndarray,
    close_kernel_size: int = 9,
    min_component_area: int | None = None,
    smooth: bool = True,
    remove_fills: bool = True,
    max_fill_ratio: float = 0.002,
    gap_close: bool = False,
    max_gap_radius: int = 500,
    fill_holes: bool = True,
    max_hole_area: int | None = None,
) -> np.ndarray:
    """Clean a binary extraction mask for use as a coloring template.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        close_kernel_size: Size of the structuring element for morphological CLOSE.
                           Larger values seal wider gaps but may merge nearby lines.
        min_component_area: Minimum area in pixels for a connected component to be kept.
                            Components smaller than this are treated as noise and removed.
                            If None, auto-calculated as 0.0005% of total image area
                            (scales naturally with resolution).
        smooth: If True, apply a light Gaussian blur then re-threshold to smooth
                jagged/aliased edges. Recommended for flood-fill coloring apps.
        remove_fills: If True, detect and convert large solid fill regions to just
                      their boundary outlines. Fixes dark ears/hair being solid black.
        max_fill_ratio: Maximum ratio of image area for a single connected component
                        before it's treated as a fill region. Default 0.02 (2%).
        fill_holes: If True, fill small enclosed holes inside outline components.
        max_hole_area: Maximum area of a hole to fill. If None, auto-calculated.

    Returns:
        Cleaned binary mask (uint8), same shape as input.
    """
    # 1. Remove large solid fills (convert to boundary outlines)
    if remove_fills:
        mask = _remove_fill_regions(mask, max_fill_ratio=max_fill_ratio)

    # 2. First close pass to seal small gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Remove noise blobs smaller than min_component_area
    if min_component_area is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        min_component_area = max(30, int(total_pixels * 0.0001))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == i] = 255

    # 3.5. Skeleton-based gap closing: skeletonize to find endpoints, connect
    # nearby pairs on different components, then merge the connecting lines
    # back into the original cleaned mask (don't replace it).
    # Run two passes: second pass may close gaps created by the first.
    if gap_close:
        for _pass in range(3):
            skel = _morphological_skeleton(cleaned)
            skel_connected = _close_gaps_via_endpoints(skel, max_gap_radius=max_gap_radius)
            new_connections = cv2.subtract(skel_connected, skel)
            if cv2.countNonZero(new_connections) == 0:
                break  # No new connections found
            # Dilate the new connections to match line width (~3px)
            line_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            new_connections = cv2.dilate(new_connections, line_k, iterations=1)
            cleaned = cv2.bitwise_or(cleaned, new_connections)

    # 4. Dilate clean lines to bridge gaps, close, erode back
    bridge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thickened = cv2.dilate(cleaned, bridge_k, iterations=1)
    close2_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thickened, cv2.MORPH_CLOSE, close2_kernel)
    cleaned = cv2.erode(cleaned, bridge_k, iterations=1)

    # 4.5. Fill small enclosed holes (highlights inside dark regions, etc.)
    if fill_holes:
        cleaned = _fill_holes(cleaned, max_hole_area=max_hole_area)

    # 5. Light Gaussian blur + re-threshold to smooth jagged edges
    if smooth:
        blurred = cv2.GaussianBlur(cleaned, (3, 3), 0.8)
        _, cleaned = cv2.threshold(blurred, 85, 255, cv2.THRESH_BINARY)

    # 6. Add page border so edge-touching regions become enclosed/colorable.
    # A 2px border around the image ensures flood-fill from the border can't
    # leak into the template, making all regions between outlines colorable.
    cleaned[0:2, :] = 255
    cleaned[-2:, :] = 255
    cleaned[:, 0:2] = 255
    cleaned[:, -2:] = 255

    return cleaned
