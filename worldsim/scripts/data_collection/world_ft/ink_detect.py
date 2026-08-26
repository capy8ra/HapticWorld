"""Classical ink detector + image enhancement for the wipe task.

The head-camera scene is controlled (white board, dark marker ink, blue tape frame), so
plain morphology finds the drawing reliably -- no learned model, so train (offline, on
the stored videos) and deploy (live) behave identically. Validated on start / mid-wipe /
late frames of 3 episodes (see data/ink_detect.png): full stroke coverage, no false
positives on the arm / gripper / eraser / tape / background; ink within ~1 cm of the arm
silhouette is suppressed by the exclusion halo.

Pipeline (all kernel sizes tuned on the full 640x480 frame -- run BEFORE any resize):
  board  = filled convex hull of the blue tape (robust to the arm crossing the board),
           shrunk past the tape, AND white surface (closing re-includes the thin ink),
           MINUS a dilated halo around large non-white objects (arm/gripper/eraser --
           their outlines would otherwise fire the thin-dark-structure filter);
  ink    = black-hat (dark details thinner than the kernel) over the board mask,
           minus specks and chunky blobs.

`enhance_ink` paints the detection into the image as saturated red (dilated), so a
~1-px stroke survives the 480->224 downscale as a fat unmissable curve. Used by
dataset.py (training, RGB frames) and eval_inference_worker.py (deploy, BGR frames) --
pass the right `bgr` flag.
"""
import cv2
import numpy as np

_E = cv2.getStructuringElement


def detect_ink(frame, bgr=True):
    """Binary ink mask (H, W) uint8 for a full-resolution camera frame."""
    frame_bgr = frame if bgr else frame[..., ::-1]
    frame_bgr = np.ascontiguousarray(frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # board interior: convex hull of the blue-tape frame, shrunk inside the tape
    blue = cv2.inRange(hsv, (90, 60, 40), (135, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, _E(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blue)
    big_blue = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 800]
    if not big_blue:
        return np.zeros(frame_bgr.shape[:2], np.uint8)
    pts = cv2.findNonZero(np.isin(lab, big_blue).astype(np.uint8))
    board = np.zeros(frame_bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(board, cv2.convexHull(pts), 1)
    board = cv2.erode(board, _E(cv2.MORPH_ELLIPSE, (71, 71)))
    # subtract the tape itself (dilated): specular highlights ON the tape are
    # desaturated enough to pass the white test and would read as ink otherwise
    board[cv2.dilate(blue, _E(cv2.MORPH_ELLIPSE, (21, 21))) > 0] = 0

    # AND white surface (closing fills the thin ink strokes back into the mask)
    white = cv2.inRange(hsv, (0, 0, 120), (180, 70, 255))
    whiteC = cv2.morphologyEx(white, cv2.MORPH_CLOSE, _E(cv2.MORPH_ELLIPSE, (35, 35)))

    # exclusion halo around LARGE non-white objects (arm/gripper/eraser). OPEN the
    # non-white mask first so THIN structures are erased before connectivity analysis:
    # a stroke drawn right up to the eraser would otherwise 8-connect into the arm
    # blob and be swallowed whole by the exclusion (observed on the 2026-07-30
    # board-filling camera view).
    nonwhite = (white == 0).astype(np.uint8)
    chunky = cv2.morphologyEx(nonwhite, cv2.MORPH_OPEN, _E(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(chunky)
    big = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 3000])
    # Known trade-off: the opening also disconnects thin CABLES from the arm blob, so
    # cables crossing the board in arm-deep (mid/late-wipe) frames can be painted as
    # ink. Accepted: episode-START frames (seeds, policy conditioning) have the arm
    # parked at home with cables off-board, and every attempted component-level cable
    # filter also re-swallowed strokes drawn up against the eraser (2026-07-31).
    excl = cv2.dilate(big.astype(np.uint8), _E(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    whiteE = cv2.erode(whiteC, _E(cv2.MORPH_ELLIPSE, (11, 11)))  # kills the gray board-rim line hugging the surface boundary
    surface = board.astype(bool) & (whiteE > 0) & (~excl)

    # ink: dark details thinner than the kernel (rejects wide soft shadows; 25 px
    # kernel keeps fat chisel-marker strokes fully inside the response)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _E(cv2.MORPH_ELLIPSE, (25, 25)))
    ink = ((blackhat > 32) & surface).astype(np.uint8)

    # drop specks and genuinely blobby residue (BOTH bbox dims large); strokes of any
    # length stay because their thin dimension is small
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < 20 or (min(w, h) > 30 and area / max(w * h, 1) > 0.5):
            continue
        keep[lab == i] = 1
    return keep


def enhance_ink(frame, bgr=True):
    """Copy of the frame with the detected ink painted saturated red (dilated so it
    survives the 480->224 downscale). Same color order in as out."""
    ink = detect_ink(frame, bgr=bgr)
    out = frame.copy()
    ink_fat = cv2.dilate(ink, _E(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    out[ink_fat] = (0, 0, 255) if bgr else (255, 0, 0)
    return out


def _order_skeleton(skel):
    """Ordered (x, y) pixel path along a 1-px skeleton: walk greedily from each
    endpoint (degree-1 pixel), keep the longest walk. Handles small spurs/loops."""
    ys, xs = np.nonzero(skel)
    pix = set(zip(xs.tolist(), ys.tolist()))
    if not pix:
        return []

    def nbrs(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pix]

    ends = [p for p in pix if len(nbrs(p)) == 1]
    starts = ends if ends else [min(pix)]  # closed curve: start anywhere
    best = []
    for start in starts:
        path, visited, cur = [start], {start}, start
        while True:
            nxt = [p for p in nbrs(cur) if p not in visited]
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            visited.add(cur)
        if len(path) > len(best):
            best = path
    return best


def stroke_keypoints(frame, k=8, bgr=True):
    """(k, 2) float32 [x, y] in [0, 1] (fractions of frame width/height): uniform-
    arclength samples along the LONGEST detected stroke, skeletonized and ordered from
    a canonical start (the endpoint with smaller x, tie-broken by y) so point i always
    means "i/(k-1) of the way along the drawing". All-zeros if nothing is detected.

    Intended use: computed ONCE from an episode's FIRST frame and held constant as a
    low-dim obs -- the drawing's shape stays known to the policy after partial wiping.
    """
    from skimage.morphology import skeletonize  # deferred import

    ink = detect_ink(frame, bgr=bgr)
    # bridge fragments before picking the component: the glare band and the arm-shadow
    # halo can split one stroke into pieces, and keypoints must describe the WHOLE
    # drawing, not the longest surviving fragment (observed 2026-07-31)
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _E(cv2.MORPH_ELLIPSE, (41, 41)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bridged)
    if n < 2:
        return np.zeros((k, 2), np.float32)
    comp = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    path = _order_skeleton(skeletonize(comp))
    if len(path) < max(2, k // 2):
        return np.zeros((k, 2), np.float32)
    pts = np.asarray(path, np.float32)  # (N, 2) as (x, y)
    if tuple(pts[0]) > tuple(pts[-1]):  # canonical direction
        pts = pts[::-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.linspace(0.0, s[-1], k)
    h, w = ink.shape
    return np.stack([np.interp(ss, s, pts[:, 0]) / w,
                     np.interp(ss, s, pts[:, 1]) / h], axis=1).astype(np.float32)
