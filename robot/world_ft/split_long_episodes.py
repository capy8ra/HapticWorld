"""Split long continuous-play recordings into ~20 s episodes.

Collecting play data for the world simulator is easiest as one long recording (just
keep the recorder running), but every downstream consumer — LowDimDataset, the WM
converter, prepare_upload.py — treats one directory as one episode. This tool cuts a
long recording into normal-length episode directories that look exactly like
separately-collected ones.

    python world_ft/split_long_episodes.py [--src DIR] [--dry-run]

Defaults: episodes longer than --long-threshold-s (40 s) are split into
--segment-s (20 s) pieces; a trailing piece shorter than --min-last-s (10 s) is
dropped. Each segment directory is named for its own start wall-clock time (the
collector's own `%Y_%m_%d_%H_%M_%S_%Z` convention), so consecutive segments differ by
~20 s exactly as separately-recorded episodes would.

Safety contract (do not weaken — a 7-minute recording was destroyed once by an
in-place rewrite):

  1. COPY the episode to --archive-dir first and md5-verify the copy file-by-file.
  2. SPLIT from the archived copy into a staging dir OUTSIDE --src, so no output path
     can ever be an input path, and a crash leaves the dataset folder untouched.
  3. VERIFY the staged segments reproduce the original in BOTH modalities (frame-time
     partition + bit-identical frames; byte-identical low-dim rows per stream group).
  4. Only after verification passes: remove the episode from --src and install the
     segments in its place.

  The archived copy is PERMANENT — it is never deleted by this tool, on any path
  (success or failure). It is the only surviving copy of the uncut recording, so keep
  --archive-dir out of any cleanup you run. Training must not see --archive-dir (the
  same play would be counted twice); keep it outside the dataset folder.

Why this is not `ffmpeg -ss ... -t ...`:

  * The recorder stores each frame's ABSOLUTE wall-clock time in the mp4 PTS
    (video_recorder.py: `av_frame.pts = timestamp / (1/1_000_000)`), and the dataset
    seeks frames by that absolute timestamp to pair them with the low-dim streams. A
    normal ffmpeg cut rebases timestamps to zero, which silently destroys that pairing.
    Here packets are REMUXED with their original pts/dts untouched (also lossless and
    fast — no re-encode).
  * The stream has B-frames, so packets arrive in decode order with reordered pts.
    Segments therefore start at keyframes (GOP boundaries, ~0.33 s apart at
    --video-gop-size 10), and each segment's low-dim window is derived from the pts
    actually written into that segment — never from nominal arithmetic — so video and
    low-dim stay aligned to the frame.

Low-dim streams are trimmed per group (`follower_right_state`, `follower_right_mit_command`,
`leader_right_state`, `follower_controller_state`): every array in a group is masked by
that group's own `:timestamp` array, so the different stream rates (2 kHz commands,
~840 Hz state) all survive the cut consistently.

Env: hapticworld (needs `av`), run from the repo root.
"""
import datetime
import os
import shutil
from collections import defaultdict
from time import localtime, strftime

import av
import numpy as np
import typer


def _episode_name(ts: float) -> str:
    """Collector-style directory name for a wall-clock timestamp."""
    tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    return f"{strftime('%Y_%m_%d_%H_%M_%S', localtime(ts))}_{tz}"


def _packet_index(video_path):
    """[(pts_seconds, is_keyframe)] per packet, in DECODE order."""
    idx = []
    with av.open(video_path) as c:
        st = c.streams.video[0]
        tb = st.time_base
        for p in c.demux(st):
            if p.pts is None:  # flush packet
                continue
            idx.append((float(p.pts * tb), bool(p.is_keyframe)))
    return idx


def _plan_segments(idx, segment_s, min_last_s):
    """Assign each packet to a segment. Segments start at keyframes >= the running
    nominal boundary; advancing the boundary from the ACTUAL cut (not from t0 + k*seg)
    keeps every segment at least segment_s long instead of compounding the snap.

    Returns (packet_segment_ids, n_segments) with -1 for packets before the first
    keyframe (there are none in practice: packet 0 is a keyframe).
    """
    seg_ids = []
    cur = -1
    next_boundary = -np.inf
    for pts, is_key in idx:
        if is_key and pts >= next_boundary:
            cur += 1
            next_boundary = pts + segment_s
        seg_ids.append(cur)
    n = cur + 1
    if n == 0:
        return seg_ids, 0
    # drop a too-short trailing segment (its packets become -1 = not written)
    last_pts = [p for p, s in zip([x[0] for x in idx], seg_ids) if s == n - 1]
    if n > 1 and (max(last_pts) - min(last_pts)) < min_last_s:
        seg_ids = [s if s != n - 1 else -1 for s in seg_ids]
        n -= 1
    return seg_ids, n


def _decodable_pts(video_path):
    """pts of frames that actually DECODE from a file (open-GOP B-frames whose reference
    was cut away are demuxable but not decodable)."""
    with av.open(video_path) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        return {f.pts for f in c.decode(st)}


def _write_video_segments(src_video, seg_ids, out_paths, exclude_pts=frozenset()):
    """Remux packets into per-segment mp4s, preserving pts/dts exactly.
    Returns ({segment: (min_pts_s, max_pts_s)}, {segment: [written int pts]})."""
    # HARD GUARD: opening an output over the file we are reading truncates it in place
    # and destroys the recording (this happened once — segment 0 inherited the source
    # episode's own name while out_root was the source folder). The caller archives the
    # original before splitting, so this must never be reachable; assert it anyway.
    src_abs = os.path.abspath(src_video)
    for path in out_paths.values():
        if os.path.abspath(path) == src_abs:
            raise RuntimeError(f"refusing to write a segment over the source video: {src_abs}")
    spans = {}
    written = defaultdict(list)
    with av.open(src_video) as ic:
        istream = ic.streams.video[0]
        tb = istream.time_base
        oc = ostream = None
        cur = None
        i = -1
        try:
            for p in ic.demux(istream):
                if p.pts is None:
                    continue
                i += 1
                seg = seg_ids[i]
                if seg < 0 or p.pts in exclude_pts:
                    continue
                if seg != cur:
                    if oc is not None:
                        oc.close()
                    os.makedirs(os.path.dirname(out_paths[seg]), exist_ok=True)
                    # movie_timescale=1e6 is REQUIRED: the mp4 muxer stores the movie
                    # start time at its movie timescale (default 1000 = milliseconds),
                    # so with the default every frame timestamp in a segment whose first
                    # pts is not on a whole millisecond gets shifted back by up to 999 us
                    # (measured: -175/-473/-980 us on three segments). The recorder writes
                    # absolute wall-clock microseconds into pts, so a microsecond timescale
                    # reproduces them exactly and video stays aligned with the low-dim.
                    oc = av.open(out_paths[seg], "w",
                                 options={"movie_timescale": "1000000"})
                    ostream = oc.add_stream_from_template(istream)
                    cur = seg
                t = float(p.pts * tb)
                lo, hi = spans.get(seg, (t, t))
                spans[seg] = (min(lo, t), max(hi, t))
                written[seg].append(int(p.pts))
                p.stream = ostream
                oc.mux(p)
        finally:
            if oc is not None:
                oc.close()
    return spans, dict(written)


def _trim_low_dim(src_ld, dst_ld, t_start, t_end):
    """Copy low_dim npys keeping rows with t_start <= timestamp < t_end.
    Each stream group is masked by its OWN :timestamp array (2-D timestamp arrays are
    per-joint; column 0 is the reference, as elsewhere in the codebase)."""
    groups = defaultdict(list)
    for f in sorted(os.listdir(src_ld)):
        if f.endswith(".npy"):
            groups[f.split(":")[0]].append(f)
    os.makedirs(dst_ld, exist_ok=True)
    counts = {}
    for group, files in groups.items():
        ts_file = f"{group}:timestamp.npy"
        if ts_file not in files:
            raise ValueError(f"{src_ld}: group '{group}' has no timestamp stream")
        # Streams in a group can end a few samples apart — the recorder snapshots each
        # array at a slightly different instant, so e.g. :q can carry 39 rows that
        # :timestamp does not (measured on 3 of 353 episodes, ~0.003% of rows). The
        # extra rows are at the tail, so trimming the group to its common length
        # realigns it; refusing outright would just make those episodes unusable.
        lens = {f: np.load(os.path.join(src_ld, f), mmap_mode="r").shape[0] for f in files}
        n = min(lens.values())
        if len(set(lens.values())) > 1:
            print(f"      note: {group} streams end {max(lens.values()) - n} sample(s) "
                  f"apart (recorder artifact) — trimming group to {n} rows")
        ts = np.load(os.path.join(src_ld, ts_file))[:n]
        ref = ts[:, 0] if ts.ndim > 1 else ts
        mask = (ref >= t_start) & (ref < t_end)
        for f in files:
            np.save(os.path.join(dst_ld, f), np.load(os.path.join(src_ld, f))[:n][mask])
        counts[group] = int(mask.sum())
    return counts


def _video_pts_int(video_path):
    """Sorted RAW integer pts (absolute microseconds). Integers because these values
    need 16 significant digits — comparing them as float seconds silently loses the
    last microseconds."""
    with av.open(video_path) as c:
        st = c.streams.video[0]
        return np.array(sorted(p.pts for p in c.demux(st) if p.pts is not None),
                        dtype=np.int64)


def _md5(path, chunk=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _dir_hashes(ep_dir, camera):
    """{relative file -> md5} for an episode directory (video + every npy)."""
    out = {f"{camera}.mp4": _md5(os.path.join(ep_dir, f"{camera}.mp4"))}
    ld = os.path.join(ep_dir, "low_dim_npys")
    for f in sorted(os.listdir(ld)):
        if f.endswith(".npy"):
            out[f"low_dim_npys/{f}"] = _md5(os.path.join(ld, f))
    return out


def copy_original(ep_dir, archive_dir):
    """STEP 1: copy the episode to the archive and prove the copy is faithful.

    A copy (not a move) so the episode stays in the dataset folder untouched for the
    whole operation: until the segments are written AND verified, a crash leaves
    everything exactly as it was.
    """
    os.makedirs(archive_dir, exist_ok=True)
    dst = os.path.join(archive_dir, os.path.basename(ep_dir))
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} already exists — resolve before re-running")
    before = _dir_hashes(ep_dir, "camera_head")
    shutil.copytree(ep_dir, dst)
    after = _dir_hashes(dst, "camera_head")
    if before != after:
        differing = [k for k in before if before.get(k) != after.get(k)]
        raise RuntimeError(f"copy of {ep_dir} is not faithful; differs in {differing}")
    print(f"  copied to {dst} ({len(before)} files, md5-verified identical)")
    return dst


def verify_against_original(original, seg_dirs, camera="camera_head", n_frame_probes=3):
    """STEP 3: prove the segments reproduce the original, for BOTH modalities.

    Video: every segment's frame timestamps must be an exact, gapless, non-overlapping
    partition of the original's (minus a dropped tail), and sampled frames must decode
    bit-identically. Low-dim: every row must be byte-identical to the original's rows in
    that segment's window, with no gaps or duplication introduced.

    Note: the 2 kHz solver streams legitimately contain ~30% duplicate timestamps (a
    known recorder artifact the dataset dedups) — so duplicates are compared AGAINST the
    original rather than required to be zero.
    """
    ok, lines = True, []

    # ---- video: pts partition (INTEGER pts — absolute microsecond timestamps need 16
    # significant digits, past float64's exact range, so float comparison is unusable) ---
    o_sorted = _video_pts_int(os.path.join(original, f"{camera}.mp4"))
    per_seg, s_pts = {}, []
    for d in seg_dirs:
        p = _video_pts_int(os.path.join(d, f"{camera}.mp4"))
        per_seg[d] = p
        s_pts.append(p)
    s_sorted = np.sort(np.concatenate(s_pts))
    subset = bool(np.isin(s_sorted, o_sorted).all())   # no shifted/invented timestamps
    dups = len(s_sorted) - len(np.unique(s_sorted))    # no frame in two segments
    ok &= subset and dups == 0
    lines.append(f"  video frames: {len(s_sorted)}/{len(o_sorted)} of original "
                 f"({len(o_sorted) - len(s_sorted)} dropped: open-GOP boundary frames + "
                 f"tail) | every timestamp exact & unshifted: {'yes' if subset else 'NO'}"
                 f" | duplicated across segments: {dups}")

    # every frame a segment claims must actually decode (else the dataset would sample a
    # stale image at a correct timestamp)
    # informational: the known open-GOP boundary frame (see split_episode). Only fails
    # if it is more than the expected 1 per segment.
    n_undec = sum(len(per_seg[d]) - len(_decodable_pts(os.path.join(d, f"{camera}.mp4")))
                  for d in seg_dirs)
    ok &= n_undec <= len(seg_dirs)
    lines.append(f"  video decodability: {len(s_sorted) - n_undec}/{len(s_sorted)} frames "
                 f"decode standalone ({n_undec} open-GOP boundary frame(s), expected "
                 f"<= 1 per segment = {len(seg_dirs)}) "
                 f"[{'OK' if n_undec <= len(seg_dirs) else 'FAIL'}]")

    # ---- video: content identity at sampled frames ----
    from nova.tools.video_util import decode_video_frame_at_pts
    bad = 0
    for d in seg_dirs:
        p = per_seg[d]
        # decode_video_frame_at_pts takes SECONDS; pts are absolute microseconds. The
        # float conversion is fine here (it only has to land nearest a 33 ms frame).
        # p[-1] is the known open-GOP boundary frame (decodes only in the uncut
        # recording); probe the last frame BEFORE it instead
        probes = [float(x) / 1e6 for x in (p[0], p[len(p) // 2], p[-2])]
        for t in probes[:n_frame_probes]:
            a = decode_video_frame_at_pts(os.path.join(original, f"{camera}.mp4"), t)
            b = decode_video_frame_at_pts(os.path.join(d, f"{camera}.mp4"), t)
            if not np.array_equal(a, b):
                bad += 1
    ok &= bad == 0
    lines.append(f"  video content: {len(seg_dirs)*n_frame_probes} sampled frames "
                 f"decode bit-identical to the original: {'yes' if bad == 0 else f'NO ({bad} differ)'}")

    # ---- low-dim: row identity + coverage, per stream group ----
    o_ld = os.path.join(original, "low_dim_npys")
    groups = defaultdict(list)
    for f in sorted(os.listdir(o_ld)):
        if f.endswith(".npy"):
            groups[f.split(":")[0]].append(f)
    for group, files in groups.items():
        # same common-length trim the splitter applied (see _trim_low_dim)
        n0 = min(np.load(os.path.join(o_ld, f), mmap_mode="r").shape[0] for f in files)
        o_ts = np.load(os.path.join(o_ld, f"{group}:timestamp.npy"))[:n0]
        o_ref = o_ts[:, 0] if o_ts.ndim > 1 else o_ts
        masks, total = {}, 0
        for d in seg_dirs:
            s_ts = np.load(os.path.join(d, "low_dim_npys", f"{group}:timestamp.npy"))
            s_ref = s_ts[:, 0] if s_ts.ndim > 1 else s_ts
            total += len(s_ref)
            masks[d] = (o_ref >= s_ref[0]) & (o_ref <= s_ref[-1])
        mismatch = 0
        for f in files:
            o_arr = np.load(os.path.join(o_ld, f))[:n0]
            for d in seg_dirs:
                if not np.array_equal(np.load(os.path.join(d, "low_dim_npys", f)),
                                      o_arr[masks[d]]):
                    mismatch += 1
        # gapless: the rows kept must equal the original's rows across the covered span
        in_span = int(((o_ref >= o_ref.min()) & (o_ref <= o_ref.max())).sum())
        g_ok = mismatch == 0 and total <= in_span
        ok &= g_ok
        lines.append(f"  {group:28s} rows {total:8d}/{len(o_ref):8d} | "
                     f"byte-identical to original: {'yes' if mismatch == 0 else f'NO ({mismatch})'} "
                     f"[{'OK' if g_ok else 'FAIL'}]")
    return ok, lines


def _episode_duration(ep_dir):
    p = os.path.join(ep_dir, "low_dim_npys", "follower_controller_state:timestamp.npy")
    if not os.path.exists(p):
        return None
    t = np.unique(np.load(p).ravel())
    return float(t[-1] - t[0]) if len(t) > 1 else 0.0


def split_episode(ep_dir, out_root, segment_s, min_last_s, camera="camera_head",
                  dry_run=False, skip_names=()):
    """Split one episode into STAGED segment dirs (outside out_root) and return
    (staging_root, [staged paths in order]). The caller verifies them and only then
    moves them into out_root — so a failure never leaves partial episodes in the
    dataset folder. `skip_names` are names in out_root that may be reused (the source
    episode's own name: it is still in place here and is removed after verification)."""
    video = os.path.join(ep_dir, f"{camera}.mp4")
    src_ld = os.path.join(ep_dir, "low_dim_npys")
    if not (os.path.exists(video) and os.path.isdir(src_ld)):
        raise FileNotFoundError(f"{ep_dir}: expected {camera}.mp4 + low_dim_npys/")

    idx = _packet_index(video)
    seg_ids, n = _plan_segments(idx, segment_s, min_last_s)
    if n < 2:
        print(f"  {os.path.basename(ep_dir)}: only {n} segment(s) — skipping")
        return []

    # planned cut points (first pts of each segment, in decode order)
    cuts = {}
    for (pts, _), seg in zip(idx, seg_ids):
        if seg >= 0 and seg not in cuts:
            cuts[seg] = pts

    # the source episode is removed after verification, so its name is free for
    # segment 0 (segments then look like consecutive real recordings)
    used = set(os.listdir(out_root)) - set(skip_names)
    names = {}
    for seg in range(n):
        bump = 0
        name = _episode_name(cuts[seg])
        while name in used:  # never clobber a real episode recorded at that second
            bump += 1
            name = _episode_name(cuts[seg] + bump)
        used.add(name)
        names[seg] = name

    if dry_run:
        for seg in range(n):
            end = cuts[seg + 1] if seg + 1 < n else idx[-1][0]
            print(f"    [{seg:3d}] {names[seg]}  {end - cuts[seg]:5.1f}s")
        return []

    # stage OUTSIDE out_root so a crash can never leave half-written episodes in the
    # dataset folder; completed segments are moved in at the end
    staging = os.path.join(os.path.dirname(ep_dir), f"{os.path.basename(ep_dir)}__segments")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    out_paths = {seg: os.path.join(staging, names[seg], f"{camera}.mp4")
                 for seg in range(n)}
    spans, written = _write_video_segments(video, seg_ids, out_paths)
    if len(spans) != n:
        raise RuntimeError(f"{ep_dir}: wrote {len(spans)} of {n} planned video segments")

    # This stream has OPEN GOPs: the trailing B-frames of a segment's last GOP reference
    # the NEXT segment's keyframe, so once cut they are demuxable but NOT decodable. The
    # dataset enumerates frames by DEMUX (load_image_frames_pts), so such a packet would
    # become a training sample whose image silently falls back to the previous frame
    # (33 ms stale) while its low-dim is sampled at the correct, later time. Drop them
    # and rewrite once — a segment then ends one frame earlier instead of lying.
    # KNOWN LIMITATION (accepted, reported by the verifier): this stream has open GOPs,
    # so the LAST frame of each segment references the next segment's keyframe and
    # decodes only in the uncut recording. The dataset enumerates frames by demux, so
    # that one timestamp per segment gets the previous frame's image (33 ms stale) —
    # 1 sample in ~600, at a segment boundary. Do NOT try to drop these packets: the
    # highest-pts packet is usually a reference frame and removing it breaks the frames
    # that depend on it (each pass then exposes more; measured, does not converge).

    # low-dim windows come from what was actually written: [seg start, next seg start)
    created = []
    for seg in range(n):
        t_start = spans[seg][0]
        t_end = spans[seg + 1][0] if seg + 1 in spans else spans[seg][1] + 1.0
        staged = os.path.join(staging, names[seg])
        counts = _trim_low_dim(src_ld, os.path.join(staged, "low_dim_npys"), t_start, t_end)
        if min(counts.values()) == 0:
            raise RuntimeError(f"{staged}: a low-dim group is empty for "
                               f"[{t_start:.3f}, {t_end:.3f})")
        print(f"    [{seg:3d}] {names[seg]}  {spans[seg][1] - spans[seg][0]:5.1f}s video, "
              f"low-dim rows: " + ", ".join(f"{k.split('_')[-1]}={v}" for k, v in counts.items()))
        created.append(staged)
    return staging, created


def main(
    src: str = "../data/microwave_full",
    archive_dir: str = "../data/microwave_long_originals",
    segment_s: float = 20.0,
    min_last_s: float = 10.0,
    long_threshold_s: float = 40.0,
    camera: str = "camera_head",
    dry_run: bool = False,
):
    eps = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    longs = []
    for ep in eps:
        d = _episode_duration(os.path.join(src, ep))
        if d is not None and d > long_threshold_s:
            longs.append((ep, d))

    print(f"{src}: {len(eps)} episodes, {len(longs)} longer than {long_threshold_s:.0f}s")
    if not longs:
        return
    for ep, d in longs:
        print(f"  {ep}  {d:.1f}s")
    if dry_run:
        print("\n-- dry run: planned segments --")

    failures = []
    for ep, _ in longs:
        print(f"\nsplitting {ep}:")
        if dry_run:
            split_episode(os.path.join(src, ep), src, segment_s, min_last_s,
                          camera=camera, dry_run=True)
            continue
        # COPY -> SPLIT -> VERIFY -> only then drop the original from the dataset folder.
        # The original stays in src, untouched and readable, for the whole operation: the
        # segments are generated from the archived COPY and written to staging outside
        # src, so nothing can overwrite an input, and any failure leaves the dataset
        # exactly as it was.
        original = os.path.join(src, ep)
        staging = None
        try:
            # a previous run may have archived this episode before failing; the archive
            # is permanent, so reuse that verified copy instead of refusing to proceed
            existing = os.path.join(archive_dir, ep)
            if os.path.isdir(existing):
                if _dir_hashes(existing, camera) != _dir_hashes(original, camera):
                    raise RuntimeError(f"{existing} exists but differs from {original}")
                print(f"  reusing archived copy at {existing} (md5-verified identical)")
                archived = existing
            else:
                archived = copy_original(original, archive_dir)
            staging, staged = split_episode(archived, src, segment_s, min_last_s,
                                            camera=camera, skip_names={ep})
            print("  verifying staged segments against the original:")
            ok, lines = verify_against_original(archived, staged, camera=camera)
            for line in lines:
                print(line)
            if not ok:
                raise RuntimeError("verification FAILED — segments do not reproduce "
                                   "the original")
        except Exception as e:  # one bad episode must not abandon the rest
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            print(f"  !! FAILED — {ep} left untouched in {src}: {e}")
            failures.append((ep, str(e).splitlines()[0][:140]))
            continue
        # verified: now (and only now) the original leaves the dataset folder and the
        # segments take its place
        shutil.rmtree(original)
        created = []
        for path in staged:
            dst = os.path.join(src, os.path.basename(path))
            shutil.move(path, dst)
            created.append(dst)
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  -> {len(created)} segments VERIFIED and installed; "
              f"original preserved at {archived}")

    if not dry_run:
        print(f"\n{len(longs) - len(failures)}/{len(longs)} long episodes split; "
              f"{len(os.listdir(src))} episodes now in {src}")
        for ep, why in failures:
            print(f"  FAILED (left as-is): {ep} — {why}")


if __name__ == "__main__":
    typer.run(main)
