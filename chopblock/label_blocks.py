"""Tie each chopped 10-min block back to the original annotation CSV.

Join key: each block .npz stores i_s/i_e (sample offsets into the original
recording). Labels are per 30-s segment (annotatedIdx), where
segment N -> samples [(N-1)*3840, N*3840). A 10-min block = 20 segments.

Produces blocks_10min/block_labels.csv with one row per block and a
majority-vote Afib label, AND re-saves each block .npz with the raw
per-segment vote arrays so signal and labels travel together:
    seg_index    (n_seg,)    1-based annotatedIdx covered by the block
    seg_votes    (n_seg, 4)  physician vote counts per state
    seg_states   (4,)        column order for seg_votes
    seg_majority (n_seg,)    majority state index per segment (-1 = no votes)
    has_afib     scalar      block-level Afib flag
"""

import os
import glob
import numpy as np
import pandas as pd

FS = 128
SEG_LEN = 30 * FS              # 3840 samples per annotated segment
STATES = ["Has Afibs", "Too Noisy", "No Afibs", "Others"]

HERE = os.path.dirname(__file__)
ANNO_CSV = os.path.join(HERE, "Vivalink20230817", "Afib20230817_annotation_sorted.csv")
BLOCKS_DIR = os.path.join(HERE, "blocks_10min")


def segment_votes(anno_df, subject_tag):
    """Pivot long-format votes into {annotatedIdx -> {state: count}} for one subject."""
    sub = anno_df[anno_df["subjectId"] == subject_tag]
    votes = {}
    for idx, grp in sub.groupby("annotatedIdx"):
        votes[int(idx)] = grp["states"].value_counts().to_dict()
    return votes


def block_label(votes, seg_start, seg_end):
    """Aggregate votes over annotatedIdx in [seg_start, seg_end] (1-based, inclusive)."""
    totals = {s: 0 for s in STATES}
    n_seg_afib = 0
    for seg in range(seg_start, seg_end + 1):
        v = votes.get(seg)
        if not v:
            continue
        for s in STATES:
            totals[s] += v.get(s, 0)
        # majority Afib within this 30-s segment?
        if v.get("Has Afibs", 0) >= max(v.get(s, 0) for s in STATES if s != "Has Afibs") + 1:
            n_seg_afib += 1
    has_afib = int(n_seg_afib > 0)
    return has_afib, n_seg_afib, totals


def main():
    anno = pd.read_csv(ANNO_CSV)
    rows = []
    for subj_dir in sorted(os.listdir(BLOCKS_DIR)):
        full = os.path.join(BLOCKS_DIR, subj_dir)
        if not os.path.isdir(full):
            continue
        subject_tag = "Subject0{}".format(subj_dir)   # 88 -> Subject088
        votes = segment_votes(anno, subject_tag)
        for npz in sorted(glob.glob(os.path.join(full, "block_*.npz"))):
            d = dict(np.load(npz))
            i_s, i_e = int(d["i_s"]), int(d["i_e"])
            seg_start = i_s // SEG_LEN + 1            # 1-based annotatedIdx
            seg_end = i_e // SEG_LEN                  # inclusive
            has_afib, n_afib_seg, totals = block_label(votes, seg_start, seg_end)

            # build raw per-segment vote arrays for this block
            seg_indices = list(range(seg_start, seg_end + 1))
            seg_votes = np.array(
                [[votes.get(s, {}).get(state, 0) for state in STATES] for s in seg_indices],
                dtype=np.int16,
            )                                         # shape (n_seg, 4)
            seg_majority = np.array(
                [int(row.argmax()) if row.sum() > 0 else -1 for row in seg_votes],
                dtype=np.int8,
            )                                         # index into STATES, -1 = no votes

            # re-save block with signal + raw segment-level labels travelling together
            d["seg_index"] = np.array(seg_indices, dtype=np.int32)   # 1-based annotatedIdx
            d["seg_votes"] = seg_votes                               # (n_seg, 4) counts
            d["seg_states"] = np.array(STATES)                       # column order for seg_votes
            d["seg_majority"] = seg_majority                         # per-segment majority state idx
            d["has_afib"] = np.int8(has_afib)
            np.savez(npz, **d)

            rows.append({
                "subject": subject_tag,
                "block_file": os.path.relpath(npz, HERE),
                "i_s": i_s, "i_e": i_e,
                "seg_start": seg_start, "seg_end": seg_end,
                "n_afib_segments": n_afib_seg,
                "has_afib": has_afib,
                **{s: totals[s] for s in STATES},
            })
    out = pd.DataFrame(rows)
    out_path = os.path.join(BLOCKS_DIR, "block_labels.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string())
    print(f"\nWrote {len(out)} block labels -> {out_path}")
    print("Afib-positive blocks:", int(out['has_afib'].sum()), "/", len(out))


if __name__ == "__main__":
    main()
