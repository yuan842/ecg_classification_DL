"""Chop the Vivalink20230817 ECG recordings into 10-minute blocks.

Reads each subject's continuous `ishne.ecg` (ISHNE format: 551-byte header
then int16 samples at 128 Hz) and writes non-overlapping 10-minute blocks
into per-subject subfolders. Chopping is continuous: every sample is kept,
no block is dropped (the final block per subject may be shorter than 10 min).

Output layout:
    blocks_10min/<subjectId>/block_0000.npz   # {signal, fs, i_s, i_e}
    blocks_10min/<subjectId>/block_0001.npz
    ...
"""

import os
import numpy as np

FS = 128                      # sampling rate (Hz)
BLOCK_MINUTES = 10
BLOCK_LEN = BLOCK_MINUTES * 60 * FS   # 76800 samples
ISHNE_HEADER_BYTES = 551

DATA_DIR = os.path.join(os.path.dirname(__file__), "Vivalink20230817")
OUT_DIR = os.path.join(os.path.dirname(__file__), "blocks_10min")


def read_ishne_ecg(path):
    with open(path, "rb") as f:
        f.read(ISHNE_HEADER_BYTES)
        raw = f.read()
    return np.frombuffer(raw, dtype=np.int16)


def chop(signal, block_len=BLOCK_LEN):
    # Chop continuously across the whole recording: every sample lands in a
    # block, nothing is dropped. The final block may be shorter than block_len.
    i = 0
    s = 0
    n = len(signal)
    while s < n:
        e = min(s + block_len, n)
        yield i, s, e, signal[s:e]
        s = e
        i += 1


def main():
    subjects = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, d, "ishne.ecg"))
    )
    print(f"Found {len(subjects)} subjects: {subjects}")

    total = 0
    for subj in subjects:
        sig = read_ishne_ecg(os.path.join(DATA_DIR, subj, "ishne.ecg"))
        out_subdir = os.path.join(OUT_DIR, subj)
        os.makedirs(out_subdir, exist_ok=True)

        count = 0
        last_len = 0
        for idx, s, e, block in chop(sig):
            np.savez(
                os.path.join(out_subdir, f"block_{idx:04d}.npz"),
                signal=block, fs=FS, i_s=s, i_e=e,
            )
            count += 1
            last_len = e - s
        total += count
        tail = "" if last_len == BLOCK_LEN else \
            f", last block partial: {last_len} samples / {round(last_len/FS,1)} s"
        print(f"  {subj}: {len(sig)} samples -> {count} blocks "
              f"(nothing dropped{tail})")

    print(f"Done. {total} blocks written under {OUT_DIR}")


if __name__ == "__main__":
    main()
