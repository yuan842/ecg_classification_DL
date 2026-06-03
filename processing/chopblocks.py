from processing.transform import Transform
import numpy as np
import os


def normalize(data):
    data = np.nan_to_num(data)
    std = np.std(data)
    if std == 0:
        return data - np.mean(data)
    return (data - np.mean(data)) / std


class ChopBlocks(Transform):
    """Chop each long recording into fixed-length non-overlapping blocks.

    Default block length is 10 minutes (`block_minutes=10`) at the given
    sampling rate `fs`. The recording-level label is replicated across every
    block produced from that recording. Tail samples shorter than one block
    are dropped (set `pad_tail=True` to zero-pad and keep the remainder).

    If `save_dir` is provided, each block is written to disk as
    `{save_dir}/{record_id}/block_{NNNN}.npz` (one subfolder per recording).
    `record_ids` may be passed to `process()`; otherwise the recording index
    is used as the folder name.
    """

    def __init__(self, fs, block_minutes=10, pad_tail=False, save_dir=None):
        input_size = int(block_minutes * 60 * fs)
        super().__init__(input_size)
        self.name = "chopblocks"
        self.fs = fs
        self.block_minutes = block_minutes
        self.pad_tail = pad_tail
        self.save_dir = save_dir

    def _save_block(self, rec_id, block_idx, block, label):
        rec_dir = os.path.join(self.save_dir, str(rec_id))
        os.makedirs(rec_dir, exist_ok=True)
        path = os.path.join(rec_dir, f"block_{block_idx:04d}.npz")
        if label is None:
            np.savez(path, signal=block)
        else:
            np.savez(path, signal=block, label=np.asarray(label))

    def process(self, X, labels=None, record_ids=None):
        new_data = []
        new_labels = []
        idmap = []
        W = self.input_size

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        for ind, sig in enumerate(X):
            sig = np.asarray(sig)
            n = len(sig)
            rec_id = record_ids[ind] if record_ids is not None else ind

            if n == W:
                blocks = [sig]
            elif n < W:
                if not self.pad_tail:
                    continue
                pad = np.zeros(W - n, dtype=sig.dtype)
                blocks = [np.concatenate([sig, pad])]
            else:
                nfull = n // W
                blocks = [sig[i * W:(i + 1) * W] for i in range(nfull)]
                rem = n - nfull * W
                if self.pad_tail and rem > 0:
                    tail = np.concatenate([sig[nfull * W:], np.zeros(W - rem, dtype=sig.dtype)])
                    blocks.append(tail)

            kept_idx = 0
            for b in blocks:
                if np.std(b) == 0:
                    continue
                nb = normalize(b)
                lab = labels[ind] if labels is not None else None
                if self.save_dir is not None:
                    self._save_block(rec_id, kept_idx, nb, lab)
                new_data.append(nb)
                if labels is not None:
                    new_labels.append(lab)
                idmap.append(ind)
                kept_idx += 1

        self.groupmap = idmap
        self.idmap = np.arange(len(new_data))

        if labels is None:
            new_data = super(ChopBlocks, self).process(new_data)
            return new_data, idmap

        new_data, new_labels, _ = super(ChopBlocks, self).process(new_data, new_labels)
        return new_data, new_labels, idmap
