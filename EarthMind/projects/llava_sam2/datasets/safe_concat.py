"""ConcatDataset that survives rare unloadable samples during multi-day training runs."""
import bisect
import random
import sys

from xtuner.dataset import ConcatDataset


class SafeConcatDataset(ConcatDataset):
    """xtuner `ConcatDataset` whose `__getitem__` replaces a sample that raises (corrupt image,
    malformed annotation) by a random other sample and reports it on stderr, instead of killing the
    DataLoader worker and with it the whole run. Loadable samples are returned unchanged.
    """

    max_retries = 20

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                return super().__getitem__(idx)
            except Exception as e:  # noqa: BLE001
                j = idx if idx >= 0 else len(self) + idx
                ds = bisect.bisect_right(self.cumulative_sizes, j)
                print(f'[SafeConcatDataset] sample {idx} (dataset #{ds}, {type(self.datasets[ds]).__name__}) '
                      f'failed: {type(e).__name__}: {e} -> using a random other sample',
                      file=sys.stderr, flush=True)
                idx = random.randrange(len(self))
        return super().__getitem__(idx)
