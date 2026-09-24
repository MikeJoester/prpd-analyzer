"""torch Dataset. 이 모듈만 torch를 필요로 한다.

학습 샘플은 매 epoch 새로 합성한다(무작위 crop × 무작위 노이즈 × gain/roll).
`set_epoch`로 seed를 바꾸므로 같은 epoch·같은 index면 항상 같은 쌍이 재현된다.
"""

from __future__ import annotations

import numpy as np

try:  # torch는 학습 단계에서만 필요하다.
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as error:  # pragma: no cover - 환경 안내용
    raise ModuleNotFoundError(
        "torch가 필요합니다. `pip install -r requirements-diffusion.txt` 후 다시 실행하세요."
    ) from error

from .noise_model import NoiseBank
from .representation import to_model_input, random_time_crop, tile_time_windows


class PairedCropDataset(Dataset):
    """(noisy, clean) crop 쌍. noisy는 조건 입력, clean은 diffusion 목표 x0."""

    def __init__(
        self,
        dataset,          # contract.AiDataset
        clean_table,      # pandas.DataFrame — clean pool
        noise_bank: NoiseBank,
        crop_width: int = 256,
        seed: int = 42,
        samples_per_file: int = 1,
    ) -> None:
        if len(clean_table) == 0:
            raise ValueError("clean pool이 비어 있습니다.")
        self._dataset = dataset
        self._sample_ids = tuple(str(value) for value in clean_table["sample_id"])
        self._labels = tuple(str(value) for value in clean_table["label"])
        self._noise_bank = noise_bank
        self._crop_width = int(crop_width)
        self._seed = int(seed)
        self._samples_per_file = max(1, int(samples_per_file))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """epoch마다 다른 crop/노이즈 조합을 뽑되 재현성은 유지한다."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._sample_ids) * self._samples_per_file

    def __getitem__(self, index: int) -> dict:
        file_index = index % len(self._sample_ids)
        rng = np.random.default_rng((self._seed, self._epoch, index))

        sample_id = self._sample_ids[file_index]
        clean_matrix = self._dataset.matrix(sample_id)
        clean_crop = random_time_crop(clean_matrix, self._crop_width, rng)

        def crop_noise(matrix: np.ndarray) -> np.ndarray:
            return random_time_crop(matrix, self._crop_width, rng)

        noisy_crop, clean_crop = self._noise_bank.make_pair(clean_crop, rng, crop_noise)

        return {
            "noisy": torch.from_numpy(to_model_input(noisy_crop)).unsqueeze(0),
            "clean": torch.from_numpy(to_model_input(clean_crop)).unsqueeze(0),
            "sample_id": sample_id,
            "label": self._labels[file_index],
        }


class FullFileWindowDataset(Dataset):
    """전체 파일을 겹치지 않는 window로 나눠 순서대로 반환한다(추론·평가용).

    반환 dict의 `file_index`/`window_index`/`pad`로 원래 파일 형태로 다시 이어붙인다.
    """

    def __init__(self, dataset, table, crop_width: int = 256) -> None:
        self._dataset = dataset
        self._sample_ids = tuple(str(value) for value in table["sample_id"])
        self._crop_width = int(crop_width)
        self._index: list[tuple[int, int, int]] = []
        for file_index, sample_id in enumerate(self._sample_ids):
            matrix = self._dataset.matrix(sample_id)
            windows, pad = tile_time_windows(matrix, self._crop_width)
            self._index.extend((file_index, window_index, pad) for window_index in range(len(windows)))

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        file_index, window_index, pad = self._index[index]
        matrix = self._dataset.matrix(self._sample_ids[file_index])
        windows, _ = tile_time_windows(matrix, self._crop_width)
        window = windows[window_index]
        return {
            "noisy": torch.from_numpy(to_model_input(window)).unsqueeze(0),
            "file_index": file_index,
            "window_index": window_index,
            "pad": pad,
        }
