"""ARDD 1D 트랙의 torch Dataset. 이 모듈만 torch를 필요로 한다.

두 가지 공급 방식이 있다.

* **캐시(기본)** — `pairs.build_profile_pairs`로 미리 만든 `(N, M, 2, 128)` 실현 중 하나를 뽑는다.
  raw memmap을 전혀 읽지 않아 학습이 빠르고, 같은 seed에서 완전히 재현된다.
* **온-더-플라이** — 매 접근마다 raw에서 새로 합성한다. 실현 다양성이 최대지만 느리다.

두 방식 모두 `set_epoch`로 epoch마다 다른 조합을 뽑되, `(seed, epoch, index)`로 rng를 만들어
같은 epoch·같은 index면 항상 같은 쌍이 나온다(기존 `PairedCropDataset`과 같은 규약).
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

from ..data.profile import to_model_input
from .pairs import PairFactory


class CachedProfilePairDataset(Dataset):
    """미리 만들어 둔 profile 쌍에서 실현 하나를 뽑는다."""

    def __init__(self, payload: dict[str, np.ndarray], seed: int = 42, samples_per_file: int = 1):
        clean = np.asarray(payload["clean"], dtype=np.float32)
        noisy = np.asarray(payload["noisy"], dtype=np.float32)
        if clean.shape[0] == 0:
            raise ValueError("profile 쌍이 비어 있습니다. pool 선택을 확인하세요.")
        if noisy.shape[0] != clean.shape[0]:
            raise ValueError(f"clean/noisy 개수 불일치: {clean.shape[0]} vs {noisy.shape[0]}")
        self._clean = clean
        self._noisy = noisy
        self._sample_ids = [str(value) for value in payload["sample_ids"]]
        self._labels = [str(value) for value in payload["labels"]]
        self._realizations = noisy.shape[1]
        self._seed = int(seed)
        self._samples_per_file = max(1, int(samples_per_file))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._sample_ids) * self._samples_per_file

    def __getitem__(self, index: int) -> dict:
        file_index = index % len(self._sample_ids)
        rng = np.random.default_rng((self._seed, self._epoch, index))
        realization = int(rng.integers(0, self._realizations))
        return {
            "noisy": torch.from_numpy(to_model_input(self._noisy[file_index, realization])),
            "clean": torch.from_numpy(to_model_input(self._clean[file_index])),
            "sample_id": self._sample_ids[file_index],
            "label": self._labels[file_index],
        }


class LiveProfilePairDataset(Dataset):
    """매 접근마다 raw에서 새로 합성한다. 캐시를 끌 때 사용."""

    def __init__(
        self,
        dataset,          # contract.AiDataset
        clean_table,      # pandas.DataFrame — clean pool
        factory: PairFactory,
        seed: int = 42,
        samples_per_file: int = 1,
    ) -> None:
        if len(clean_table) == 0:
            raise ValueError("clean pool이 비어 있습니다.")
        self._dataset = dataset
        self._factory = factory
        self._sample_ids = [str(value) for value in clean_table["sample_id"]]
        self._labels = [str(value) for value in clean_table["label"]]
        self._seed = int(seed)
        self._samples_per_file = max(1, int(samples_per_file))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._sample_ids) * self._samples_per_file

    def __getitem__(self, index: int) -> dict:
        file_index = index % len(self._sample_ids)
        rng = np.random.default_rng((self._seed, self._epoch, index))
        sample_id = self._sample_ids[file_index]
        noisy, clean = self._factory.make_pair(self._dataset.matrix(sample_id), rng)
        return {
            "noisy": torch.from_numpy(to_model_input(noisy)),
            "clean": torch.from_numpy(to_model_input(clean)),
            "sample_id": sample_id,
            "label": self._labels[file_index],
        }


@torch.no_grad()
def denoise_profiles(
    diffusion,
    model,
    noisy_profiles: np.ndarray,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 64,
    device=None,
    seed: int = 0,
) -> np.ndarray:
    """`(N, 2, 128)` `[0, 255]` noisy profile → 같은 shape의 복원 profile.

    기존 `diffusion.sampler.ddim_sample`을 **수정 없이** 그대로 쓴다. 샘플러는
    `torch.randn(condition.shape)`와 채널 concat만 사용하므로 1D 텐서에서도 동작한다.
    """
    from ..data.profile import from_model_output
    from ..diffusion.sampler import ddim_sample

    array = np.asarray(noisy_profiles, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    generator = torch.Generator(device=device or "cpu").manual_seed(int(seed))

    outputs = []
    for start in range(0, array.shape[0], batch_size):
        chunk = to_model_input(array[start : start + batch_size])
        condition = torch.from_numpy(chunk).to(device)
        restored = ddim_sample(
            diffusion, model, condition, steps=steps, eta=eta, generator=generator
        )
        outputs.append(restored.cpu().numpy())

    return from_model_output(np.concatenate(outputs, axis=0))
