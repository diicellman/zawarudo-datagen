from __future__ import annotations

import itertools
from collections.abc import Iterator

import verifiers.v1 as vf

from ..contracts import GenerationSeedData


class GenerationSeedTask(vf.Task[GenerationSeedData]):
    pass


class GenerationSeedTaskset(vf.Taskset[GenerationSeedTask, vf.TasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[GenerationSeedTask]:
        for seed in itertools.count():
            yield GenerationSeedTask(
                GenerationSeedData(
                    idx=seed,
                    name=f"slack-generation-{seed:08d}",
                    prompt=None,
                    generation_seed=seed,
                    network_allow=[],
                    network_block=["*"],
                ),
                self.config.task,
            )


__all__ = ["GenerationSeedTask", "GenerationSeedTaskset"]
