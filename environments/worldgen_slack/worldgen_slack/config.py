"""Pinned native harness/runtime defaults; TOML can override each agent."""

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.harnesses.null.harness import NullHarnessConfig
from verifiers.v1.harnesses.rlm.harness import RLMHarnessConfig

RLM_REVISION = "ad081dbcf5e8c1d4e5b431b4b7d4dd5f30b7367c"
IMAGE = "python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a"


def role(model: str, *, author: bool = False, solver: bool = False) -> vf.AgentConfig:
    return vf.AgentConfig(
        model=model,
        harness=(
            NullHarnessConfig(id="null")
            if solver
            else RLMHarnessConfig(
                id="rlm",
                version=RLM_REVISION,
                max_depth=0,
                builtin_skills=["edit"] if author else [],
            )
        ),
        runtime=vf.PrimeConfig(
            vm=True,
            image=IMAGE,
            workdir="/task",
            allow=[],
            block=["*"],
            cpu=2,
            memory=4,
            disk=8,
            idle_timeout=1800,
        ),
        sampling=vf.Sampling(temperature=0.5 if author else 0.0, max_tokens=16_000),
        max_turns=160 if not solver else 40,
        timeout=TimeoutConfig(setup=600, rollout=3600, finalize=120),
        retries=RetryConfig(max_retries=0),
    )
