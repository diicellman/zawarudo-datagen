"""Paired judge reviews and one author revision; never publishes a dataset."""

import argparse
import ast
import asyncio
import fcntl
import json
import random
import re
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.clients import EvalClientConfig, ModelContext
from worldgen_slack.dataset import atomic_json, read_json
from worldgen_slack.slack.api import digest
from worldgen_slack.slack.models import SafeId, StrictModel
from .agents.author import FILE_GUIDE
from .agents.builder import BuilderTask, BUILDER_GUIDE
from .agents.synthesizer import SynthesizerTask, CATALOG_GUIDE
from .agents.judge import JudgeTask, REVIEW_GUIDE, REVIEW_PROMPT, review_payload
from .config import ROOT, PipelineConfig
from .contracts import Candidate, Catalog, Verdict, validate_candidate
from .env import require_trace


class Case(StrictModel):
    id: SafeId
    source: Path
    phase: Literal["catalog", "world"]
    repair: bool = False


class Study(StrictModel):
    output: Path
    guide_addendum: Path
    seed: int = Field(ge=0)
    cases: list[Case]

    @model_validator(mode="after")
    def fixed_scope(self):
        if len(self.cases) != 6 or len({c.id for c in self.cases}) != 6:
            raise ValueError("the study requires six distinct fixed cases")
        if sum(c.repair for c in self.cases) != 3:
            raise ValueError("the study requires three paired repair cases")
        return self


def freeze(study):
    source = subprocess.check_output(
        ["git", "show", "5ae9671:generators/worldgen_slack/agents/judge.py"], cwd=ROOT, text=True
    )
    prompts = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"REVIEW_GUIDE", "REVIEW_PROMPT"}
    }
    if prompts != {"REVIEW_GUIDE": REVIEW_GUIDE, "REVIEW_PROMPT": REVIEW_PROMPT}:
        raise ValueError("production prompts differ from the frozen baseline")
    guides = {"baseline": prompts["REVIEW_GUIDE"]}
    guides["revised"] = guides["baseline"] + (ROOT / study.guide_addendum).read_text()
    payloads = {}
    for case in study.cases:
        raw = read_json(ROOT / case.source)
        catalog = Catalog.model_validate(raw["catalog"])
        candidate = Candidate.model_validate(raw["candidate"]) if case.phase == "world" else None
        payloads[case.id] = review_payload(catalog, candidate, [t.id for t in catalog.tasks], case.phase)
    config = PipelineConfig(max_concurrent_agents=2, timeout={"episode": 14400, "finalize": 300})
    manifest = {
        "study": study.model_dump(mode="json"),
        "baseline_commit": "5ae9671",
        "guides": guides,
        "entry_prompt": REVIEW_PROMPT,
        "cases": {key: {"hash": digest(value), "payload": value} for key, value in payloads.items()},
        "env": config.model_dump(mode="json"),
        "author_instructions_hash": digest([FILE_GUIDE, CATALOG_GUIDE, BUILDER_GUIDE]),
        "lock_hash": digest((ROOT / "uv.lock").read_text()),
        "dispatch_stop_usd": 25,
        "reported_budget_usd": 30,
    }
    return manifest, config, payloads, guides


def has_feedback(verdict):
    if not verdict.approved:
        return True
    section = re.search(r"Nonblocking improvements:\s*(.*)", verdict.summary, re.I | re.S)
    return bool(section and section[1].strip().lower().rstrip(".") not in {"", "none"})


def author_context(payload, verdict):
    catalog = Catalog.model_validate(payload["catalog"])
    context = {
        "phase": payload["phase"],
        "workspace_id": catalog.workspace_id,
        "feedback": verdict.model_dump_json(),
        "instructions": "Make at most one revision addressing blocking issues and any Nonblocking improvements "
        "in the summary. Preserve stable task IDs, group assignments, and reader identities. "
        "Preserve supported evidence and chronology; do not change artifacts owned by another role.",
    }
    if payload["phase"] == "catalog":
        context.update(
            sector=catalog.sector,
            task_count=len(catalog.tasks),
            group_size=max(sum(t.group_id == g.id for t in catalog.tasks) for g in catalog.groups),
            previous_output=payload["catalog"],
        )
    else:
        context.update(
            group_id="all",
            catalog=payload["catalog"],
            required_task_ids=[t.id for t in catalog.tasks],
            previous_output=payload["candidate"],
        )
    # Each arm owns a detached copy, including its nested candidate lists.
    return json.loads(json.dumps(context))


def revised_payload(payload, raw):
    catalog = Catalog.model_validate(payload["catalog"])
    if payload["phase"] == "catalog":
        revised = Catalog.model_validate_json(raw)
        if (
            revised.workspace_id,
            revised.sector,
            [(g.id) for g in revised.groups],
            {(t.id, t.group_id, t.actor_id) for t in revised.tasks},
        ) != (
            catalog.workspace_id,
            catalog.sector,
            [(g.id) for g in catalog.groups],
            {(t.id, t.group_id, t.actor_id) for t in catalog.tasks},
        ):
            raise ValueError("catalog revision changed stable identities or requested scope")
        return review_payload(revised, None, [t.id for t in revised.tasks], "catalog")
    candidate = Candidate.model_validate_json(raw)
    report = validate_candidate(catalog, candidate, [t.id for t in catalog.tasks])
    if not report["ok"]:
        raise ValueError(json.dumps(report["errors"]))
    return review_payload(catalog, candidate, [t.id for t in catalog.tasks], "world")


def accounting(root):
    cost, unpriced = 0.0, 0
    for path in (root / "traces").glob("*.json"):
        trace = read_json(path)
        for usage in [trace.get("usage"), *trace.get("extra_usage", [])]:
            if usage and usage.get("cost") is not None:
                cost += usage["cost"]
            else:
                unpriced += 1
    return {"reported_model_cost": round(cost, 6), "unpriced_usage_records": unpriced}


class AblationEnv(vf.Env[PipelineConfig]):
    def __init__(self, config, study, payloads, guides):
        super().__init__(config)
        self.study, self.payloads, self.guides = study, payloads, guides
        self.root = ROOT / study.output
        self.slots = asyncio.Semaphore(2)

    async def setup(self, agents):
        for name in ("synthesizer", "builder", "judge", "solver"):
            getattr(agents, name).trainable = False

    def trace(self, trace):
        record = trace.to_record()
        record["usage"] = trace.usage.model_dump(mode="json") if trace.usage else None
        atomic_json(self.root / "traces" / (trace.id + ".json"), record)

    async def job(self, name, operation):
        async with self.slots:
            path = self.root / "jobs" / name / "result.json"
            if path.exists():
                result = read_json(path)
                if result["status"] == "running":
                    result.update(status="interrupted", reason="Reserved before interruption; not retried.")
                    atomic_json(path, result)
                return result
            if accounting(self.root)["reported_model_cost"] >= 25:
                result = {"status": "budget_stopped"}
                atomic_json(path, result)
                return result
            start = time.time()
            atomic_json(path, {"status": "running", "started": start})
            print(json.dumps({"event": "started", "job": name}), flush=True)
            try:
                result = {"status": "complete", **await operation()}
            except Exception as error:
                # Independent experimental arms must survive another arm's failure; never retry it.
                result = {"status": "failed", "error_type": type(error).__name__, "reason": str(error)}
            result.update(started=start, elapsed_seconds=time.time() - start)
            atomic_json(path, result)
            print(
                json.dumps(
                    {"event": "finished", "job": name, "status": result["status"], **accounting(self.root)}
                ),
                flush=True,
            )
            return result

    async def review(self, agents, payload, guide, name):
        atomic_json(self.root / "jobs" / name / "input.json", payload)
        task = JudgeTask.create(payload, candidate_id=name, review_guide=guide)
        trace = await agents.judge.run(task)
        self.trace(trace)
        require_trace(trace)
        return {"verdict": trace.info["verdict"], "trace_id": trace.id}

    async def revise(self, agents, payload, verdict, name):
        context = author_context(payload, verdict)
        atomic_json(self.root / "jobs" / name / "input.json", context)
        kind = SynthesizerTask if payload["phase"] == "catalog" else BuilderTask
        agent = agents.synthesizer if payload["phase"] == "catalog" else agents.builder
        task = kind.create(context, name)
        trace = None
        try:
            async with agent.provision(task) as runtime:
                async with agent.interaction(task, runtime=runtime) as interaction:
                    trace = interaction.trace
                    segment = await interaction.turn()
                    if segment.terminated:
                        raise RuntimeError("author interaction exhausted")
                    raw = (await runtime.read("/task/output.json", max_bytes=24_000_000)).decode()
                    atomic_json(self.root / "jobs" / name / "author_output.json", {"text": raw})
            require_trace(trace)
            updated = revised_payload(payload, raw)
            atomic_json(self.root / "jobs" / name / "payload.json", updated)
            return {"payload": updated, "trace_id": trace.id, "changed": digest(updated) != digest(payload)}
        finally:
            if trace is not None:
                self.trace(trace)

    async def arm(self, case, variant, agents):
        name = case.id + "-" + variant
        payload = self.payloads[case.id]
        initial = await self.job(
            name + "-review", lambda: self.review(agents, payload, self.guides[variant], name + "-review")
        )
        if not case.repair or initial["status"] != "complete":
            return
        verdict = Verdict.model_validate(initial["verdict"])
        if has_feedback(verdict):
            revision = await self.job(
                name + "-revision", lambda: self.revise(agents, payload, verdict, name + "-revision")
            )
            if revision["status"] != "complete":
                return
            payload = revision["payload"]
        else:
            atomic_json(
                self.root / "jobs" / (name + "-revision") / "result.json",
                {
                    "status": "unchanged",
                    "changed": False,
                    "payload": payload,
                    "reason": "No blocking issue or labeled improvement suggestion.",
                },
            )
        await self.job(
            name + "-verification",
            lambda: self.review(agents, payload, self.guides["baseline"], name + "-verification"),
        )

    async def run(self, task, agents):
        # Finish all first reviews before repairs so changing spend does not favor early cases.
        jobs = [(case, variant) for case in self.study.cases for variant in self.guides]
        random.Random(self.study.seed).shuffle(jobs)
        await asyncio.gather(
            *(
                self.job(
                    c.id + "-" + v + "-review",
                    lambda c=c, v=v: self.review(
                        agents, self.payloads[c.id], self.guides[v], c.id + "-" + v + "-review"
                    ),
                )
                for c, v in jobs
            )
        )
        await asyncio.gather(*(self.arm(c, v, agents) for c, v in jobs if c.repair))


def readable(payload):
    catalog = payload["catalog"]
    lines = ["# Catalog", "", json.dumps(catalog, indent=2, ensure_ascii=False)]
    if "candidate" in payload:
        candidate = payload["candidate"]
        lines = ["# Questions and answer specifications", "", json.dumps(catalog["tasks"], indent=2)]
        world = candidate["snapshot"]
        lines += ["", "# People", "", json.dumps(world["users"], indent=2)]
        for channel in world["conversations"]:
            lines += ["", f"## {channel.get('name') or channel['id']}", "", json.dumps(channel)]
            for m in sorted(
                (m for m in world["messages"] if m["conversation_id"] == channel["id"]),
                key=lambda m: (m["timestamp"], m["id"]),
            ):
                lines += [
                    "",
                    f"**{m['id']} | {m['timestamp']} | {m['author_id']} | reply to {m.get('thread_root_id')}**",
                    "",
                    m["text"],
                ]
        lines += ["", "# Evidence bindings", "", json.dumps(candidate["bindings"], indent=2)]
    return "\n".join(lines) + "\n"


def report(study):
    root = ROOT / study.output
    jobs = {str(p.parent.name): read_json(p) for p in (root / "jobs").glob("*/result.json")}
    pairs, key = [], {}
    rng = random.Random(study.seed)
    manifest = read_json(root / "manifest.json")
    for index, case in enumerate((c for c in study.cases if c.repair), 1):
        variants = ["baseline", "revised"]
        rng.shuffle(variants)
        folder = root / "blind" / f"pair-{index}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "before.md").write_text(readable(manifest["cases"][case.id]["payload"]))
        key[f"pair-{index}"] = dict(zip(("left", "right"), variants, strict=True)) | {"case": case.id}
        for side, variant in zip(("left", "right"), variants, strict=True):
            revision = jobs.get(case.id + "-" + variant + "-revision", {})
            text = (
                readable(revision["payload"])
                if "payload" in revision
                else "No valid revised artifact is available.\n"
            )
            (folder / (side + ".md")).write_text(text)
        pairs.append(
            {
                "pair": index,
                "preference": None,
                "validity": None,
                "richness": None,
                "realism": None,
                "reason": "",
            }
        )
    atomic_json(root / "unblinding-key.json", key)
    if not (root / "blind/ratings.json").exists():
        atomic_json(root / "blind/ratings.json", pairs)
    (root / "blind/README.md").write_text(
        "# Blind comparison\n\nRead before.md, then left.md and right.md for each pair. "
        "Prefer valid, substantive, credible work; message count alone is not quality. "
        "Record left/right/tie/neither and reasons in ratings.json. Catalog outputs are plans, "
        "not newly supported Slack worlds. Do not open the unblinding key before rating.\n"
    )
    summary = {
        **accounting(root),
        "execution_status": "complete"
        if len(jobs) == 24 and all(v["status"] in {"complete", "unchanged"} for v in jobs.values())
        else "incomplete",
        "job_statuses": {k: v["status"] for k, v in jobs.items()},
        "human_review": "pending",
        "production_guide": "baseline",
        "published": False,
    }
    atomic_json(root / "summary.json", summary)
    return summary


async def run(study, manifest, config, payloads, guides):
    root = ROOT / study.output
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "manifest.json"
        if path.exists() and read_json(path) != manifest:
            raise ValueError("frozen study inputs changed; use a new output directory")
        atomic_json(path, manifest)
        for name, guide in guides.items():
            folder = root / "prompts"
            folder.mkdir(exist_ok=True)
            (folder / (name + ".md")).write_text(guide)
        env = AblationEnv(config, study, payloads, guides)
        context = ModelContext(model=config.solver.model, client=EvalClientConfig(), sampling=vf.Sampling())
        async with env.serving():
            episode = await env.run_episode(
                vf.Task(vf.TaskData(prompt="Run the fixed judge ablation.")), context
            )
        for trace in episode.traces:
            env.trace(trace)
        atomic_json(root / f"episode-{episode.id}.json", episode.to_record())
        return report(study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    study = Study.model_validate_json(json.dumps(tomllib.loads(args.config.read_text())))
    manifest, config, payloads, guides = freeze(study)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cases": {k: v["hash"] for k, v in manifest["cases"].items()},
                    "maximum_reviews": 18,
                    "maximum_author_revisions": 6,
                    "dispatch_stop_usd": 25,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(asyncio.run(run(study, manifest, config, payloads, guides)), indent=2))


if __name__ == "__main__":
    main()
