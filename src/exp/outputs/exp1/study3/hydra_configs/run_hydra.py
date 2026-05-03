#!/usr/bin/env python3
"""
Hydra-based launcher for Study 3 multi-agent simulation.

This is a thin wrapper that reads Hydra configs and delegates to the
existing simulation engine (study3_multi_agent_simulation.py).

Usage:
  # Default run (4-agent, azure)
  python run_hydra.py dilemmas_csv=data/dilemmas.csv

  # Switch to 3-agent mode
  python run_hydra.py agent_mode=3agent dilemmas_csv=data/dilemmas.csv

  # Use an experiment preset
  python run_hydra.py +experiment=full_3agent dilemmas_csv=data/dilemmas.csv

  # Multi-run sweep across seeds
  python run_hydra.py --multirun +experiment=full_3agent run.seed=0,1,2,3,4 dilemmas_csv=data/dilemmas.csv

  # Cross-compare agent modes x seeds
  python run_hydra.py --multirun agent_mode=3agent,4agent run.seed=0,1,2 dilemmas_csv=data/dilemmas.csv

  # Override model
  python run_hydra.py model.name=gpt-4o-mini model.temperature=0.5 dilemmas_csv=data/dilemmas.csv

  # Dry run
  python run_hydra.py +experiment=smoke_test dilemmas_csv=data/dilemmas.csv
"""

import os
import sys
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Ensure the simulation module is importable from the same directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from study3_multi_agent_simulation import (
    load_dilemmas,
    ChatClient,
    ALL_CONDITIONS,
    ALL_CONDITIONS_3AGENT,
    simulate_episode,
    save_jsonl,
    append_jsonl,
    log_progress,
    load_completed_keys,
)

log = logging.getLogger(__name__)


def resolve_conditions(cfg: DictConfig, condition_pool):
    """Filter conditions by name if cfg.run.conditions != 'all'."""
    if cfg.run.conditions == "all":
        return list(condition_pool)
    wanted = {c.strip() for c in cfg.run.conditions.split(",") if c.strip()}
    matched = [c for c in condition_pool if c["name"] in wanted]
    if not matched:
        raise ValueError(f"No conditions matched: {wanted}")
    return matched


@hydra.main(version_base=None, config_path="conf", config_name="configs")
def main(cfg: DictConfig) -> None:
    log.info("Resolved configs:\n%s", OmegaConf.to_yaml(cfg))

    # ---- Load dilemmas ----
    dilemmas = load_dilemmas(cfg.dilemmas_csv)
    if cfg.limit and cfg.limit > 0:
        dilemmas = dilemmas[: cfg.limit]

    # ---- Select agent mode and conditions ----
    agent_mode = cfg.agent_mode.name
    schedule = cfg.agent_mode.schedule
    condition_pool = ALL_CONDITIONS_3AGENT if agent_mode == "3agent" else ALL_CONDITIONS
    conditions = resolve_conditions(cfg, condition_pool)

    log.info(
        "Agent mode: %s | Conditions: %d | Dilemmas: %d | Schedule: %s",
        agent_mode, len(conditions), len(dilemmas), schedule,
    )

    # ---- Output paths ----
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = out_dir / "transcripts.jsonl"
    summary_path = out_dir / "summaries_2.jsonl"
    error_path = out_dir / "errors.jsonl"
    progress_path = out_dir / "progress.log"

    # ---- Dry run ----
    if cfg.run.dry_run:
        plan = []
        for d in dilemmas:
            for cond in conditions:
                plan.append({
                    "qid": d.id,
                    "condition": cond["name"],
                    "ratio": cond["ratio"],
                    "rounds": cfg.run.rounds,
                    "schedule": schedule,
                    "order_mode": cfg.run.order_mode,
                    "seed": cfg.run.seed,
                    "agent_mode": agent_mode,
                    "note": "dry_run_only; real-time requires API calls",
                })
        save_jsonl(plan, out_dir / "dry_run_plan.jsonl")
        log.info("Dry-run plan written: %s (%d episodes)", out_dir / "dry_run_plan.jsonl", len(plan))
        return

    # ---- Build API client ----
    client = ChatClient(
        provider=cfg.provider.provider,
        model=cfg.model.name,
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
        endpoint=cfg.provider.endpoint,
        api_key=cfg.provider.api_key,
        api_version=cfg.provider.get("api_version", None),
        timeout=cfg.model.timeout,
    )

    # ---- Resume support ----
    done = set()
    if not cfg.run.no_resume:
        done = load_completed_keys(summary_path)
    log_progress(f"resume: found {len(done)} completed (qid,condition) pairs", progress_path)

    # ---- Main loop ----
    episode_i = 0
    last_heartbeat = time.time()

    for d in dilemmas:
        for cond in conditions:
            if (d.id, cond["name"]) in done:
                continue

            episode_i += 1
            seed_tag = f"{d.id}|{cond['name']}"

            if time.time() - last_heartbeat > cfg.run.heartbeat_seconds:
                log_progress(
                    f"heartbeat: episode={episode_i} qid={d.id} cond={cond['name']}",
                    progress_path,
                )
                last_heartbeat = time.time()

            try:
                tr, summ = simulate_episode(
                    client=client,
                    d=d,
                    cond=cond,
                    rounds=cfg.run.rounds,
                    schedule=schedule,
                    order_mode=cfg.run.order_mode,
                    seed=cfg.run.seed,
                    seed_tag=seed_tag,
                    max_tokens_per_turn=cfg.model.max_tokens,
                    agent_mode=agent_mode,
                )

                for row in tr:
                    append_jsonl(row, transcript_path)
                append_jsonl(summ, summary_path)
                done.add((d.id, cond["name"]))

                if summ.get("skipped"):
                    log_progress(f"skipped episode={episode_i} {seed_tag} ({summ.get('skip_reason', '')})", progress_path)
                elif episode_i % 25 == 0:
                    log_progress(f"completed episode={episode_i} last={seed_tag}", progress_path)

            except Exception as e:
                err_row = {
                    "qid": d.id,
                    "condition": cond["name"],
                    "ratio": cond["ratio"],
                    "seed_tag": seed_tag,
                    "error_type": type(e).__name__,
                    "error_message": str(e)[:2000],
                }
                append_jsonl(err_row, error_path)
                log_progress(
                    f"ERROR episode={episode_i} {seed_tag} {type(e).__name__}",
                    progress_path,
                )

            if cfg.run.sleep and cfg.run.sleep > 0:
                time.sleep(cfg.run.sleep)

    log_progress("DONE", progress_path)
    log.info("Wrote transcripts: %s", transcript_path)
    log.info("Wrote summaries:   %s", summary_path)
    log.info("Wrote errors:      %s", error_path)


if __name__ == "__main__":
    main()