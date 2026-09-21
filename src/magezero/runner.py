"""
runner.py — orchestrates the full MageZero training pipeline.

For each generation:
  1. Resolve curriculum settings
  2. For each opponent (up to max_jvms in parallel):
       reserve session ids for primary + opponent
       build a mutated game.yml
       start servers (skipped for offline)
       launch JVM, write hdf5 files, stop servers
       parse the JVM log into game / behaviour metrics
  3. For every deck being trained this gen (primary, plus `cotrain` opponents):
       dataset metrics, optional dataset_stats plots, eval previous model on new data,
       move testing/ → training/, archive out-of-window data, train, restore archive
  4. Optional: strength eval of the new checkpoints against fixed baselines
  5. Record gen completion in the run file

Co-training: an opponent with `cotrain: true` plays with its own network on OPPONENT_PORT
and trains on its side of the same games, so both decks of a matchup improve together.
All metrics land in runs/<id>/metrics.jsonl and runs/<id>/games.jsonl (see metrics.py).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from magezero import metrics
from magezero.util.config import (
    CurriculumConfig,
    GenSettings,
    Opponent,
    RunConfig,
    load_all,
    resolve_gen,
)


# constants

PRIMARY_PORT = 50052
OPPONENT_PORT = 50053
TMP_DIR = Path(".mz_tmp")
RUNS_DIR = Path("runs")
SRC = "src/magezero"
PYTHON = sys.executable
EPOCHS_BOOTSTRAP = 2
EPOCHS_ONLINE = 1


# deck-level state (session counter)

def deck_state_path(deck: str) -> Path:
    return Path("models") / deck / "state.json"


def next_session_id(deck: str) -> int:
    """Reserve and return the next session id for this deck."""
    p = deck_state_path(deck)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(p.read_text()) if p.exists() else {"next_session_id": 1}
    sid = state["next_session_id"]
    state["next_session_id"] = sid + 1
    p.write_text(json.dumps(state, indent=2))
    return sid


# run folders (history + active state)

def find_active_run(deck: str, version: int) -> Optional[Path]:
    if not RUNS_DIR.exists():
        return None
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        json_file = d / "run.json"
        if not json_file.exists():
            continue
        data = json.loads(json_file.read_text())
        if (data.get("completed_at") is None
                and data.get("abandoned_at") is None
                and data["primary"]["deck"] == deck
                and data["primary"]["version"] == version):
            return d
    return None


def create_run_dir(run: RunConfig) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir()
    data = {
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "abandoned_at": None,
        "primary": {"deck": run.deck, "version": run.version},
        "opponents": [
            {"deck": o.deck, "mode": o.mode, "version": o.version, "offline": o.offline, "cotrain": o.cotrain}
            for o in run.opponents
        ],
        "run_config_snapshot": {
            "generations": run.generations,
            "games_per_gen": run.games_per_gen,
            "replay_buffer_gens": run.replay_buffer_gens,
            "start_from_version": run.start_from_version,
            "jvm": vars(run.jvm),
            "eval": vars(run.eval),
            "embed_rows": os.environ.get("MZ_EMBED_ROWS"),
            "action_vocab": os.environ.get("MZ_ACTION_VOCAB"),
            "platform": os.environ.get("MZ_PLATFORM", "local"),
        },
        "current_gen": 0,
        "stage": "init",
        "gens": {},
    }
    (run_dir / "run.json").write_text(json.dumps(data, indent=2))
    return run_dir


def update_run(run_dir: Path, **fields) -> dict:
    json_file = run_dir / "run.json"
    data = json.loads(json_file.read_text())
    data.update(fields)
    if "stage" in fields:
        data["stage_started_at"] = datetime.now().isoformat()
    json_file.write_text(json.dumps(data, indent=2))
    return data


def record_gen(run_dir: Path, gen: int, settings: GenSettings,
               sessions: dict, extra: Optional[dict] = None) -> None:
    json_file = run_dir / "run.json"
    data = json.loads(json_file.read_text())
    data["gens"][str(gen)] = {
        # deck -> opponent -> [session ids] for every deck that produced data this gen
        "sessions": sessions,
        "primary_sessions": sessions.get(data["primary"]["deck"], {}),
        "settings": {
            "td_discount": settings.td_discount,
            "prior_temperature": settings.prior_temperature,
            "priors": vars(settings.priors),
        },
        "completed_at": datetime.now().isoformat(),
        **(extra or {}),
    }
    json_file.write_text(json.dumps(data, indent=2))


# version helpers

def latest_version(deck: str) -> Optional[int]:
    d = Path("models") / deck
    if not d.exists():
        return None
    versions = []
    for sub in d.iterdir():
        if sub.is_dir() and sub.name.startswith("ver"):
            try:
                v = int(sub.name[3:])
                if (sub / "model.pt.gz").exists():
                    versions.append(v)
            except ValueError:
                pass
    return max(versions) if versions else None


def has_checkpoint(deck: str, version: int, checkpoint: Optional[str] = None) -> bool:
    name = f"{checkpoint}.pt.gz" if checkpoint else "model.pt.gz"
    return (Path("models") / deck / f"ver{version}" / name).exists()


def copy_starting_checkpoint(deck: str, start_from: Optional[int], version: int) -> None:
    """If start_from is set, seed the new ver folder from the source."""
    if start_from is None:
        return
    src = Path("models") / deck / f"ver{start_from}"
    dst = Path("models") / deck / f"ver{version}"
    if dst.exists() and (dst / "model.pt.gz").exists():
        return  # already seeded (resume case)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt.gz", "ignore.roar"):
        f = src / name
        if f.exists():
            shutil.copy(f, dst / name)


# data file path helpers

def data_file(deck: str, version: int, sid: int, opponent: str, split: str) -> Path:
    name = f"session{sid}_{deck}_vs_{opponent}.hdf5"
    return Path("data") / deck / f"ver{version}" / split / name


def parse_session_id(filename: str) -> Optional[int]:
    """session{N}_..."""
    try:
        return int(filename.split("_")[0].replace("session", ""))
    except (ValueError, IndexError):
        return None


# game.yml mutation

def _set_player(p: dict, deck: str, out: Path, ptype: str, offline: bool, settings: GenSettings) -> None:
    p["deckPath"] = str((Path("xmage/decks") / f"{deck}.dck").resolve())
    p["output_file"] = str(out.resolve())
    p["type"] = ptype
    p["mcts"]["offline_mode"] = offline
    p["mcts"]["td_discount"] = settings.td_discount
    p["priors"]["prior_temperature"] = settings.prior_temperature
    # priors only matter when a network is attached
    for head in ("binary", "priority", "target", "opponent"):
        p["priors"][head] = bool(getattr(settings.priors, head)) and not offline


def build_game_yml(base_path: str, settings: GenSettings, run: RunConfig, name: str,
                   deck_a: str, out_a: Path, offline_a: bool,
                   deck_b: str, out_b: Path, type_b: str, offline_b: bool,
                   games: int) -> str:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    _set_player(cfg["player_a"], deck_a, out_a, "mcts", offline_a, settings)
    _set_player(cfg["player_b"], deck_b, out_b, type_b, offline_b, settings)

    cfg["training"]["games"] = games
    cfg["training"]["threads"] = run.jvm.threads
    for key in ("search_budget", "timeout_ms"):
        val = getattr(run.jvm, key)
        if val is not None:
            cfg["player_a"]["mcts"][key] = val
            cfg["player_b"]["mcts"][key] = val
    cfg["server"]["port"] = PRIMARY_PORT
    cfg["server"]["opponent_port"] = OPPONENT_PORT

    out = TMP_DIR / f"game_{name}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(out)


# subprocess wrappers

def start_server(deck: str, version: int, port: int, run_dir: Path,
                 checkpoint: Optional[str] = None) -> subprocess.Popen:
    label = f"{deck} v{version}" + (f" {checkpoint}" if checkpoint else "")
    print(f"[server] start {label} on :{port}")
    log_path = run_dir / f"server_{port}.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n=== START {datetime.now().isoformat()} {label} ===\n")
    log_file.flush()

    cmd = [PYTHON, "-u", f"{SRC}/server.py", "--deck", deck, "--version", str(version), "--port", str(port)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._mz_log_file = log_file

    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(f"server process exited before becoming ready (see {log_path})")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            print(f"[server] ready on :{port}")
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    log_file.close()
    raise TimeoutError(f"server on :{port} did not come up within 600s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if hasattr(proc, "_mz_log_file"):
        proc._mz_log_file.close()


def jvm_command(game_yml_path: str, heap: str = "24g") -> tuple[list[str], Optional[str]]:
    """Return (argv, cwd). Windows keeps the bundled .bat; elsewhere java is invoked directly
    so heap size is configurable (the bundled scripts hardcode -Xmx24g)."""
    cfg = str(Path(game_yml_path).resolve())
    if sys.platform == "win32":
        return ["cmd", "/c", "xmage\\mz-xmage.bat", cfg], None
    return ["java", "-Dlog.file=magezero.log", "-Derrors.file=magezeroErrors.log",
            "-Dlog4j.configuration=file:log4j.properties", "-Xms1g", f"-Xmx{heap}", "-XX:+UseZGC",
            "--add-opens=java.base/java.lang=ALL-UNNAMED", "--enable-native-access=ALL-UNNAMED",
            "-jar", "lib/mage-magezero-1.4.58.jar", cfg], "xmage"


def launch_jvm(game_yml_path: str, log_path: Optional[Path] = None, heap: str = "24g") -> None:
    print(f"[jvm] launching with {game_yml_path}")
    cmd, cwd = jvm_command(game_yml_path, heap)
    if log_path is None:
        subprocess.run(cmd, check=True, cwd=cwd)
        return
    with open(log_path, "a") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)


def _run_logged(cmd: list[str], log_path: Path, header: str) -> None:
    with open(log_path, "a") as f:
        f.write(f"\n=== {header} {datetime.now().isoformat()} ===\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def run_train(deck: str, version: int, epochs: int, use_checkpoint: bool,
              run_dir: Path, gen: int, dense_vocab: bool = False) -> None:
    cmd = [PYTHON, "-u", f"{SRC}/train.py",
           "--deck", deck, "--version", str(version),
           "--epochs", str(epochs), "--gen", str(gen),
           "--metrics-out", str(run_dir / "metrics.jsonl")]
    if use_checkpoint:
        cmd.append("--checkpoint")
    if dense_vocab:
        cmd.append("--dense-vocab")
    _run_logged(cmd, run_dir / "train.log", f"GEN {gen} TRAIN {deck}")


def run_test(deck: str, version: int, run_dir: Path, gen: int) -> None:
    _run_logged([PYTHON, "-u", f"{SRC}/test.py", "--deck", deck, "--version", str(version),
                 "--gen", str(gen), "--metrics-out", str(run_dir / "metrics.jsonl")],
                run_dir / "test.log", f"GEN {gen} TEST {deck}")


def run_dataset_stats(deck: str, version: int, split: str,
                      run_dir: Path, gen: int) -> None:
    _run_logged([PYTHON, "-u", f"{SRC}/dataset_stats.py",
                 "--deck", deck, "--version", str(version), "--split", split],
                run_dir / "dataset_stats.log", f"GEN {gen} DATASET_STATS {deck}")


# data movement

def move_testing_to_training(deck: str, version: int) -> None:
    src = Path("data") / deck / f"ver{version}" / "testing"
    dst = Path("data") / deck / f"ver{version}" / "training"
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for f in src.glob("*.hdf5"):
        shutil.move(str(f), str(dst / f.name))


def archive_out_of_window(deck: str, version: int, gens: dict,
                          current_gen: int, window: int) -> list[Path]:
    """Move training files outside the replay window into archive/. Returns moved paths."""
    cutoff = current_gen - window + 1
    if cutoff <= 0:
        return []

    archive_ids: set[int] = set()
    for g_str, g_data in gens.items():
        if int(g_str) < cutoff:
            per_deck = g_data.get("sessions", {}).get(deck, g_data.get("primary_sessions", {}))
            for sids in per_deck.values():
                archive_ids.update(sids)

    if not archive_ids:
        return []

    training = Path("data") / deck / f"ver{version}" / "training"
    archive = Path("data") / deck / f"ver{version}" / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    moved = []
    for f in training.glob("*.hdf5"):
        sid = parse_session_id(f.name)
        if sid is not None and sid in archive_ids:
            dst = archive / f.name
            shutil.move(str(f), str(dst))
            moved.append(dst)
    return moved


def restore_from_archive(deck: str, version: int, files: list[Path]) -> None:
    training = Path("data") / deck / f"ver{version}" / "training"
    for f in files:
        shutil.move(str(f), str(training / f.name))


# metrics helpers

def refresh_dashboard(run_dir: Path) -> None:
    try:
        from magezero import report
        report.render(run_dir, run_dir / "dashboard.html")
    except Exception as e:
        print(f"[metrics] WARNING dashboard render failed: {e}")


def collect_match_metrics(run_dir: Path, log_path: Path, gen: int, kind: str,
                          deck_a: str, deck_b: str, label: dict) -> dict:
    """Parse a finished JVM log; append per-game rows and one summary row. Never raises."""
    try:
        parsed = metrics.parse_jvm_log(str(log_path), deck_a, deck_b,
                                       f"xmage/decks/{deck_a}.dck", f"xmage/decks/{deck_b}.dck")
        for g in parsed["games"]:
            metrics.append_jsonl(run_dir / "games.jsonl", {"kind": kind, "gen": gen, **label, **g})
        row = {"kind": kind, "gen": gen, **label, **metrics.summarize_games(parsed)}
        metrics.append_jsonl(run_dir / "metrics.jsonl", row)
        wr = row.get("a_winrate")
        print(f"[metrics] {kind} gen {gen} {deck_a} vs {deck_b}: {row['games_completed']} games, "
              f"{deck_a} WR={wr if wr is None else round(wr, 3)}, "
              f"failed={row['games_failed']}, mean turns={row.get('mean_turns')}")
        refresh_dashboard(run_dir)
        return row
    except Exception as e:  # metrics must never kill a training run
        print(f"[metrics] WARNING could not parse {log_path}: {e}")
        return {}


# session deadline (Kaggle and other time-limited hosts)

DEADLINE_EXIT_CODE = 75  # "paused": resume later with --resume


def out_of_time(run_dir: Path, gen: int, will_eval: bool) -> bool:
    """True if MZ_DEADLINE (unix seconds) would be passed by running one more generation.
    Estimate = slowest self-play+train so far, plus the slowest strength eval if this gen runs
    one (doubled while only gen 0's eval is known: it has no frozen:N baseline yet), plus 15%.
    MZ_GEN_ESTIMATE_S is used before any generation has finished. Stopping between generations
    keeps the run resumable: a generation killed half-way is redone from self-play."""
    deadline = os.environ.get("MZ_DEADLINE")
    if not deadline:
        return False
    gens = json.loads((run_dir / "run.json").read_text()).get("gens", {})
    base = [g["gen_seconds"] - g.get("eval_seconds", 0) for g in gens.values() if g.get("gen_seconds")]
    evals = {int(k): g["eval_seconds"] for k, g in gens.items() if g.get("eval_seconds")}
    if not base:
        estimate = float(os.environ.get("MZ_GEN_ESTIMATE_S", 0))
    else:
        estimate = max(base)
        if will_eval and evals:
            estimate += max(evals.values()) * (2 if set(evals) == {0} else 1)
        estimate *= 1.15
    left = float(deadline) - time.time()
    if estimate > left:
        print(f"[run] stopping before gen {gen}: ~{estimate / 3600:.1f} h needed, {left / 3600:.1f} h left")
        return True
    return False


# main pipeline

def run_pipeline(run: RunConfig, curriculum: CurriculumConfig,
                 base_game_yml: str = "configs/game.yml", resume: Optional[bool] = None) -> None:
    trained_decks = [run.deck] + [o.deck for o in run.opponents if o.cotrain]

    # resume detection
    active = find_active_run(run.deck, run.version)
    if active and resume is None:
        ans = input(f"Active run found: {active.name}. Resume? [Y/n] ").strip().lower()
        resume = ans in ("", "y", "yes")
    if active and resume:
        run_dir = active
        start_gen = json.loads((active / "run.json").read_text())["current_gen"]
        print(f"Resuming from gen {start_gen}")
    else:
        if active:
            update_run(active, abandoned_at=datetime.now().isoformat())
        for deck in trained_decks:
            copy_starting_checkpoint(deck, run.start_from_version, run.version)
        run_dir = create_run_dir(run)
        start_gen = 0
    print(f"[run] {run_dir}  trained decks: {trained_decks}")

    # gen loop
    for gen in range(start_gen, run.generations):
        will_eval = run.eval.games > 0 and any(o.cotrain for o in run.opponents) and gen % run.eval.every == 0
        if out_of_time(run_dir, gen, will_eval):
            update_run(run_dir, current_gen=gen, stage="paused")
            sys.exit(DEADLINE_EXIT_CODE)
        print(f"\n========== GEN {gen} ==========")
        t_gen = time.time()
        update_run(run_dir, current_gen=gen, stage="generate")
        settings = resolve_gen(curriculum, gen)
        bootstrap = run.start_from_version is None and gen == 0
        primary_offline = bootstrap

        sessions: dict[str, dict[str, list[int]]] = {}
        jobs = []
        cotrain_opp = None
        for opp in run.opponents:
            if opp.cotrain:
                if cotrain_opp is not None:
                    raise NotImplementedError("only one cotrain opponent is supported (one opponent port)")
                cotrain_opp = opp
                opp_ver, opp_offline, split = run.version, bootstrap, "testing"
            else:
                opp_ver = opp.version if opp.version is not None else 1
                opp_offline, split = opp.offline, "archive"
                if opp.mode == "mcts" and not opp_offline:
                    raise NotImplementedError("online mcts opponents need `cotrain: true` (one opponent port)")

            primary_sid = next_session_id(run.deck)
            opponent_sid = next_session_id(opp.deck)
            primary_path = data_file(run.deck, run.version, primary_sid, opp.deck, "testing")
            opponent_path = data_file(opp.deck, opp_ver, opponent_sid, run.deck, split)
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            opponent_path.parent.mkdir(parents=True, exist_ok=True)

            game_yml = build_game_yml(
                base_game_yml, settings, run, f"{run.deck}_vs_{opp.deck}",
                run.deck, primary_path, primary_offline,
                opp.deck, opponent_path, "minimax" if opp.mode == "minimax" else "mcts", opp_offline,
                run.games_per_gen,
            )
            jobs.append((opp.deck, game_yml))
            sessions.setdefault(run.deck, {}).setdefault(opp.deck, []).append(primary_sid)
            if opp.cotrain:
                sessions.setdefault(opp.deck, {}).setdefault(run.deck, []).append(opponent_sid)

        failed = []
        servers = []
        try:
            if not primary_offline:
                servers.append(start_server(run.deck, run.version, PRIMARY_PORT, run_dir))
            if cotrain_opp is not None and not bootstrap:
                servers.append(start_server(cotrain_opp.deck, run.version, OPPONENT_PORT, run_dir))
            with ThreadPoolExecutor(max_workers=run.max_jvms) as pool:
                futures = {}
                for deck, game_yml in jobs:
                    log_path = run_dir / f"jvm_gen{gen}_{deck}.log"
                    futures[pool.submit(launch_jvm, game_yml, log_path, run.jvm.heap)] = (deck, log_path)
                for future in as_completed(futures):
                    deck, log_path = futures[future]
                    if future.exception() is None:
                        print(f"[gen {gen}] vs {deck} done")
                    else:
                        print(f"[gen {gen}] vs {deck} FAILED: {future.exception()}")
                        failed.append(deck)
                    collect_match_metrics(run_dir, log_path, gen, "selfplay", run.deck, deck,
                                          {"primary_offline": primary_offline})
        finally:
            for s in servers:
                stop_server(s)

        if failed:
            raise RuntimeError(f"{len(failed)} matchups failed this gen: {failed}")

        gen_extra = {"generate_seconds": time.time() - t_gen}

        data = json.loads((run_dir / "run.json").read_text())
        provisional = dict(data["gens"])
        provisional[str(gen)] = {"sessions": sessions}
        for deck in trained_decks:
            testing = Path("data") / deck / f"ver{run.version}" / "testing"
            metrics.append_jsonl(run_dir / "metrics.jsonl", {
                "kind": "dataset", "gen": gen, "deck": deck,
                **metrics.dataset_stats([str(p) for p in sorted(testing.glob("*.hdf5"))])})

            # analyze new data
            if run.training.analyze_dataset:
                update_run(run_dir, stage=f"analyze:{deck}")
                run_dataset_stats(deck, run.version, "testing", run_dir, gen)

            # eval previous model on new data
            if run.training.eval_previous_model and has_checkpoint(deck, run.version):
                update_run(run_dir, stage=f"eval_prev:{deck}")
                run_test(deck, run.version, run_dir, gen)

            # move testing to training
            update_run(run_dir, stage=f"move:{deck}")
            move_testing_to_training(deck, run.version)

            # archive out-of-window data, then train
            update_run(run_dir, stage=f"train:{deck}")
            archived = archive_out_of_window(deck, run.version, provisional, gen, run.replay_buffer_gens)
            t_train = time.time()
            try:
                use_ckpt = has_checkpoint(deck, run.version)
                epochs = EPOCHS_ONLINE if use_ckpt else EPOCHS_BOOTSTRAP
                run_train(deck, run.version, epochs, use_checkpoint=use_ckpt, run_dir=run_dir, gen=gen,
                          dense_vocab=run.training.dense_vocab)
            finally:
                restore_from_archive(deck, run.version, archived)
            gen_extra[f"train_seconds_{deck}"] = time.time() - t_train

        # strength eval of the new checkpoints against fixed baselines
        if run.eval.games > 0 and cotrain_opp is not None and gen % run.eval.every == 0:
            update_run(run_dir, stage="strength_eval")
            t_eval = time.time()
            run_strength_eval(run, base_game_yml, curriculum, run_dir, gen, cotrain_opp.deck)
            gen_extra["eval_seconds"] = time.time() - t_eval

        gen_extra["gen_seconds"] = time.time() - t_gen
        record_gen(run_dir, gen, settings, sessions, gen_extra)
        metrics.append_jsonl(run_dir / "metrics.jsonl", {"kind": "gen_timing", "gen": gen, **gen_extra})
        refresh_dashboard(run_dir)

    update_run(run_dir, completed_at=datetime.now().isoformat(), stage="done")
    print(f"\n✓ Run complete: {run_dir.name}")


def run_strength_eval(run: RunConfig, base_game_yml: str, curriculum: CurriculumConfig,
                      run_dir: Path, gen: int, other_deck: str) -> None:
    """Play each freshly trained checkpoint (as player A) against fixed opponents:
         offline   — the opponent deck with network-free MCTS (constant across gens)
         minimax   — XMage's scripted minimax AI (constant across gens)
         frozen:N  — the opponent deck's gen-N checkpoint (constant once N exists)
    Results are strength-only: the HDF5 output is written to a scratch dir and deleted."""
    settings = resolve_gen(curriculum, gen)  # evaluate the checkpoint as it actually played
    scratch = run_dir / "eval_tmp"
    for deck, opp in ((run.deck, other_deck), (other_deck, run.deck)):
        for baseline in run.eval.baselines:
            frozen = None
            if baseline.startswith("frozen:"):
                frozen = int(baseline.split(":")[1])
                if frozen >= gen or not has_checkpoint(opp, run.version, f"gen{frozen}"):
                    continue  # nothing frozen to compare against yet
            elif baseline not in ("offline", "minimax"):
                print(f"[eval] unknown baseline {baseline!r}, skipping")
                continue

            scratch.mkdir(parents=True, exist_ok=True)
            name = f"eval_gen{gen}_{deck}_vs_{opp}_{baseline.replace(':', '')}"
            game_yml = build_game_yml(
                base_game_yml, settings, run, name,
                deck, scratch / f"{name}_A.hdf5", False,
                opp, scratch / f"{name}_B.hdf5", "minimax" if baseline == "minimax" else "mcts", frozen is None,
                run.eval.games,
            )
            log_path = run_dir / f"{name}.log"
            servers = []
            try:
                servers.append(start_server(deck, run.version, PRIMARY_PORT, run_dir, checkpoint=f"gen{gen}"))
                if frozen is not None:
                    servers.append(start_server(opp, run.version, OPPONENT_PORT, run_dir, checkpoint=f"gen{frozen}"))
                launch_jvm(game_yml, log_path, run.jvm.heap)
            except Exception as e:
                print(f"[eval] {name} FAILED: {e}")
            finally:
                for s in servers:
                    stop_server(s)
            collect_match_metrics(run_dir, log_path, gen, "strength_eval", deck, opp,
                                  {"agent": deck, "agent_checkpoint": f"gen{gen}", "baseline": baseline})
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="configs/run.yml")
    args = parser.parse_args()
    run_cfg, cur_cfg = load_all(args.run)
    run_pipeline(run_cfg, cur_cfg)
