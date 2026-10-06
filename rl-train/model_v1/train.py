from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_RL_TRAIN = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_RL_TRAIN))

from helpers.helper import build_deck  # noqa: E402
from env import OPPONENT_LATEST, OPPONENT_OLDER, OPPONENT_RANDOM, PokemonTCGEnv, make_env_fn  # noqa: E402
from policy import IN_KEYS, PolicyConfig, build_policy, count_parameters, make_actor_critic, save_policy  # noqa: E402

from torchrl.collectors import Collector  # noqa: E402
from torchrl.envs import ParallelEnv  # noqa: E402
from torchrl.objectives import ClipPPOLoss  # noqa: E402
from torchrl.objectives.value import GAE  # noqa: E402

KIND_NAMES = {OPPONENT_RANDOM: "random", OPPONENT_LATEST: "latest", OPPONENT_OLDER: "older"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="run", help="run directory: runs/<name>")
    p.add_argument("--agent-deck", default="decks/main_agent.csv")
    p.add_argument("--opponent-decks", default="decks/opponents", help="a directory of .csv decklists")
    p.add_argument("--no-mirror", action="store_true", help="do not add the agent's own deck to the opponents' decks")
    p.add_argument("--workers", type=int, default=16, help="ParallelEnv workers (one game each)")
    p.add_argument("--frames-per-batch", type=int, default=4096, help="agent decisions collected per iteration")
    p.add_argument("--iterations", type=int, default=1000, help="stop after this many iterations")
    p.add_argument("--epochs", type=int, default=3, help="PPO epochs per iteration")
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-final", type=float, default=0.1, help="the learning rate decays linearly to this fraction")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lmbda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--critic-coeff", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03, help="stop the epochs early when the KL passes 1.5x this")
    p.add_argument("--prize-reward", type=float, default=0.1, help="reward per Prize card the agent takes")
    p.add_argument("--snapshot-every", type=int, default=5, help="iterations between opponent snapshots")
    p.add_argument("--pool-size", type=int, default=20, help="snapshots kept in the pool")
    p.add_argument("--p-random", type=float, default=0.2, help="share of games against a random opponent")
    p.add_argument("--latest-prob", type=float, default=0.5, help="share of snapshot games vs the newest snapshot")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--encoder-layers", type=int, default=6)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads of this (the main) process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="a checkpoint to continue from")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _RL_TRAIN / p


def load_decks(args) -> tuple[list[int], list[list[int]], list[str]]:
    agent_deck = build_deck(resolve(args.agent_deck))
    paths = sorted(resolve(args.opponent_decks).glob("*.csv"))
    names = [p.stem for p in paths]
    decks = [build_deck(p) for p in paths]
    if not args.no_mirror:
        names.append("mirror")
        decks.append(agent_deck)
    assert decks, f"no opponent decks in {args.opponent_decks}"
    return agent_deck, decks, names


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    variance = target.var()
    return float("nan") if variance < 1e-8 else float(1.0 - (target - prediction).var() / variance)


def episode_stats(data, deck_names: list[str]) -> dict[str, float]:
    done = data["next", "done"].squeeze(-1)
    truncated = data["next", "truncated"].squeeze(-1)
    outcome = data["next", "outcome"].squeeze(-1)[done]
    deck = data["next", "opponent_deck"].squeeze(-1)[done]
    kind = data["next", "opponent_kind"].squeeze(-1)[done]
    finished = ~truncated[done]
    stats = {
        "episodes": float(done.sum()),
        "truncated": float(truncated.sum()),
        "win_rate": float((outcome[finished] > 0).float().mean()) if finished.any() else float("nan"),
        "loss_rate": float((outcome[finished] < 0).float().mean()) if finished.any() else float("nan"),
    }
    for index, name in enumerate(deck_names):
        sel = finished & (deck == index)
        stats[f"wr_{name}"] = float((outcome[sel] > 0).float().mean()) if sel.any() else float("nan")
        stats[f"n_{name}"] = float(sel.sum())
    for value, name in KIND_NAMES.items():
        sel = finished & (kind == value)
        stats[f"wr_vs_{name}"] = float((outcome[sel] > 0).float().mean()) if sel.any() else float("nan")
        stats[f"n_vs_{name}"] = float(sel.sum())
    return stats


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    run_dir = _RL_TRAIN / "runs" / args.name
    pool_dir, ckpt_dir = run_dir / "pool", run_dir / "checkpoints"
    for d in (pool_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    agent_deck, opponent_decks, deck_names = load_decks(args)
    print(f"run {run_dir} | opponents: {', '.join(deck_names)} | device {device}", flush=True)

    probe = PokemonTCGEnv(agent_deck, opponent_decks)
    network = build_policy(probe.encoder, PolicyConfig(d_model=args.d_model, n_encoder_layers=args.encoder_layers,
                                                         n_decoder_layers=args.decoder_layers)).to(device)
    probe.close()
    actor, critic = make_actor_critic(network)
    print(f"policy: {count_parameters(network):,} parameters", flush=True)

    loss_module = ClipPPOLoss(actor, critic, clip_epsilon=args.clip, entropy_coeff=args.entropy,
                              critic_coeff=args.critic_coeff, normalize_advantage=True)
    gae = GAE(gamma=args.gamma, lmbda=args.lmbda, value_network=critic,
              value_chunk_size=16, value_chunk_dim=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    start_iteration, frames = 0, 0
    if args.resume:
        checkpoint = torch.load(resolve(args.resume), map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration, frames = checkpoint["iteration"], checkpoint["frames"]
        print(f"resumed from {args.resume} at iteration {start_iteration} ({frames:,} frames)", flush=True)

    workers = args.workers
    frames_per_batch = max(workers, args.frames_per_batch // workers * workers)
    env_fns = [make_env_fn(agent_deck, opponent_decks, opponent_pool=pool_dir, prize_reward=args.prize_reward,
                           worker_threads=1, seed=args.seed * 1000 + i, p_random=args.p_random,
                           latest_prob=args.latest_prob) for i in range(workers)]
    collector = Collector(ParallelEnv(workers, env_fns), actor, frames_per_batch=frames_per_batch, total_frames=-1,
                          policy_device=device, env_device="cpu", storing_device=device,
                          auto_register_policy_transforms=False)

    metrics_path = run_dir / "metrics.csv"
    metrics_file = open(metrics_path, "a", newline="")
    writer: csv.DictWriter | None = None
    keep = [*IN_KEYS, "action", "action_log_prob", "advantage", "value_target", "state_value"]

    def save_checkpoint(path: Path, iteration: int) -> None:
        torch.save({"state_dict": network.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration,
                    "frames": frames, "config": vars(args)}, path)

    iteration = start_iteration
    try:
        batches = iter(collector)
        while iteration < args.iterations:
            t0 = time.time()
            data = next(batches)
            t_collect = time.time() - t0
            frames += data.numel()
            iteration += 1

            lr = args.lr * (1.0 - (1.0 - args.lr_final) * min(1.0, (iteration - 1) / max(1, args.iterations - 1)))
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.no_grad():
                gae(data)
            stats = episode_stats(data, deck_names)
            stats["reward_per_step"] = float(data["next", "reward"].mean())
            stats["explained_variance"] = explained_variance(data["state_value"], data["value_target"])
            flat = data.select(*keep).reshape(-1)

            t1 = time.time()
            totals: dict[str, float] = {}
            updates = 0
            stop = False
            for epoch in range(args.epochs):
                order = torch.randperm(flat.shape[0], device=device)
                for start in range(0, flat.shape[0], args.minibatch):
                    batch = flat[order[start:start + args.minibatch]]
                    out = loss_module(batch)
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                    optimizer.step()
                    for key in ("loss_objective", "loss_critic", "entropy", "kl_approx", "clip_fraction"):
                        totals[key] = totals.get(key, 0.0) + float(out[key].detach())
                    totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm)
                    updates += 1
                    if float(out["kl_approx"]) > 1.5 * args.target_kl:
                        stop = True
                        break
                if stop:
                    break
            t_update = time.time() - t1
            collector.update_policy_weights_()

            if iteration % args.snapshot_every == 0:
                save_policy(network, pool_dir / f"snapshot_{iteration:06d}.pt", iteration=iteration, frames=frames)
                for old in sorted(pool_dir.glob("snapshot_*.pt"))[:-args.pool_size]:
                    old.unlink(missing_ok=True)
            save_checkpoint(ckpt_dir / "latest.pt", iteration)
            if iteration % args.checkpoint_every == 0:
                save_checkpoint(ckpt_dir / f"iter_{iteration:06d}.pt", iteration)

            row = {"iteration": iteration, "frames": frames, "lr": lr, "collect_s": t_collect, "update_s": t_update,
                   "fps": data.numel() / (time.time() - t0), "updates": updates, "early_stop": int(stop), **stats,
                   **{k: v / max(updates, 1) for k, v in totals.items()}}
            if writer is None:
                writer = csv.DictWriter(metrics_file, fieldnames=list(row), extrasaction="ignore")
                if metrics_path.stat().st_size == 0:
                    writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()
            print(f"[{iteration:4d}] {frames:>9,} frames {row['fps']:6.0f} fps | games {stats['episodes']:3.0f} "
                  f"win {stats['win_rate']:.2f} (random {stats['wr_vs_random']:.2f} latest {stats['wr_vs_latest']:.2f} "
                  f"older {stats['wr_vs_older']:.2f}) | reward/step {stats['reward_per_step']:+.3f} ev "
                  f"{stats['explained_variance']:+.2f} | ent {row['entropy']:.2f} kl {row['kl_approx']:.4f} "
                  f"clip {row['clip_fraction']:.2f} | collect {t_collect:.0f}s update {t_update:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("interrupted; saving a checkpoint", flush=True)
        save_checkpoint(ckpt_dir / "latest.pt", iteration)
    finally:
        metrics_file.close()
        collector.shutdown()


if __name__ == "__main__":
    main()
