"""Arena-measure the resume candidates at n=350/opponent so the next long run
starts from the better one. Bundles, not checkpoints: the bundle is what the
decode floor lives in, and it is what a submission would ship."""
import subprocess, sys
from pathlib import Path

CANDIDATES = {
    "preserved-ep9232": "checkpoints/preserved/eval0.517-ep9232.pt",
    "shaped-ep11792": "checkpoints/ppo-exploit/latest.pt",
}


def main():
    from arena import versus_field
    for name, ckpt in CANDIDATES.items():
        out = f"/tmp/bundle_{name}"
        subprocess.run([sys.executable, "make_submission.py",
                        "--checkpoint", ckpt, "--out", out], check=True,
                       capture_output=True)
        print(f"--- {name}  ({ckpt})", flush=True)
        versus_field(f"bundle:{out}", 350, 11, {"max_steps": 500, "seed": 0}, 0)
        print(flush=True)


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
