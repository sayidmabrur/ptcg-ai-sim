"""A/B the decode floor on the packaged bundle, identical weights both arms.

Needs the __main__ guard: arena spawns workers, and spawn re-imports this
module in each child, so top-level work recurses into new pools.
"""
import os, sys
sys.path.insert(0, '.')


def main():
    from arena import versus_field
    for label, flag in (("no floor", "0"), ("floored at 1", "1")):
        os.environ["SUBMISSION_DECODE_FLOOR"] = flag
        print(f"--- {label}", flush=True)
        versus_field("bundle:submission_ppo", 80, 5, {"max_steps": 500, "seed": 0}, 0)
        print(flush=True)


if __name__ == "__main__":
    main()
