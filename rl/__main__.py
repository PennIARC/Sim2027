"""python3 -m rl <command>. Keep this guarded: spawned env workers re-import __main__."""
import sys

if __name__ == "__main__":
    from rl.cli import main
    main(sys.argv[1:])
