from analyzer.cli import runCli
from . import setup_logging

if __name__ == "__main__":
    args = runCli()
    setup_logging(default_level=args.log_level)
    if args:
        args.func(args)
    
