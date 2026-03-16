import sys
import os
import call.config_vbn as cfg
from call.llm_runner import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m call.run_vbn <run_dir>")
        sys.exit(1)
    main(sys.argv[1], cfg)
