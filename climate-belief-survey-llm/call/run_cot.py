import sys
import os
import call.config_cot as cfg
from call.llm_runner import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m call.run_cot <run_dir>")
        sys.exit(1)
    main(sys.argv[1], cfg)
