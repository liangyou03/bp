#!/usr/bin/env python3
"""
Minimal BP finetune entrypoint based on ssl_clip_old/age_tune_v2.py.
Keeps the same model/loss design and only switches to BP target/data config.
"""

from age_tune_v2 import main


if __name__ == "__main__":
    main()
