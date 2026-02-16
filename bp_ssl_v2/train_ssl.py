#!/usr/bin/env python3
"""
Compatibility entrypoint.

Keeps the old command style (`python train_ssl.py`) while delegating to
`clip_pretrain.py`, so behavior stays aligned with `bp_recode_v1`.
"""

from clip_pretrain import main


if __name__ == "__main__":
    main()
