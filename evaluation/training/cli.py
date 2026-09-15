"""``train`` — the trainer's console script: ``sasrec`` (``training.train``) and
``upload-checkpoint`` (``training.checkpoints``)."""

from __future__ import annotations

import click

from training.checkpoints import upload_checkpoint
from training.train import sasrec


@click.group()
def main() -> None:
    """gSASRec training and checkpoint upload."""


main.add_command(sasrec)
main.add_command(upload_checkpoint, "upload-checkpoint")


if __name__ == "__main__":
    main()

__all__ = ["main"]
