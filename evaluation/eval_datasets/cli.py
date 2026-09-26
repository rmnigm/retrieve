"""``eval-data`` — one console script over the per-dataset ETL scripts (``etl/``) and the Hub
commands (``hub``). The ETL modules are ``argparse`` programs; each is mounted as a
pass-through subcommand, so ``eval-data arxiv prep --output-dir …`` is ``arxiv.main([...])``
and ``eval-data arxiv --help`` is argparse's help."""

from __future__ import annotations

import importlib
import sys

import click

from eval_datasets.hub import fetch, publish, publish_checkpoint

ETL = {
    "arxiv": "arxiv",
    "goodreads": "goodreads",
    "yambda": "yambda",
    "synth-arxiv": "synth_arxiv",
    "yfcc": "yfcc",
    "yfcc-check-gt": "yfcc_check_gt",
    "pubmed": "pubmed",
    "kuairand": "kuairand",
    "openalex": "openalex",
}


@click.group()
def main() -> None:
    """Dataset ETL and Hub transfer."""


for command, module in ETL.items():

    @main.command(
        command,
        context_settings={"ignore_unknown_options": True, "help_option_names": []},
        help=f"eval_datasets.etl.{module} (argparse; --help lists its subcommands)",
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    def _forward(args: tuple[str, ...], module: str = module) -> None:
        sys.exit(importlib.import_module(f"eval_datasets.etl.{module}").main(list(args)))


main.add_command(fetch)
main.add_command(publish)
main.add_command(publish_checkpoint)


if __name__ == "__main__":
    main()

__all__ = ["main"]
