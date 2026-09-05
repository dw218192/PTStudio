"""Check source formatting."""

import click

from tasks import fmt
from tasks.utils.project import load_context, project_options


@click.command(help="Check source formatting without modifying files")
@project_options
def main(platform: str, build_type: str) -> None:
    fmt.run(load_context(platform, build_type), {"verify": True})


if __name__ == "__main__":
    main()
