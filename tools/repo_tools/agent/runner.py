"""Abstract base class for agent runners."""

from abc import ABC, abstractmethod
import argparse


class AgentRunner(ABC):
    """Base class for coding agent launchers."""

    name: str = ""
    help: str = ""

    @abstractmethod
    def setup(self, parser: argparse.ArgumentParser) -> None:
        """Configure subcommand-specific arguments."""
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Launch the agent. Returns exit code."""
        pass
