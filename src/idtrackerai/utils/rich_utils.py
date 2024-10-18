import logging
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import IO, TypeVar

from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    _Reader,
)

InputType = TypeVar("InputType")


def track(
    sequence: Iterable[InputType],
    desc: str = "Working...",
    total: float | None = None,
    verbose: bool = True,
) -> Iterable[InputType]:
    """A custom interpretation of rich.progress.track"""

    if not verbose:
        yield from sequence
        return

    progress = Progress(
        "         " + desc,
        BarColumn(bar_width=None),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True),
        transient=True,
    )

    with progress:
        yield from progress.track(sequence, total, description=desc)

    task = progress.tasks[0]

    logging.info(
        "[green]%s[/] (%s iterations). It took %s",
        desc,
        int(task.total) if task.total is not None else "unknown",
        "--:--" if task.elapsed is None else timedelta(seconds=int(task.elapsed)),
        stacklevel=2,
        extra={"markup": True},
    )


class _ReaderWriter(_Reader):
    """An extension of rich.progress._Reader to allow file writing"""

    def writable(self) -> bool:
        return True

    def write(self, s) -> int:
        block = self.handle.write(s)
        self.progress.advance(self.task, advance=block)
        return block


class open_track:
    """
    Context manager to track progress in large object disk read/write processes.
    So far, only used in :class:`ListOfBlobs` and :class:`ListOfFragments`.
    """

    def __init__(
        self,
        file: str | Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        verbose: bool = True,
    ) -> None:

        reading = mode in ("r", "rb", "rt")

        self.desc = (
            ("Reading" if reading else "Writing") + " [italic]" + Path(file).name
        )

        if not verbose:
            self.reader = open(file, mode, buffering, encoding)  # noqa: SIM115
            self.progress = None
            return

        self.progress = Progress(
            "         " + self.desc,
            BarColumn(bar_width=None),
            DownloadColumn() if reading else FileSizeColumn(),
            (
                TimeRemainingColumn(elapsed_when_finished=True)
                if reading
                else TransferSpeedColumn()
            ),
            transient=True,
        )

        if reading:
            self.reader = self.progress.open(
                file, mode, buffering, encoding, description=self.desc
            )
        else:
            self.reader = _ReaderWriter(
                # We are reusing rich.progress._Reader, with works with BinaryIO, with an any IO, lets hope it works on all OS
                open(file, mode, buffering, encoding),  # type: ignore # noqa: SIM115
                self.progress,
                self.progress.add_task(self.desc, total=None),
            )

    def __enter__(self) -> IO:
        if self.progress is not None:
            self.progress.start()
        return self.reader.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.progress is not None:
            self.progress.stop()
            task = self.progress.tasks[0]

            logging.info(
                "[green]%s[/] (%s). It took %s",
                self.desc,
                FileSizeColumn().render(task),
                (
                    "--:--"
                    if task.elapsed is None
                    else timedelta(seconds=int(task.elapsed))
                ),
                stacklevel=2,
                extra={"markup": True},
            )
        self.reader.__exit__(exc_type, exc_value, traceback)
