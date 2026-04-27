from __future__ import annotations

import json
import random
import threading
import concurrent.futures
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar

import numpy as np
import torch
import yaml


_InputT = TypeVar("_InputT")
_ResultT = TypeVar("_ResultT")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or base_dir is None:
        return candidate.expanduser().resolve(strict=False)
    return (Path(base_dir) / candidate).expanduser().resolve(strict=False)


_WORKER_TASK_TIMEOUT_SECONDS = 120


def _wait_guaranteed(futures_seq: tuple, timeout: float):
    """Like concurrent.futures.wait(FIRST_COMPLETED) but guaranteed to return within timeout.

    When a worker crashes at the C level (e.g. malloc SIGABRT), the underlying pipe
    never sends EOF and wait() hangs forever despite its timeout argument.  Running it
    in a daemon thread lets us enforce a hard deadline and treat all unfinished futures
    as still-pending so the existing timeout-cancellation logic kicks in.
    """
    container: list = []

    def _run() -> None:
        container.append(wait(futures_seq, timeout=None, return_when=FIRST_COMPLETED))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout)
    if container:
        return container[0]
    return frozenset(), frozenset(futures_seq)


def iter_bounded_executor_map(
    executor: Executor,
    func: Callable[[_InputT], _ResultT],
    items: Iterable[_InputT],
    *,
    max_in_flight: int,
) -> Iterator[_ResultT]:
    items_iter = iter(items)
    in_flight_limit = max(1, int(max_in_flight))
    pending_by_index: dict[int, Any] = {}
    future_to_index: dict[Any, int] = {}
    completed: dict[int, _ResultT] = {}
    next_submit_index = 0
    next_yield_index = 0
    exhausted = False

    def _submit_until_full() -> None:
        nonlocal exhausted
        nonlocal next_submit_index
        while not exhausted and len(pending_by_index) < in_flight_limit:
            try:
                item = next(items_iter)
            except StopIteration:
                exhausted = True
                return
            future = executor.submit(func, item)
            pending_by_index[next_submit_index] = future
            future_to_index[future] = next_submit_index
            next_submit_index += 1

    _submit_until_full()

    while pending_by_index:
        done, still_pending = _wait_guaranteed(
            tuple(pending_by_index.values()),
            timeout=_WORKER_TASK_TIMEOUT_SECONDS,
        )
        if not done:
            # Timed out — cancel stuck futures and treat them as skipped
            for future in tuple(still_pending):
                future.cancel()
                index = future_to_index.pop(future, None)
                if index is not None:
                    pending_by_index.pop(index, None)
                    completed[index] = {"samples": [], "skipped": 1, "error": "worker timed out"}  # type: ignore[assignment]
        else:
            for future in done:
                index = future_to_index.pop(future)
                pending_by_index.pop(index, None)
                try:
                    completed[index] = future.result()
                except concurrent.futures.CancelledError:
                    completed[index] = {"samples": [], "skipped": 1, "error": "worker cancelled"}  # type: ignore[assignment]
                except Exception as exc:
                    completed[index] = {"samples": [], "skipped": 1, "error": str(exc)}  # type: ignore[assignment]

        while next_yield_index in completed:
            yield completed.pop(next_yield_index)
            next_yield_index += 1

        _submit_until_full()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(path: str | Path, data: Any) -> None:
    if is_dataclass(data):
        data = asdict(data)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]
