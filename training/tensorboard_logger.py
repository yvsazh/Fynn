from __future__ import annotations

import os
import socket
import time
import uuid
from pathlib import Path

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.histogram_pb2 import HistogramProto
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_STRING
from tensorboard.summary.writer.record_writer import RecordWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path, flush_secs: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_secs = flush_secs
        self._last_flush_time = time.time()
        event_path = self.log_dir / self._event_filename()
        self._file = event_path.open("wb")
        self._writer = RecordWriter(self._file)
        self._write_raw_event(Event(wall_time=time.time(), file_version="brain.Event:2"))

    def _event_filename(self) -> str:
        return f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"

    def _write_raw_event(self, event: Event) -> None:
        self._writer.write(event.SerializeToString())
        self._maybe_flush()

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self._write_raw_event(Event(wall_time=time.time(), step=int(step), summary=summary))

    def add_text(self, tag: str, text: str, step: int) -> None:
        tensor = TensorProto(
            dtype=DT_STRING,
            string_val=[text.encode("utf-8")],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
        )
        metadata = SummaryMetadata(
            plugin_data=SummaryMetadata.PluginData(plugin_name="text"),
        )
        summary = Summary(value=[Summary.Value(tag=tag, metadata=metadata, tensor=tensor)])
        self._write_raw_event(Event(wall_time=time.time(), step=int(step), summary=summary))

    def add_histogram(self, tag: str, values: np.ndarray, step: int, bins: int = 64) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        counts, limits = np.histogram(array, bins=bins)
        histogram = HistogramProto(
            min=float(array.min()),
            max=float(array.max()),
            num=int(array.size),
            sum=float(array.sum()),
            sum_squares=float(np.square(array).sum()),
        )
        histogram.bucket_limit.extend(float(edge) for edge in limits[1:])
        histogram.bucket.extend(float(count) for count in counts)
        summary = Summary(value=[Summary.Value(tag=tag, histo=histogram)])
        self._write_raw_event(Event(wall_time=time.time(), step=int(step), summary=summary))

    def _maybe_flush(self) -> None:
        now = time.time()
        if now - self._last_flush_time >= self.flush_secs:
            self._writer.flush()
            self._last_flush_time = now

    def flush(self) -> None:
        self._writer.flush()
        self._last_flush_time = time.time()

    def close(self) -> None:
        self._writer.close()
