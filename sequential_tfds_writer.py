import dataclasses
from typing import Any, Dict, List, Optional
from envlogger import step_data
from envlogger.backends import rlds_utils
from envlogger.backends.tfds_backend_writer import TFDSBackendWriter
import tensorflow_datasets as tfds
from datetime import datetime

@dataclasses.dataclass
class Episode(object):
  """Episode that is being constructed."""
  prev_step: step_data.StepData
  metadata: Optional[Dict[str, Any]] = None

  def add_step(self, step: step_data.StepData) -> None:
    rlds_step = rlds_utils.to_rlds_step(self.prev_step, step)
    self.prev_step = step
    return rlds_step

  def get_rlds_episode(self) -> Dict[str, Any]:
    last_step = rlds_utils.to_rlds_step(self.prev_step, None)
    if self.steps is None:
      self.steps = []
    if self.metadata is None:
      self.metadata = {}

    return {'steps': self.steps + [last_step], **self.metadata}

class SequentialTFDSWriter(TFDSBackendWriter):

    def __init__(self,
                data_directory: str,
                ds_config: tfds.rlds.rlds_base.DatasetConfig,
                max_episodes_per_file: int = 1000,
                split_name: Optional[str] = None,
                version: str = '0.0.1',
                store_ds_metadata: bool = False,
                **base_kwargs):
        super()._init_(data_directory, ds_config, max_episodes_per_file, split_name, version, store_ds_metadata, **base_kwargs)

        self._current_episode_name = None
        self._prev_step = None

        self._step_writer = tfds.core.SequentialWriter(
            self._ds_info, 1000, overwrite=False
        )

    def _write_and_reset_episode(self):
        if self._current_episode_name is not None:
            # self._step_writer.add_examples(
            #     {self._split_name: [self._current_episode.get_rlds_episode()]})
            pass
        self._current_episode_name = None

    def _record_step(self, data: step_data.StepData,
                     is_new_episode: bool) -> None:
        """Stores RLDS steps in TFDS format."""

        if is_new_episode:
            self._write_and_reset_episode()

        if self._current_episode_name is None:
            self._current_episode_name = f"episode_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self._step_writer.initialize_splits([self._current_episode_name])
        else:
            self._step_writer.add_examples(
                {self._current_episode_name: [rlds_utils.to_rlds_step(self._prev_step, data)]}
            )

        self._prev_step = data

    def close(self) -> None:
        self._write_and_reset_episode()
        super()._sequential_writer.close_all()
        self._step_writer.close_all()
