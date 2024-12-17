import time
import envlogger
from sequential_tfds_writer import SequentialTFDSWriter
import numpy as np
import tensorflow_datasets as tfds
from xarm_env import XArmEnvironment

def main():
    env = XArmEnvironment()

    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="xarm",
        observation_info=tfds.features.Tensor(
            shape=(480, 640, 3), dtype=np.uint8,
            encoding=tfds.features.Encoding.ZLIB),
        action_info=np.int64,
        reward_info=np.float64,
        discount_info=np.float64
    )

    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="xarm",
        observation_info=np.float64,
        action_info=np.int64,
        reward_info=np.float64,
        discount_info=np.float64
    )

    env = envlogger.EnvLogger(
        env,
        backend = SequentialTFDSWriter(
            data_directory="test_data", # ensure dir exists
            ds_config=dataset_config
        )
    )

    print("Starting environment")

    target_duration = 1.0 / 50.0
    samples = 300
    env.reset()
    for _ in range(samples):
        start = time.time()
        env.step(0)
        end = time.time()
        if end - start < target_duration:
            time.sleep(target_duration - (end - start))
        else:
            print(f"Took too long to step: {end - start}")
    env.close()

    dataset = tfds.builder_from_directory('test_data').as_dataset(split='all')
    for element in dataset:
        for k, v in element.items():
            for x in v:
                print(x)

if __name__ == "__main__":
    main()
