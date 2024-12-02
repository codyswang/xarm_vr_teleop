from pathlib import Path
import pickle
from datetime import datetime
import tensorflow as tf

class Logger:

    def __init__(self):
        # Dataset fields
        self.episode_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.step_count = 0

        # Logger parameters
        self.write_path = Path(f"rlds_data/{self.episode_id}")
        self.write_path.mkdir(parents=True, exist_ok=False)

    def create_image_example(image, label):

        feature = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


    def log_step(self, timestamp, color_img):
        """Save step to episode record."""
        step = {
            "timestamp": timestamp,
            "is_first": self.step_count == 0,
            "is_last": False,
            "color_img": color_img,
        }
        with open(self.temp_path / f"step_{self.step_count}.pkl", "wb") as f:
            pickle.dump(step, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.step_count += 1

def main():
    logger = Logger()

if __name__ == "__main__":
    main()
