from collections import deque

from PIL import Image
import numpy as np


def atari_preprocess(frame):
    frame = frame.mean(axis=2).astype(np.uint8)
    img = Image.fromarray(frame)
    img = img.resize((84, 110))
    img = img.crop((0, 16, 84, 100))
    return np.array(img)


class StackedFrames:
    def __init__(self, stack_size=4, preprocess_fn=atari_preprocess):
        self.stack = None
        self.stack_size = stack_size
        self.preprocess = preprocess_fn

    def on_new_episode(self, frame):
        frame = self.preprocess(frame)
        self.stack = deque(
            [frame for _ in range(self.stack_size)],
            maxlen=self.stack_size
        )

    def on_new_step(self, frame):
        self.stack.append(self.preprocess(frame))

    def get(self):
        return np.stack([s for s in self.stack])
