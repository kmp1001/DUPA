IMG_EXTENSIONS = (
    "*.png",
    "*.JPEG",
    "*.jpeg",
    "*.jpg"
)

PATH = "/mnt/bn/wangshuai6/neural_sampling_workdirs/expbaseline_adam2_timeshift1.5"

import os
import pathlib
PATH = pathlib.Path(PATH)
images = []

# find images
for ext in IMG_EXTENSIONS:
    images.extend(PATH.rglob(ext))

for image in images:
    os.system(f"rm -f {image}")

