from typing import List

import exiftool


class DeepFake:
    def __init__(self):
        self.a = 1

    def check_software(self, video_paths: List[str]):
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata_batch(video_paths)

        for d in metadata:
            print(d)

        return False
