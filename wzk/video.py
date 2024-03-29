import os
import numpy as np

from wzk.files import dir_dir2file_array
from wzk.strings import uuid4
import fire


def stack_videos(videos=None, file=None):
    """
    https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg/33764934#33764934
    """
    if isinstance(videos, str):
        videos = dir_dir2file_array(videos)

    _format = "mp4"
    kwargs = ""

    s = np.shape(videos)

    if file is None:
        file = f"stacked_video__{s[0]}x{s[1]}__{uuid4()}.{_format}"

    uuid_list = [f"{uuid4()}.{_format}" for _ in range(s[0])]

    for i, in_i in enumerate(videos):
        in_i_str = " -i ".join(in_i)
        stack_str = "".join([f"[{j}:v]" for j in range(s[1])])

        os.system(f"ffmpeg -i {in_i_str} "
                  f"{kwargs} "
                  f'-filter_complex "{stack_str}"hstack=inputs={s[1]}[v] -map "[v]" '
                  f"{uuid_list[i]}")

    in_i_str = " -i ".join(uuid_list)
    stack_str = "".join([f"[{j}:v]" for j in range(s[0])])
    os.system(f"ffmpeg -i {in_i_str} "
              f"{kwargs} "
              f'-filter_complex "{stack_str}"vstack=inputs={s[0]}[v] -map "[v]" '
              f"{file}")

    for u in uuid_list:
        os.remove(u)


if __name__ == "__main__":
    fire.Fire(stack_videos)
