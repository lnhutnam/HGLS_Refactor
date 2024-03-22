import os
import logging, logging.config
from pathlib import Path


def set_logger(log_file):
    """Write logs to checkpoint and console

    Args:
        log_file (str): filename string/ log filename.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """Function for increment path
    
    Adapted from Yolov7, 
    Ref: https://github.com/WongKinYiu/yolov7

    Args:
        path (pathlib.Path): Path
        exist_ok (bool, optional): existing project/name ok, do not increment.. Defaults to False.
        sep (str, optional): seperating string. Defaults to "".
        mkdir (bool, optional): need make directory?. Defaults to False.

    Returns:
        pathlib.Path: path to file.
    """
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
