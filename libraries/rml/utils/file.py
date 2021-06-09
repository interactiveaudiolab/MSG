import os
import shutil
import tarfile
import zipfile

import urllib.request as req

from typing import Callable


# add tqdm
def download(src: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)
    
    file = src.split("/")[-1]    
    target = os.path.join(root, file)
    
    req.urlretrieve(src, target)
    
    return target


def download_extract_try(src: str, root: str) -> None:
    ext = src.split(".")[-1]
    
    if ext == "zip":
        download_extract(src, root, use_zip)
    elif ext == "gz" or ext == "tgz":
        download_extract(src, root, use_tar)
    else:
        download(src, root)


def download_extract(src: str, root: str, method: Callable[[str, str], None]) -> None:
    target = download(src, root)
    dr = os.path.join(root, src.split("/")[-1].split(".")[0])

    method(root, target)
    
    if os.path.exists(dr):
        for file in os.listdir(dr):
            shutil.move(os.path.join(dr, file), root)

        os.rmdir(dr)
    
    os.remove(target)


def use_tar(root: str, archive: str) -> None:
    tar_file = tarfile.open(archive, "r:gz")
    tar_file.extractall(path=root)
    tar_file.close()


def use_zip(root: str, archive: str) -> None:
    zip_file = zipfile.ZipFile(archive, 'r')
    zip_file.extractall(root)
    zip_file.close()
