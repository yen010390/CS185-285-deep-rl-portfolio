"""
Script tai truoc dataset OGBench, dung khi ogbench.make_env_and_datasets()
bi loi "HTTP Error 403: Forbidden" khi tu dong tai dataset.

Cach dung:
    uv run python download_ogbench_dataset.py --env_name=cube-single-play-singletask-task1-v0

(chay lenh nay trong thu muc hw5, cung noi ban chay src/scripts/run.py)

Sau khi chay xong, chay lai lenh train binh thuong:
    uv run src/scripts/run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
"""

import argparse
import os
import urllib.request

DEFAULT_DATASET_DIR = "~/.ogbench/data"
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"

# Header gia lam trinh duyet that de tranh bi chan 403.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def env_name_to_dataset_name(env_name: str) -> str:
    """
    Chuyen ten env (vd: cube-single-play-singletask-task1-v0)
    thanh ten dataset (vd: cube-single-play-v0), theo dung logic
    trong ogbench/utils.py (make_env_and_datasets).
    """
    splits = env_name.split("-")
    if "singletask" in splits:
        pos = splits.index("singletask")
        # Bo tu "singletask" va "task\\d" (neu co), giu tu "-v0" cuoi cung.
        return "-".join(splits[:pos] + splits[-1:])
    elif "oraclerep" in splits:
        return "-".join(splits[:-2] + splits[-1:])
    else:
        return "-".join(splits[:-2] + splits[-1:])


def download_one(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"[OK] Da co san: {dest_path}")
        return
    print(f"Dang tai: {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    tmp_path = dest_path + ".tmp"
    with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as f:
        total = getattr(response, "length", None)
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded/1e6:.1f} MB / {total/1e6:.1f} MB ({pct:.1f}%)", end="")
        print()
    os.rename(tmp_path, dest_path)
    print(f"[DONE] Da luu: {dest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True,
                         help="Ten env, vd: cube-single-play-singletask-task1-v0")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR,
                         help="Thu muc luu dataset (mac dinh giong ogbench: ~/.ogbench/data)")
    args = parser.parse_args()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    dataset_name = env_name_to_dataset_name(args.env_name)
    print(f"env_name = {args.env_name}")
    print(f"-> dataset_name = {dataset_name}")
    print(f"-> luu vao thu muc: {dataset_dir}")
    print()

    for suffix in ["", "-val"]:
        file_name = f"{dataset_name}{suffix}.npz"
        url = f"{DATASET_URL}/{file_name}"
        dest_path = os.path.join(dataset_dir, file_name)
        download_one(url, dest_path)

    print()
    print("Xong! Gio ban co the chay lai lenh train binh thuong,")
    print("ogbench se thay file da co san va bo qua buoc tai.")


if __name__ == "__main__":
    main()
