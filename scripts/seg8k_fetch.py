# scripts/seg8k_fetch.py
"""
One-shot CholecSeg8k fetcher:
- Downloads via kagglehub to user cache
- Then links/copies/moves into repo data directory

Usage examples:
  python scripts/seg8k_fetch.py                        # auto strategy -> data/seg8k
  python scripts/seg8k_fetch.py --strategy move        # copy+verify then delete cache
  python scripts/seg8k_fetch.py --strategy link        # prefer link (junction/symlink)
  python scripts/seg8k_fetch.py --dst data/seg8k_v11   # custom destination
  python scripts/seg8k_fetch.py --dataset newslab/cholecseg8k

Train afterwards:
  python -m src.training.train_offline_min --data_root data/seg8k --split train \
      --img_size 512 --batch_size 6 --epochs 1 --lr 1e-3 --num_classes 2 \
      --save_path checkpoints/baseline_offline_min.pth
"""
import os, json, shutil, argparse, subprocess, sys, platform

def human(n):
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}EB"

def dir_stats(root):
    total_bytes, total_files = 0, 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total_bytes += os.path.getsize(p)
                total_files += 1
            except FileNotFoundError:
                pass
    return total_bytes, total_files

def try_symlink(src, dst):
    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"[OK] Symlink created: {dst} -> {src}")
        return True
    except Exception as e:
        print(f"[INFO] symlink failed: {e}")
        return False

def try_junction(src, dst):
    if platform.system().lower() != "windows":
        return False
    try:
        # Create parent dirs
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.check_call(['cmd', '/c', 'mklink', '/J', dst, src])
        print(f"[OK] Junction created: {dst} -> {src}")
        return True
    except Exception as e:
        print(f"[INFO] junction failed: {e}")
        return False

def safe_copytree(src, dst):
    if os.path.exists(dst):
        print(f"[INFO] Destination exists, skip copy: {dst}")
        return True
    print(f"[INFO] Copying to {dst} (this may take a while)...")
    shutil.copytree(src, dst)
    b_src, f_src = dir_stats(src)
    b_dst, f_dst = dir_stats(dst)
    print(f"[INFO] Source : {f_src} files, {human(b_src)}")
    print(f"[INFO] Copied : {f_dst} files, {human(b_dst)}")
    if (b_src == b_dst) and (f_src == f_dst):
        print("[OK] Verification passed.")
        return True
    print("[ERR] Verification FAILED (counts/sizes mismatch).")
    return False

def remove_dir(path):
    try:
        shutil.rmtree(path)
        print(f"[OK] Removed cache: {path}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to remove cache: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="newslab/cholecseg8k",
                    help="Kaggle dataset slug for kagglehub")
    ap.add_argument("--dst", default="data/seg8k", help="Destination inside repo")
    ap.add_argument("--strategy", choices=["auto","link","copy","move"], default="auto",
                    help="How to integrate into repo data directory")
    args = ap.parse_args()

    # 1) Download to cache
    try:
        import kagglehub
    except Exception:
        print("[ERR] Missing dependency: kagglehub. Install via 'pip install kagglehub'", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Downloading: {args.dataset}")
    cache_path = kagglehub.dataset_download(args.dataset)
    print(f"[OK] Cache path: {cache_path}")

    # Record cache path
    os.makedirs("data", exist_ok=True)
    with open("data/seg8k_cache_path.json", "w", encoding="utf-8") as f:
        json.dump({"cache_path": cache_path}, f, ensure_ascii=False, indent=2)

    dst = os.path.abspath(args.dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f"[INFO] Destination already exists: {dst}")

    # 2) Integrate according to strategy
    def do_link():
        if platform.system().lower() == "windows":
            return try_junction(cache_path, dst) or try_symlink(cache_path, dst)
        return try_symlink(cache_path, dst)

    if args.strategy == "link":
        ok = do_link()
        if not ok:
            print("[ERR] Link creation failed. Try '--strategy copy' or 'move'.", file=sys.stderr)
            sys.exit(2)

    elif args.strategy == "copy":
        ok = safe_copytree(cache_path, dst)
        if not ok:
            sys.exit(3)

    elif args.strategy == "move":
        ok = safe_copytree(cache_path, dst)
        if not ok:
            sys.exit(3)
        # remove cache only after verified OK
        remove_dir(cache_path)

    else:  # auto
        ok = do_link()
        if not ok:
            print("[INFO] Falling back to COPY...")
            ok = safe_copytree(cache_path, dst)
            if not ok:
                sys.exit(3)

    # 3) Final hints
    print("\n[NEXT] Train your baseline with:")
    print(f"  python -m src.training.train_offline_min --data_root {args.dst} "
          "--split train --img_size 512 --batch_size 6 --epochs 1 --lr 1e-3 "
          "--num_classes 2 --save_path checkpoints/baseline_offline_min.pth")
    print("\n[NOTE] Ensure '.gitignore' ignores 'data/', 'checkpoints/', 'logs/', 'outputs/'. "
          "Keep 'data/README.md' or '.gitkeep' to track the folder.")
if __name__ == "__main__":
    main()
