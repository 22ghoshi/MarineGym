"""
fix_asset_paths.py - repoint MarineGym's USD assets at files that exist on THIS machine.

The shipped USDs contain absolute paths from the authors' computers, e.g.
    /home/david/codeSpace/MarineGym/marinegym/robots/assets/usd/...   (textures)
    /mnt/Documents/isaac_underwater/models/Unreal_Engine_4_Sky.usdz    (sky / lighting)
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/...  (MDL materials)
Isaac can't find them, so materials fall back to plain white.

This script walks every .usd/.usda/.usdc under marinegym/robots/assets/usd, and for
each asset path that does not resolve, tries to find the same file inside the repo
(including the local copies in worlds/Collected_EmptyMarine). Matches are rewritten
as paths relative to the layer, so the repo stays portable. Originals are kept as
<file>.bak.

    cd /mnt/crucial/MarineGym
    python scripts/fix_asset_paths.py            # dry run: show what would change
    python scripts/fix_asset_paths.py --apply    # rewrite files (backups made)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from pxr import Sdf, UsdUtils

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "marinegym" / "robots" / "assets" / "usd"
COLLECTED = ASSETS / "worlds" / "Collected_EmptyMarine"

# Known prefixes from the authors' machines -> where the same tree lives locally.
PREFIX_MAP = [
    ("/home/david/codeSpace/MarineGym/", str(REPO) + "/"),
    ("/mnt/Documents/", str(COLLECTED / "Documents") + "/"),
    ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/",
     str(COLLECTED / "omniverse-content-production.s3-us-west-2.amazonaws.com") + "/"),
    ("http://omniverse-content-production.s3-us-west-2.amazonaws.com/",
     str(COLLECTED / "omniverse-content-production.s3-us-west-2.amazonaws.com") + "/"),
    ("https://omniverse-content-production.s3.us-west-2.amazonaws.com/",
     str(COLLECTED / "omniverse-content-production.s3.us-west-2.amazonaws.com") + "/"),
    ("http://omniverse-content-production.s3.us-west-2.amazonaws.com/",
     str(COLLECTED / "omniverse-content-production.s3.us-west-2.amazonaws.com") + "/"),
]

# Index every file in the asset tree by basename, as a last-resort lookup.
BY_NAME = {}
for p in ASSETS.rglob("*"):
    if p.is_file():
        BY_NAME.setdefault(p.name, []).append(p)


def resolves(layer_dir: Path, asset: str) -> bool:
    if not asset or asset.startswith(("omniverse://", "anon:")):
        return True  # leave Nucleus / anonymous paths alone
    if asset.startswith(("http://", "https://")):
        return False
    p = Path(asset)
    if not p.is_absolute():
        p = layer_dir / p
    return p.exists()


def find_local(asset: str):
    for old, new in PREFIX_MAP:
        if asset.startswith(old):
            cand = Path(new + asset[len(old):])
            if cand.exists():
                return cand
    hits = BY_NAME.get(Path(asset.split("?")[0]).name, [])
    return hits[0] if len(hits) == 1 else None  # only if unambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    files = [p for ext in ("*.usd", "*.usda", "*.usdc") for p in ASSETS.rglob(ext)]
    total_fixed, unresolved = 0, set()

    for f in sorted(files):
        layer = Sdf.Layer.FindOrOpen(str(f))
        if layer is None:
            print(f"[skip] could not open {f}")
            continue
        layer_dir = f.parent
        changes = []

        def remap(asset: str) -> str:
            if resolves(layer_dir, asset):
                return asset
            local = find_local(asset)
            if local is None:
                unresolved.add(asset)
                return asset
            new = os.path.relpath(local, layer_dir)
            if not new.startswith("."):
                new = "./" + new
            changes.append((asset, new))
            return new

        UsdUtils.ModifyAssetPaths(layer, remap)

        if changes:
            rel = f.relative_to(REPO)
            print(f"\n{rel}: {len(changes)} path(s)")
            for old, new in sorted(set(changes)):
                print(f"   {old}\n -> {new}")
            total_fixed += len(changes)
            if args.apply:
                shutil.copy2(f, str(f) + ".bak")
                layer.Save()
        else:
            layer.Reload(force=True)  # discard nothing-changes

    print(f"\n{'Rewrote' if args.apply else 'Would rewrite'} {total_fixed} asset path(s).")
    if unresolved:
        print("\nStill missing (no local copy found):")
        for a in sorted(unresolved):
            print("   ", a)
    if not args.apply and total_fixed:
        print("\nRe-run with --apply to write the changes.")


if __name__ == "__main__":
    sys.exit(main())