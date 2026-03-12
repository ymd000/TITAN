from pathlib import Path

import h5py
import torch
from transformers import AutoModel

EMBEDDING_DIR = Path("../SFT_Meni/tmp/embedding")
FEATURE_KEY = "conch15_768/features"
COORD_KEY = "conch15_768/coordinates"
OUTPUT_KEY = "conch15_768/slide_embedding/titan"
PATCH_SIZE_LV0 = 256 * 2  # 256px @ level1, downsample=2.0


def main():
    print("Loading TITAN model...")
    titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    titan.eval()

    h5_paths = sorted(EMBEDDING_DIR.glob("*.h5"))
    print(f"Found {len(h5_paths)} files")

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if FEATURE_KEY not in f or COORD_KEY not in f:
                print(f"[SKIP] {h5_path.name}: keys not found")
                continue
            features = torch.from_numpy(f[FEATURE_KEY][:])
            coords = torch.from_numpy(f[COORD_KEY][:])

        with torch.autocast("cuda", torch.float16), torch.inference_mode():
            slide_embedding = titan.encode_slide_from_patch_features(
                features, coords, PATCH_SIZE_LV0
            )

        with h5py.File(h5_path, "a") as f:
            if OUTPUT_KEY in f:
                del f[OUTPUT_KEY]
            f.create_dataset(OUTPUT_KEY, data=slide_embedding.cpu().float().numpy())

        print(f"[DONE] {h5_path.name}: {slide_embedding.shape}")


if __name__ == "__main__":
    main()
