import h5py
import torch
from transformers import AutoModel

titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

with h5py.File('../SFT_Meni/tmp/embedding/18-1825.h5', 'r') as f:
    features = torch.from_numpy(f['conch15_768/features'][:])
    coords = torch.from_numpy(f['conch15_768/coordinates'][:])

# 256px @ level1 (downsample=2.0) → level0換算
patch_size_lv0 = 256 * 2  # = 512

with torch.autocast('cuda', torch.float16), torch.inference_mode():
    slide_embedding = titan.encode_slide_from_patch_features(
        features, coords, patch_size_lv0
    )

print(slide_embedding.shape)
