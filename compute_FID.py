import os

import torch_fidelity

generated_images = "path/to/generated_images"
dataset = "path/to/dataset"
print("generated_images:", generated_images)
print("dataset:", dataset)

assert os.path.exists(generated_images)
assert os.path.exists(dataset)

metrics_dict = torch_fidelity.calculate_metrics(
    input1=generated_images,
    input2=dataset,
    cuda=True,
    batch_size=64,
    fid=True,
    verbose=True,
    cache_root=".fidelity_cache",
    rng_seed=42,
)

print("metrics_dict:", metrics_dict)
