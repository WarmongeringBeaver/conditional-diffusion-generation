# TODO's

- Visualize an interpolation
- `UNet2DModel` ⇾ `UNet2DConditionModel` in training script (actually encapsulates all use cases)
- Explore other conditioning schemes
  - pay attention to value range: images are in [-1; 1], where's the embedding?
  - try passing the embedding later on in the net
  - try a simple OHE
  - try adding cross-attention layers attending *class names* directly?
Might seem weird but it's actually what us humans are doing!
Then cross-attention layers in the UNet to incorporate this information into the denoising path
  - try guidance like in (https://arxiv.org/abs/2105.05233)
(that's +/- how Stable Diffusion handles text conditioning, reportedly)
(also fun to implement this, but probably time-consuming)
- Fix IS being constant
- Adapt loss to unbalanced classes?
- profile!
- explore using other metrics (w/ pytorch-fidelity)
  - Perceptual Distance
  - ..?
  - Classification accuracy if we have a pretrained classifier?
- try velocity model
- try gradient accumulation
- allow to choose compile strategy in args (max-autotune, etc.)
- parallelize image generation
