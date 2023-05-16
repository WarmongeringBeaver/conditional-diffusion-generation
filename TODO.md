# TODO's

## Global, bold ideas

- Learn a base model on DMSO and fine-tune a derived one on each condition, then use Schrodinger Bridges to go from one to the other

## Code

- `UNet2DModel` $\rightarrow$ `UNet2DConditionModel` in training script (actually encapsulates all use cases)
- Try gradient accumulation after thinking about why using it
- Allow choosing compile strategy in args (max-autotune, etc.) $\rightarrow$ simply pass an arg to `acccelerate`?
- Parallelize image generation
- Profile!
- Log the $\log$ of the loss to better see its evolution

## Conditioning

- Explore other conditioning schemes:
  - Try passing the embedding later on in the net
  - Try a simple OHE?
  - Try concatenating the class embedding instead of adding it to the timestep embedding
  - Try adding cross-attention layers attending *class names* directly?
Might seem weird, but it's actually what us humans are doing! Or attend to the class embedding, then project it to the timestep embedding dimension before adding it to the timestep embedding, or concatenate. Then cross-attention layers in the UNet to incorporate this information into the denoising path
  - Try guidance like in [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
(that's +/- how Stable Diffusion handles text conditioning, reportedly)
(also fun to implement this, but probably time-consuming). For example guidance by L2 norm between leanred embeddings of classes? Need a classifier and an associated differentiable loss anyway.
- Pay attention to value range: images are in [-1; 1], where's the embedding?
- Control the "Gausssianity" of the obtained noise at the end of the diffusion process $\rightarrow$ if not Gaussian enough, sampling $x_T$ should be done *conditionally*

## Misc

- Adapt loss to unbalanced classes?
- Explore using other metrics (w/ pytorch-fidelity)
  - Fix IS being constant
  - Perceptual Distance
  - Classification accuracy if we have a pretrained classifier?
  - ..?
- Try velocity model
