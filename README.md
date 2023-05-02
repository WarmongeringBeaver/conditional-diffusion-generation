# README

## Install

First:

```bash
{micromamba,mamba,conda} install -f environment.yml
```

Then:

```bash
pip install pytorch-fid torch-fidelity
```

Don't forget to:

```bash
{micromamba,mamba,conda} activate diffusion-experiments
```

## Config & run

Specified in a `TOML` file, passed in arg as:

```bash
python conditional_diffusion_generation.py CONFIG_FILE_PATH
