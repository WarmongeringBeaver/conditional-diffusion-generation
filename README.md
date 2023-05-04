# README

## Install

First:

```bash
{micromamba,mamba,conda} install -f environment.yml
```

Don't forget to:

```bash
{micromamba,mamba,conda} activate diffusion-experiments
```

## Config & run

The full configuration is specified in a `TOML` file, passed in arg as:

```bash
python conditional_diffusion_generation.py CONFIG_FILE_PATH
```

See `example-config.toml` for an example.
