<h1 style="text-align: center;">Structured 4D Latent Predictive Model for Robot Planning</h1>

<p align="center">
  <a href="https://sky-lzy.github.io/">Zhiyi Li</a>,
  <a href="https://peilinwu.site/">Peilin Wu</a>,
  <a href="https://xshan.site/">Xiaoshen Han</a>,
  <a href="https://ruojincai.github.io/">Ruojin Cai</a>,
  <a href="https://yilundu.github.io/">Yilun Du</a>
</p>

<p align="center">
  <a href="">
    <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv'>
  </a>
  <a href='https://structured-4d-model.github.io/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </a>
  <a href='https://huggingface.co/zhiyi24/structured-4d-model' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
  </a>
</p>

<p align="center">
  ICML 2026
</p>


<p align="center">
  <img src="assets/figures/teaser.svg" alt="Structured 4D Latent Predictive Model teaser" width="95%">
</p>

We propose a **Structured 4D Latent Predictive Model** that predicts text-conditioned 3D scene dynamics from multi-view observations. The model forecasts future scene structure in a latent 3D representation, decodes the predictions into 3D-consistent geometry, and uses the generated futures for goal-conditioned robot planning.

## Setup

Create the release environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate structured_4d_model
```

## Inference

### Checkpoints

Download the released model weights from Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhiyi24/structured-4d-model",
    local_dir=".ckpt/structured-4d-model",
    local_dir_use_symlinks=False,
)
```

The Hugging Face model repo contains:

- `generator/single_dynamics.json`
- `generator/single_dynamics.safetensors`
- `generator/latent_generator.json`
- `generator/latent_generator.safetensors`
- `inverse_dynamics/inverse_dynamics.ckpt`
- `pipeline.json`

The upstream TRELLIS decoder/encoder weights and CLIP text model are downloaded from their original Hugging Face repositories on first use.


### 3D Future Generation

The example below unrolls one LIBERO initial state with a text instruction.

```bash
python sample_unroll.py \
  --data_path assets/latents/libero_example.npz \
  --instruction "open the top drawer of the cabinet" \
  --num_steps 3
```

This writes generated Gaussian splats to `render/generate` by default.


To generate all supported LIBERO examples:

```bash
instructions=(
  "open the top drawer of the cabinet"
  "put the black bowl at the back on the plate"
  "put the black bowl at the front on the plate"
  "put the middle black bowl on the plate"
  "put the middle black bowl on top of the cabinet"
  "stack the black bowl at the front on the black bowl in the middle"
  "stack the middle black bowl on the back black bowl"
)

for instruction in "${instructions[@]}"; do
  name=$(echo "$instruction" | tr ' ' '_' | tr -cd '[:alnum:]_')
  python sample_unroll.py \
    --data_path assets/latents/libero_example.npz \
    --instruction "$instruction" \
    --output_dir "render/generate/${name}" \
    --num_steps 3
done
```

### Policy Evaluation

We provide policy evaluation on ManiSkill `StackCube-v1`.

```bash
python eval.py --num-seeds 100 --seed-start 10000
```

This loads `.ckpt/structured-4d-model/inverse_dynamics/inverse_dynamics.ckpt` and writes policy videos plus `success_list.txt` to `render/policy` by default.

## Training

Training starts from ManiSkill demonstration trajectories, converts them into multi-view scene observations, extracts TRELLIS-compatible features, encodes latent scene states, and then trains the two generator configs.

### ManiSkill Demonstrations

Download the [ManiSkill demo trajectories](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html) with the ManiSkill downloader. `StackCube-v1` is the default example used by `data_tools/configs/stackcube.json`.

```bash
python -m mani_skill.utils.download_demo StackCube-v1 -o ./data/raw/maniskill
```

### Data Preparation

The full StackCube preprocessing pipeline is:

```bash
python data_tools/simulation_maniskill.py -c data_tools/configs/stackcube.json
python data_tools/extract_feature.py --output_dir data/maniskill/StackCube-v1 --batch_size 16
python data_tools/encode_latent.py --output_dir data/maniskill/StackCube-v1
python data_tools/encode_ss_latent.py --output_dir data/maniskill/StackCube-v1
```

This writes frame folders under `data/maniskill/StackCube-v1`, with `voxels.ply`, `latents.npz`, and `ss_latents.npz` in each frame directory.


### Generator Training

Train the single-dynamics sparse-structure generator with `configs/gen_ss.json`:

```bash
python train.py \
  --config configs/gen_ss.json \
  --data_dir data/maniskill/StackCube-v1 \
  --output_dir outputs/gen_ss_stackcube
```

Train the structured-latent generator with `configs/gen_slat.json`:

```bash
python train.py \
  --config configs/gen_slat.json \
  --data_dir data/maniskill/StackCube-v1 \
  --output_dir outputs/gen_slat_stackcube
```

## Checklist
- [x] Release the inference code
- [x] Release the data generation pipeline
- [x] Release the training code


## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{li2026structured4d,
  title     = {Structured 4D Latent Predictive Model for Robot Planning},
  author    = {Li, Zhiyi and Wu, Peilin and Han, Xiaoshen and Cai, Ruojin and Du, Yilun},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## Acknowledgements

We would like to thank the following repositories for their code, data, and models that we build upon in this work: [TRELLIS](https://github.com/microsoft/TRELLIS), [ManiSkill](https://github.com/mani-skill/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [RLBench](https://github.com/stepjam/RLBench), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy). 
