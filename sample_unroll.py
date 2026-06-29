import os
os.environ['SPCONV_ALGO'] = 'auto'        # Can be 'native' or 'auto', default is 'auto'.

import argparse
import json
import numpy as np
from trellis_utils import pipelines as trellis_pipelines
from trellis_utils.pipelines import TrellisSDM3DGenPipeline


DEFAULT_MODEL_PATH = ".ckpt/structured-4d-model"
DEFAULT_OUTPUT_DIR = "render/generate"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.json")
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local directory or Hugging Face repo id containing pipeline.json and model weights.",
    )
    parser.add_argument("--data_path", type=str, default="assets/latents/libero_example.npz")
    parser.add_argument("--instruction", type=str, required=True, help="The instruction to guide the generation.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ss_steps", type=int, default=25)
    parser.add_argument("--slat_steps", type=int, default=25)
    parser.add_argument("--render_ply", action="store_true", default=True)
    return parser.parse_args(argv)


def load_pipeline(config_path, model_path=None):
    if model_path:
        return trellis_pipelines.from_pretrained(model_path)
    with open(config_path, 'r') as f:
        cfgs = json.load(f)
    return TrellisSDM3DGenPipeline.from_config(cfgs)


def convert_ply(xyz, f_dc, output_path):
    from plyfile import PlyData, PlyElement

    # Convert spherical harmonics to RGB
    SH_C0 = 0.28209479177387814
    rgb = np.stack([f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]], axis=-1)
    rgb = rgb * SH_C0 + 0.5
    rgb = np.clip(rgb, 0., 1.)
    rgb = (rgb * 255).astype(np.uint8)

    vertices = np.empty(xyz.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertices['x'] = xyz[:, 0].astype(np.float32)
    vertices['y'] = xyz[:, 1].astype(np.float32)
    vertices['z'] = xyz[:, 2].astype(np.float32)
    vertices['red'] = rgb[:, 0]
    vertices['green'] = rgb[:, 1]
    vertices['blue'] = rgb[:, 2]

    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(output_path)

def save_gaussian_ply(outputs, output_dir, basename):
    gaussian = outputs['gaussian'][0]
    xyz = gaussian.get_xyz.detach().cpu().numpy() * 0.8 + np.array([0.35, 0, 0.5])
    f_dc = gaussian._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    convert_ply(xyz, f_dc, os.path.join(output_dir, f"{basename}.ply"))
    

def main():
    args = parse_args()
    pipeline = load_pipeline(args.config, args.model_path)
    pipeline.cuda()

    outputs, decoded_gt = pipeline.run(
        data_path=args.data_path,
        num_step=args.num_steps,
        seed=args.seed,
        formats=("gaussian",),
        ss_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": 3,
        },
        instruction=args.instruction,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.render_ply:
        save_gaussian_ply(decoded_gt, args.output_dir, "gt_input")
        for n_step in range(args.num_steps):
            save_gaussian_ply(outputs[n_step], args.output_dir, f"sample_step{n_step+1}")



if __name__ == "__main__":
    main()
