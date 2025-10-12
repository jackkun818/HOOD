#!/usr/bin/env python3
"""
Minimal HOOD inference wrapper.

Usage (called by pose8/hood_pipeline.py):
  python run_hood.py \
    --config /path/to/configs/aux/from_any_pose.yaml \
    --checkpoint /path/to/ckpt.pth \
    --body-sequence /abs/path/to/body_seq.npz|.pkl \
    --garment /abs/path/to/garment.obj|.pkl \
    --output-dir /abs/path/to/out_dir \
    --device cuda:0 \
    --garment-type upper
"""

import argparse
import os
import sys
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True, help="Path to HOOD config YAML")
    p.add_argument("--checkpoint", type=Path, required=False, default=None, help="Path to HOOD checkpoint .pth (optional; resolves from $HOOD_DATA/trained_models if omitted)")
    p.add_argument("--body-sequence", type=Path, required=True, help="Path to body sequence (.npz or .pkl)")
    p.add_argument("--garment", type=Path, required=True, help="Path to garment (.obj or .pkl)")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to save garment_sequence.npz")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--garment-type", type=str, choices=["upper", "lower"], default="upper")
    p.add_argument("--use-pinned-verts", action="store_true", help="Enable pinned vertices from garment template")
    return p.parse_args()


def main():
    args = parse_args()

    hood_project = os.environ.get("HOOD_PROJECT")
    hood_data = os.environ.get("HOOD_DATA")
    if not hood_project or not hood_data:
        print("[ERROR] HOOD_PROJECT and HOOD_DATA must be set in environment", file=sys.stderr)
        sys.exit(2)

    hood_project = Path(hood_project).resolve()
    hood_data = Path(hood_data).resolve()

    # Log current working directory for debugging
    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] HOOD_PROJECT: {hood_project}")
    print(f"[INFO] HOOD_DATA: {hood_data}")

    # Ensure HOOD repo on sys.path
    if str(hood_project) not in sys.path:
        sys.path.insert(0, str(hood_project))

    # Late imports (require env)
    from utils.arguments import load_params, create_modules, create_dataloader_module
    from utils.validation import load_runner_from_checkpoint
    from utils.defaults import DEFAULTS

    # Resolve checkpoint if not provided
    if args.checkpoint is None:
        trained_models = Path(DEFAULTS.data_root) / 'trained_models'
        preferred = trained_models / 'postcvpr.pth'
        if preferred.exists():
            args.checkpoint = preferred
        elif trained_models.exists():
            pths = sorted(trained_models.glob('*.pth'))
            if len(pths) == 1:
                args.checkpoint = pths[0]
            elif len(pths) > 1:
                names = ', '.join(p.name for p in pths[:10])
                more = '...' if len(pths) > 10 else ''
                print(f"[ERROR] Multiple checkpoints found under {trained_models}. Please specify --checkpoint. Candidates: {names}{more}", file=sys.stderr)
                sys.exit(2)
        if args.checkpoint is None:
            print(f"[ERROR] No --checkpoint provided and none found under {trained_models}", file=sys.stderr)
            sys.exit(2)

    # Prepare from_any_pose paths under HOOD_DATA
    fap_dir = Path(DEFAULTS.data_root) / 'fromanypose'
    fap_dir.mkdir(parents=True, exist_ok=True)

    # Body sequence: convert to .pkl if needed and place under HOOD_DATA
    mesh_seq_pkl = fap_dir / 'mesh_sequence.pkl'

    # Resolve body_sequence path (handle relative paths from parent process)
    body_seq_path = args.body_sequence
    if not body_seq_path.is_absolute():
        # Relative path - resolve from current working directory
        body_seq_path = Path.cwd() / body_seq_path

    if not body_seq_path.exists():
        print(f"[ERROR] Body sequence file not found: {body_seq_path}", file=sys.stderr)
        print(f"[ERROR] Original path: {args.body_sequence}", file=sys.stderr)
        print(f"[ERROR] Current working directory: {os.getcwd()}", file=sys.stderr)
        sys.exit(3)

    print(f"[INFO] Loading body sequence from: {body_seq_path}")

    if body_seq_path.suffix.lower() == '.npz':
        data = np.load(body_seq_path)
        verts = data['verts']
        faces = data['faces']
        with open(mesh_seq_pkl, 'wb') as f:
            pickle.dump({'verts': verts, 'faces': faces}, f)
    elif body_seq_path.suffix.lower() == '.pkl':
        shutil.copy(body_seq_path, mesh_seq_pkl)
    else:
        print(f"[ERROR] Unsupported body sequence format: {body_seq_path}", file=sys.stderr)
        sys.exit(3)

    # Garment template: copy under HOOD_DATA
    garment_rel = None

    # Resolve garment path (handle relative paths from parent process)
    garment_path = args.garment
    if not garment_path.is_absolute():
        # Relative path - resolve from current working directory
        garment_path = Path.cwd() / garment_path

    if not garment_path.exists():
        print(f"[ERROR] Garment file not found: {garment_path}", file=sys.stderr)
        print(f"[ERROR] Original path: {args.garment}", file=sys.stderr)
        print(f"[ERROR] Current working directory: {os.getcwd()}", file=sys.stderr)
        sys.exit(4)

    print(f"[INFO] Loading garment from: {garment_path}")

    if garment_path.suffix.lower() == '.obj':
        garment_dst = fap_dir / 'garment.obj'
        shutil.copy(garment_path, garment_dst)
        garment_rel = 'fromanypose/garment.obj'
    elif garment_path.suffix.lower() == '.pkl':
        garment_dst = fap_dir / 'garment.pkl'
        shutil.copy(garment_path, garment_dst)
        garment_rel = 'fromanypose/garment.pkl'

        # Validate and log pkl template info
        try:
            with open(garment_path, 'rb') as f:
                template_data = pickle.load(f)

            if 'vertices' not in template_data:
                print(f"[ERROR] Invalid pkl template: missing 'vertices' key", file=sys.stderr)
                print(f"[ERROR] Found keys: {list(template_data.keys())}", file=sys.stderr)
                print(f"[ERROR] Please use prepare_pinned_template.py with --single-layer to generate a proper template", file=sys.stderr)
                sys.exit(5)

            num_verts = len(template_data['vertices'])
            num_faces = len(template_data.get('faces', []))
            print(f"[INFO] Template validated: {num_verts} vertices, {num_faces} faces")

            if 'node_type' in template_data:
                node_type = template_data['node_type']
                if len(node_type) != num_verts:
                    print(f"[WARNING] node_type length ({len(node_type)}) does not match vertices ({num_verts})", file=sys.stderr)
                else:
                    num_pinned = np.sum(node_type == 3)  # NodeType.HANDLE = 3
                    if num_pinned > 0:
                        print(f"[INFO] Using {num_pinned} pinned vertices ({100*num_pinned/num_verts:.1f}%)")
                    else:
                        print(f"[INFO] Template has node_type but no pinned vertices")
            else:
                print(f"[INFO] Template has no node_type (all vertices free)")

        except Exception as e:
            print(f"[WARNING] Could not validate pkl template: {e}", file=sys.stderr)

    else:
        print(f"[ERROR] Unsupported garment format: {garment_path}", file=sys.stderr)
        sys.exit(4)

    # Derive config name relative to <HOOD_PROJECT>/configs
    try:
        cfg_name = str(args.config.resolve().relative_to((hood_project / 'configs').resolve()))
        cfg_name = cfg_name.replace('.yaml', '')
    except Exception:
        # Fallback to common default
        cfg_name = 'aux/from_any_pose'

    # Load config and modules
    modules, conf = load_params(cfg_name)
    conf.device = args.device

    # Override dataset settings
    ds_key = list(conf.dataloader.dataset.keys())[0]
    conf.dataloader.dataset[ds_key].pose_sequence_type = 'mesh'
    conf.dataloader.dataset[ds_key].pose_sequence_path = 'fromanypose/mesh_sequence.pkl'
    conf.dataloader.dataset[ds_key].garment_template_path = garment_rel
    conf.dataloader.batch_size = 1
    conf.dataloader.num_workers = 0

    # Enable pinned vertices if requested
    if args.use_pinned_verts:
        conf.dataloader.dataset[ds_key].pinned_verts = True
        print(f"[INFO] Enabled pinned vertices from garment template")

    # Build dataloader and runner
    print(f"[DBG] conf.device={conf.device}")
    print(f"[DBG] torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            print(f"[DBG] torch.cuda.current_device={idx}, name={torch.cuda.get_device_name(idx)}")
        except Exception as e:
            print(f"[DBG] cuda device query failed: {e}")

    dataloader_m = create_dataloader_module(modules, conf)
    dataloader = dataloader_m.create_dataloader(is_eval=True)
    _, runner = load_runner_from_checkpoint(str(args.checkpoint), modules, conf)
    # Move model to the requested device to match sample device inside valid_rollout
    try:
        device = torch.device(args.device)
    except Exception:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    runner.to(device)
    runner.eval()
    try:
        pdev = next(runner.parameters()).device
        print(f"[DBG] runner.param.device={pdev}")
    except Exception as e:
        print(f"[DBG] runner param device check failed: {e}")

    # Single-sample inference
    batch = next(iter(dataloader))
    # Print a few key tensors' devices before valid_rollout (they will be moved inside)
    try:
        cloth_pos = batch['cloth'].pos
        obst_pos = batch['obstacle'].pos
        print(f"[DBG] sample devices pre-move: cloth.pos={cloth_pos.device}, obstacle.pos={obst_pos.device}")
    except Exception as e:
        print(f"[DBG] sample device check failed: {e}")
    trajectories = runner.valid_rollout(batch, bare=True)

    verts_seq = trajectories['pred']  # [T, V, 3]
    cloth_faces = trajectories['cloth_faces']  # [F, 3]

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / 'garment_sequence.npz'
    np.savez_compressed(out_path, verts=verts_seq, faces=cloth_faces)
    print(f"[INFO] Saved HOOD garment sequence to {out_path}")


if __name__ == '__main__':
    main()
