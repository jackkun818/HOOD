#!/usr/bin/env python
"""
HOOD Worker Process - Persistent inference server for garment simulation.
Runs in the HOOD conda environment and handles inference requests via JSON protocol.
"""

import sys
import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import uuid
import os
import pickle
import shutil

import torch
import numpy as np

# Add HOOD modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import HOOD utilities (same as run_hood.py)
from utils.arguments import load_params, create_modules, create_dataloader_module
from utils.validation import load_runner_from_checkpoint
from utils.defaults import DEFAULTS


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[WORKER][%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class HOODInferenceWorker:
    """Persistent worker for HOOD inference."""

    def __init__(self):
        self.modules = None
        self.conf = None
        self.runner = None
        self.config_path = None
        self.config_name = None
        self.checkpoint_path = None
        self.device = None
        self.initialized = False
        self.fap_dir = None  # fromanypose directory

    def _resolve_checkpoint(self, checkpoint_path: Optional[str]) -> Path:
        """Resolve checkpoint path according to policy:
        - If an explicit path is provided, use it as-is (relative to CWD if not absolute).
          Error if not found.
        - If no path provided, attempt to derive from $HOOD_DATA/trained_models/.
          Prefer 'postcvpr.pth'; otherwise, if exactly one .pth exists, use it; else error.
        """
        # Worker runs with CWD at HOOD_PROJECT
        cwd = Path.cwd()

        if checkpoint_path:
            cand = Path(checkpoint_path)
            if not cand.is_absolute():
                cand = cwd / cand
            if not cand.exists():
                raise FileNotFoundError(f"Checkpoint not found: {cand}")
            return cand

        # No explicit checkpoint provided: derive from HOOD_DATA
        from utils.defaults import DEFAULTS
        trained_models = Path(DEFAULTS.data_root) / 'trained_models'
        preferred = trained_models / 'postcvpr.pth'
        tried = [preferred]
        if preferred.exists():
            return preferred

        if trained_models.exists():
            pths = sorted(trained_models.glob('*.pth'))
            if len(pths) == 1:
                return pths[0]
            elif len(pths) > 1:
                names = ', '.join(p.name for p in pths[:10])
                more = '...' if len(pths) > 10 else ''
                raise RuntimeError(
                    f"Multiple checkpoints found under {trained_models}. "
                    f"Please specify one explicitly. Candidates: {names}{more}"
                )

        raise FileNotFoundError(
            f"No checkpoint provided and none found under {trained_models}. "
            f"Tried: {', '.join(str(p) for p in tried)}"
        )

    def initialize(self, config_path: str, checkpoint_path: Optional[str], device: str = 'cuda') -> bool:
        """Initialize model and runner with config and checkpoint."""
        try:
            logger.debug(
                f"Initializing with config={config_path}, checkpoint={checkpoint_path}, device={device}"
            )

            # Store paths
            self.config_path = Path(config_path)
            # Resolve checkpoint path per policy
            resolved_ckpt = self._resolve_checkpoint(checkpoint_path)
            self.checkpoint_path = resolved_ckpt
            self.device = device

            # Get HOOD project root
            hood_project = Path(__file__).parent.parent.resolve()

            # Derive config name relative to <HOOD_PROJECT>/configs
            try:
                cfg_name = str(self.config_path.resolve().relative_to((hood_project / 'configs').resolve()))
                cfg_name = cfg_name.replace('.yaml', '')
            except Exception:
                # Fallback to common default
                cfg_name = 'aux/from_any_pose'
            self.config_name = cfg_name

            # Load config and modules (same as run_hood.py)
            self.modules, self.conf = load_params(cfg_name)
            self.conf.device = device

            # Load runner from checkpoint
            _, self.runner = load_runner_from_checkpoint(str(self.checkpoint_path), self.modules, self.conf)

            # Move runner to device
            device_obj = torch.device(device)
            self.runner.to(device_obj)
            self.runner.eval()

            # Prepare fromanypose directory under HOOD_DATA
            self.fap_dir = Path(DEFAULTS.data_root) / 'fromanypose'
            self.fap_dir.mkdir(parents=True, exist_ok=True)

            self.initialized = True
            logger.debug("Initialization complete")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def run_inference(self,
                     body_sequence_path: str,
                     garment_path: str,
                     output_dir: str,
                     garment_type: str = 'upper',
                     use_pinned_verts: bool = False,
                     request_id: Optional[str] = None) -> Dict[str, Any]:
        """Run inference on a single garment."""
        if not self.initialized:
            return {
                'success': False,
                'error': 'Worker not initialized',
                'request_id': request_id
            }

        try:
            logger.debug(f"[{request_id}] Starting inference: garment_type={garment_type}")
            logger.debug(f"[{request_id}] Body sequence: {body_sequence_path}")
            logger.debug(f"[{request_id}] Garment: {garment_path}")
            logger.debug(f"[{request_id}] Output dir: {output_dir}")

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Resolve paths (handle relative paths)
            body_seq_path = Path(body_sequence_path)
            if not body_seq_path.is_absolute():
                body_seq_path = Path.cwd() / body_seq_path

            garment_path = Path(garment_path)
            if not garment_path.is_absolute():
                garment_path = Path.cwd() / garment_path

            # Prepare body sequence under HOOD_DATA/fromanypose (same as run_hood.py)
            mesh_seq_pkl = self.fap_dir / 'mesh_sequence.pkl'

            if body_seq_path.suffix.lower() == '.npz':
                data = np.load(body_seq_path)
                verts = data['verts'] if 'verts' in data else data['vertices']
                faces = data['faces']
                with open(mesh_seq_pkl, 'wb') as f:
                    pickle.dump({'verts': verts, 'faces': faces}, f)
            elif body_seq_path.suffix.lower() == '.pkl':
                shutil.copy(body_seq_path, mesh_seq_pkl)
            else:
                raise ValueError(f"Unsupported body sequence format: {body_seq_path}")

            # Prepare garment under HOOD_DATA/fromanypose (same as run_hood.py)
            if garment_path.suffix.lower() == '.obj':
                garment_dst = self.fap_dir / 'garment.obj'
                shutil.copy(garment_path, garment_dst)
                garment_rel = 'fromanypose/garment.obj'
            elif garment_path.suffix.lower() == '.pkl':
                garment_dst = self.fap_dir / 'garment.pkl'
                shutil.copy(garment_path, garment_dst)
                garment_rel = 'fromanypose/garment.pkl'

                # Validate and log pkl template info
                try:
                    with open(garment_path, 'rb') as f:
                        template_data = pickle.load(f)

                    if 'vertices' not in template_data:
                        raise ValueError(
                            f"Invalid pkl template: missing 'vertices' key. "
                            f"Found keys: {list(template_data.keys())}. "
                            f"Please use prepare_pinned_template.py with --single-layer to generate a proper template."
                        )

                    num_verts = len(template_data['vertices'])
                    num_faces = len(template_data.get('faces', []))
                    logger.debug(
                        f"[{request_id}] Template validated: {num_verts} vertices, {num_faces} faces"
                    )

                    if 'node_type' in template_data:
                        node_type = template_data['node_type']
                        if len(node_type) != num_verts:
                            logger.debug(
                                f"[{request_id}] node_type length ({len(node_type)}) does not match vertices ({num_verts})"
                            )
                        else:
                            num_pinned = int(np.sum(node_type == 3))  # NodeType.HANDLE = 3
                            if num_pinned > 0:
                                logger.debug(
                                    f"[{request_id}] Using {num_pinned} pinned vertices ({100 * num_pinned / num_verts:.1f}%)"
                                )
                            else:
                                logger.debug(f"[{request_id}] Template has node_type but no pinned vertices")
                    else:
                        logger.debug(f"[{request_id}] Template has no node_type (all vertices free)")

                except Exception as e:
                    logger.debug(f"[{request_id}] Could not validate pkl template: {e}")

            else:
                raise ValueError(f"Unsupported garment format: {garment_path}")

            # Update dataset configuration (same as run_hood.py)
            ds_key = list(self.conf.dataloader.dataset.keys())[0]
            self.conf.dataloader.dataset[ds_key].pose_sequence_type = 'mesh'
            self.conf.dataloader.dataset[ds_key].pose_sequence_path = 'fromanypose/mesh_sequence.pkl'
            self.conf.dataloader.dataset[ds_key].garment_template_path = garment_rel
            self.conf.dataloader.batch_size = 1
            self.conf.dataloader.num_workers = 0

            # Enable pinned vertices if requested
            if use_pinned_verts:
                self.conf.dataloader.dataset[ds_key].pinned_verts = True
                logger.debug(f"[{request_id}] Enabled pinned vertices from garment template")

            # Create dataloader (same as run_hood.py)
            dataloader_m = create_dataloader_module(self.modules, self.conf)
            dataloader = dataloader_m.create_dataloader(is_eval=True)

            # Get batch and run inference (same as run_hood.py)
            batch = next(iter(dataloader))

            # Log device info for debugging
            try:
                cloth_pos = batch['cloth'].pos
                obst_pos = batch['obstacle'].pos
                logger.debug(f"[{request_id}] Sample devices: cloth.pos={cloth_pos.device}, obstacle.pos={obst_pos.device}")
            except Exception as e:
                logger.debug(f"[{request_id}] Device check: {e}")

            # Run valid_rollout (same as run_hood.py)
            trajectories = self.runner.valid_rollout(batch, bare=True)

            verts_seq = trajectories['pred']  # [T, V, 3]
            cloth_faces = trajectories['cloth_faces']  # [F, 3]

            # Save output (same as run_hood.py)
            output_path = output_dir / 'garment_sequence.npz'
            np.savez_compressed(output_path, verts=verts_seq, faces=cloth_faces)

            logger.debug(f"[{request_id}] Saved results to {output_path}")

            return {
                'success': True,
                'output_path': str(output_path),
                'num_frames': verts_seq.shape[0] if hasattr(verts_seq, 'shape') else len(verts_seq),
                'num_vertices': verts_seq.shape[1] if hasattr(verts_seq, 'shape') else verts_seq[0].shape[0],
                'num_faces': cloth_faces.shape[0] if hasattr(cloth_faces, 'shape') else len(cloth_faces),
                'request_id': request_id
            }

        except Exception as e:
            logger.error(f"[{request_id}] Inference failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'request_id': request_id
            }

    def reload_config(self, config_path: str, checkpoint_path: str, device: str = 'cuda') -> bool:
        """Reload with new config/checkpoint."""
        logger.debug("Reloading configuration...")
        self.cleanup()
        return self.initialize(config_path, checkpoint_path, device)

    def cleanup(self):
        """Clean up resources."""
        if self.runner is not None:
            del self.runner
            self.runner = None
        if self.modules is not None:
            del self.modules
            self.modules = None
        if self.conf is not None:
            del self.conf
            self.conf = None
        self.initialized = False
        torch.cuda.empty_cache()
        logger.debug("Cleanup complete")


def main():
    """Main worker loop."""
    logger.debug("HOOD Worker starting...")
    logger.debug(f"Python: {sys.executable}")
    logger.debug(f"PyTorch: {torch.__version__}")
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Check environment variables required by HOOD
    hood_project = os.environ.get("HOOD_PROJECT")
    hood_data = os.environ.get("HOOD_DATA")

    if not hood_project or not hood_data:
        logger.error("HOOD_PROJECT and HOOD_DATA must be set in environment")
        sys.exit(2)

    logger.debug(f"HOOD_PROJECT: {hood_project}")
    logger.debug(f"HOOD_DATA: {hood_data}")

    worker = HOODInferenceWorker()

    # Process commands from stdin
    while True:
        try:
            # Read JSON command from stdin
            line = sys.stdin.readline()
            if not line:
                logger.debug("EOF received, shutting down")
                break

            # Parse command
            try:
                command = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                response = {'success': False, 'error': f'Invalid JSON: {e}'}
                print(json.dumps(response), flush=True)
                continue

            # Generate request ID if not provided
            request_id = command.get('request_id', str(uuid.uuid4())[:8])
            logger.debug(f"[{request_id}] Received command: {command.get('action', 'unknown')}")

            # Process command based on action
            action = command.get('action')

            if action == 'initialize':
                success = worker.initialize(
                    config_path=command['config_path'],
                    checkpoint_path=command.get('checkpoint_path'),
                    device=command.get('device', 'cuda')
                )
                response = {
                    'success': success,
                    'request_id': request_id,
                    'action': 'initialize'
                }

            elif action == 'infer':
                if not worker.initialized:
                    # Auto-initialize if needed (checkpoint may be omitted; worker will resolve from HOOD_DATA)
                    if 'config_path' in command:
                        worker.initialize(
                            config_path=command['config_path'],
                            checkpoint_path=command.get('checkpoint_path'),
                            device=command.get('device', 'cuda')
                        )

                response = worker.run_inference(
                    body_sequence_path=command['body_sequence_path'],
                    garment_path=command['garment_path'],
                    output_dir=command['output_dir'],
                    garment_type=command.get('garment_type', 'upper'),
                    use_pinned_verts=command.get('use_pinned_verts', False),
                    request_id=request_id
                )
                response['action'] = 'infer'

            elif action == 'reload':
                success = worker.reload_config(
                    config_path=command['config_path'],
                    checkpoint_path=command['checkpoint_path'],
                    device=command.get('device', 'cuda')
                )
                response = {
                    'success': success,
                    'request_id': request_id,
                    'action': 'reload'
                }

            elif action == 'shutdown':
                logger.debug(f"[{request_id}] Shutdown requested")
                response = {
                    'success': True,
                    'request_id': request_id,
                    'action': 'shutdown'
                }
                print(json.dumps(response), flush=True)
                break

            elif action == 'ping':
                response = {
                    'success': True,
                    'request_id': request_id,
                    'action': 'ping',
                    'initialized': worker.initialized
                }

            else:
                response = {
                    'success': False,
                    'error': f'Unknown action: {action}',
                    'request_id': request_id
                }

            # Send response
            print(json.dumps(response), flush=True)

        except KeyboardInterrupt:
            logger.debug("Keyboard interrupt, shutting down")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            response = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            print(json.dumps(response), flush=True)

    # Cleanup
    worker.cleanup()
    logger.debug("HOOD Worker shutdown complete")


if __name__ == '__main__':
    main()
