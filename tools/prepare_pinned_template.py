#!/usr/bin/env python3
"""
Tool to prepare single-layer garment templates with pinned vertices for HOOD inference.

This script generates a single-layer .pkl template that HOOD can directly use.
The template contains vertices, faces, node_type, and coarse edges at the top level.

Usage:
    python prepare_pinned_template.py \
        --input pose8/lower/0.obj \
        --output hood/hood_data/lower_template.pkl \
        --waist-y-min 0.9 \
        --waist-y-max 0.95

The output .pkl file contains a single dictionary with:
    - vertices: np.float32 array of vertex positions
    - faces: np.int32 array of face indices
    - node_type: np.int64 array marking pinned vertices (NodeType.HANDLE=3)
    - coarse edges and other mesh metadata
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pickle

# Add HOOD repo to path
hood_root = Path(__file__).parent.parent
sys.path.insert(0, str(hood_root))

from utils.mesh_creation import (
    obj2template,
    load_obj,
    add_coarse_edges
)
from utils.common import NodeType
from utils.defaults import DEFAULTS


def find_waist_vertices(vertices, y_min=0.8, y_max=1.0):
    """
    Find vertices in the waist region based on Y coordinate range.

    Args:
        vertices: np.array of shape (V, 3)
        y_min: Minimum Y coordinate for waist region (normalized)
        y_max: Maximum Y coordinate for waist region (normalized)

    Returns:
        List of vertex indices in waist region
    """
    # Normalize Y coordinates to [0, 1]
    y_coords = vertices[:, 1]
    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

    # Find vertices in waist range
    waist_mask = (y_norm >= y_min) & (y_norm <= y_max)
    waist_indices = np.where(waist_mask)[0]

    return waist_indices.tolist()


def find_waist_vertices_by_distance(vertices, body_vertices, max_dist=0.015):
    """
    Find garment vertices that are close to body vertices in the waist region.

    Args:
        vertices: Garment vertices (V_g, 3)
        body_vertices: Body vertices (V_b, 3)
        max_dist: Maximum distance to consider as waist attachment

    Returns:
        List of garment vertex indices
    """
    from sklearn.neighbors import KDTree

    # Build KD-tree for body vertices
    body_tree = KDTree(body_vertices)

    # Find nearest body vertex for each garment vertex
    distances, _ = body_tree.query(vertices, k=1)
    distances = distances.flatten()

    # Find vertices within max_dist
    close_mask = distances < max_dist
    close_indices = np.where(close_mask)[0]

    return close_indices.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare single-layer garment template with pinned vertices for HOOD inference"
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input garment OBJ file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output template .pkl file (single-layer format)")
    parser.add_argument("--waist-y-min", type=float, default=0.9,
                        help="Min Y for waist region (normalized 0-1)")
    parser.add_argument("--waist-y-max", type=float, default=0.95,
                        help="Max Y for waist region (normalized 0-1)")
    parser.add_argument("--waist-max-dist", type=float, default=0.015,
                        help="Max distance for waist vertices in meters (requires --body-mesh)")
    parser.add_argument("--body-mesh", type=Path,
                        help="Reference body mesh for distance-based waist detection")
    parser.add_argument("--n-coarse-levels", type=int, default=4,
                        help="Number of coarse edge levels for HOOD (default: 4)")
    parser.add_argument("--process", action="store_true", default=False,
                        help="Process mesh with trimesh (WARNING: may reorder vertices, default: False)")

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Load garment from OBJ
    print(f"Loading garment from {args.input}")

    if args.process:
        # Use obj2template with mesh processing (may reorder vertices)
        print("Using obj2template with mesh processing...")
        template = obj2template(str(args.input))
    else:
        # Load without processing to preserve vertex order
        print("Loading OBJ without processing to preserve vertex order...")
        vertices, faces = load_obj(str(args.input), tex_coords=False)

        # Create template dictionary
        template = {
            'vertices': vertices.astype(np.float32),
            'faces': faces.astype(np.int32)
        }

        # Add coarse edges
        print(f"Adding coarse edges with {args.n_coarse_levels} levels...")
        template = add_coarse_edges(template, args.n_coarse_levels)

    print(f"✓ Loaded mesh: {len(template['vertices'])} vertices, {len(template['faces'])} faces")

    # Find waist vertices for pinning
    vertices = template['vertices']

    if args.body_mesh:
        # Distance-based detection
        print(f"Using distance-based waist detection with body mesh {args.body_mesh}")
        if not args.body_mesh.exists():
            print(f"WARNING: Body mesh not found: {args.body_mesh}")
            print(f"Falling back to Y-coordinate detection")
            waist_indices = find_waist_vertices(
                vertices, args.waist_y_min, args.waist_y_max
            )
        else:
            body_verts, _ = load_obj(str(args.body_mesh), tex_coords=False)
            waist_indices = find_waist_vertices_by_distance(
                vertices, body_verts, args.waist_max_dist
            )
            print(f"  Max distance threshold: {args.waist_max_dist}m")
    else:
        # Y-coordinate based detection
        print(f"Using Y-coordinate waist detection:")
        print(f"  Y range: [{args.waist_y_min:.2f}, {args.waist_y_max:.2f}] (normalized)")
        waist_indices = find_waist_vertices(
            vertices, args.waist_y_min, args.waist_y_max
        )

    # Add node_type field to template
    node_type = np.zeros((len(vertices), 1), dtype=np.int64)
    node_type[waist_indices] = NodeType.HANDLE  # 3 = pinned vertex
    template['node_type'] = node_type

    # Validate node_type length
    assert len(node_type) == len(vertices), \
        f"node_type length ({len(node_type)}) does not match vertices count ({len(vertices)})"

    # Count and report pinned vertices
    num_pinned = np.sum(node_type == NodeType.HANDLE)
    print(f"\n✓ Found {num_pinned} waist vertices out of {len(vertices)} total vertices")
    print(f"  Percentage pinned: {100*num_pinned/len(vertices):.1f}%")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save single-layer template
    with open(args.output, 'wb') as f:
        pickle.dump(template, f)

    # Report success
    print(f"\n✓ Saved single-layer template to {args.output}")
    print(f"\nTemplate structure:")
    print(f"  vertices:     {len(template['vertices'])} (np.float32)")
    print(f"  faces:        {len(template['faces'])} (np.int32)")
    print(f"  node_type:    {len(template['node_type'])} (np.int64)")
    print(f"  pinned verts: {num_pinned} ({100*num_pinned/len(vertices):.1f}%)")

    if 'coarse_edges' in template:
        print(f"  coarse_edges: {args.n_coarse_levels} levels")
    if 'center' in template:
        print(f"  center nodes: {len(template.get('center', []))} candidates")

    # Print sample pinned vertex indices for verification
    if num_pinned > 0:
        pinned_indices = np.where(node_type == NodeType.HANDLE)[0]
        sample_size = min(10, len(pinned_indices))
        print(f"\nSample pinned vertex indices: {pinned_indices[:sample_size].tolist()}")

        # Print Y-range of pinned vertices
        pinned_y = vertices[pinned_indices, 1]
        print(f"Pinned vertices Y range: [{pinned_y.min():.3f}, {pinned_y.max():.3f}]")

    print(f"\nDone! Template is ready for HOOD inference with pinned vertices.")


if __name__ == '__main__':
    main()