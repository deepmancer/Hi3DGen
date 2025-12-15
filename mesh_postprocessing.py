#!/usr/bin/env python3
"""
Mesh Post-Processing using Blender's bpy API
Provides watertight, clean, and high-quality mesh refinement.
"""

import os
import sys
import trimesh
import numpy as np
from pathlib import Path

def postprocess_mesh_trimesh(mesh_path, output_path=None, target_faces=None):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')

        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "geometry") and mesh.geometry:
                meshes = list(mesh.geometry.values())
                mesh = max(meshes, key=lambda m: (m.area, len(m.faces)))
            else:
                raise TypeError("Loaded asset is not a triangle mesh")

        mesh = mesh.copy()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        try:
            mesh.process(validate=True)
        except Exception:
            mesh.remove_unreferenced_vertices()

        print(f"  Clean mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"  Found {len(components)} connected components")
            mesh = max(components, key=lambda m: (m.area, len(m.faces)))
            mesh.remove_unreferenced_vertices()
            print(f"  Retained largest component: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        
        if target_faces:
            target_faces = max(int(target_faces), 4)
            if len(mesh.faces) > target_faces:
                simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
                if isinstance(simplified, trimesh.Trimesh) and len(simplified.faces) >= target_faces // 2:
                    mesh = simplified
                    mesh.remove_unreferenced_vertices()
                    mesh.fix_normals()
                    print(f"  Simplified mesh to {len(mesh.faces)} faces")
                else:
                    print("  Simplification skipped (poor result)")

        if not mesh.is_watertight:
            print("  Mesh not watertight, attempting repairs")
            changed = mesh.fill_holes()
            if changed:
                print("    Filled holes")
            trimesh.repair.fix_normals(mesh, multibody=True)
            trimesh.repair.fix_inversion(mesh)
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()

        trimesh.smoothing.filter_laplacian(mesh, iterations=1)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        mesh.vertex_normals  # force normal recalculation

        is_watertight = mesh.is_watertight
        is_winding_consistent = mesh.is_winding_consistent
        print(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Watertight: {is_watertight}, Winding consistent: {is_winding_consistent}")

        if output_path is None:
            base, ext = os.path.splitext(mesh_path)
            output_path = f"{base}_processed{ext}"

        mesh.export(output_path)
        return output_path

    except Exception as e:
        print(f"Mesh post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return mesh_path

def postprocess_mesh(input_mesh_path, output_mesh_path=None, 
                   target_faces=200_000):
    
    if not os.path.exists(input_mesh_path):
        raise FileNotFoundError(f"Input mesh not found: {input_mesh_path}")
    
    if output_mesh_path is None:
        base, ext = os.path.splitext(input_mesh_path)
        output_mesh_path = f"{base}_processed{ext}"
    
    return postprocess_mesh_trimesh(
        input_mesh_path,
        output_mesh_path,
        target_faces=target_faces
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-process 3D mesh for watertightness and quality")
    parser.add_argument("input", help="Input mesh file")
    parser.add_argument("--output", help="Output mesh file")
    parser.add_argument("--target_faces", type=int, default=200_000,
                        help="Target number of faces after processing")

    args = parser.parse_args()
    
    result = postprocess_mesh(
        args.input,
        args.output,
        target_faces=100_000,
    )
    
    print(f"\nFinal mesh: {result}")
