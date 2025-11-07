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
import igl


# From Hunyuan3D-2.1
def watertight(V, F, grid_res=512):
    # Compute bounding box
    epsilon = 2.0 / grid_res
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # Create a uniform grid
    x = np.linspace(min_corner[0], max_corner[0], grid_res)
    y = np.linspace(min_corner[1], max_corner[1], grid_res)
    z = np.linspace(min_corner[2], max_corner[2], grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Compute SDF at grid points using igl.signed_distance with pseudo normals
    # igl.signed_distance returns (S, I, C, N) - distances, face indices, closest points, normals
    sdf, _, _, _ = igl.signed_distance(
        grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    
    # Reshape SDF to grid dimensions for marching cubes
    sdf_grid = sdf.reshape(grid_res, grid_res, grid_res)
 
    # igl.marching_cubes returns (vertices, faces) from the scalar field
    # Pass the SDF grid and grid spacing information
    mc_verts, mc_faces = igl.marching_cubes(sdf_grid, 0.0)
    
    # Scale vertices back to original coordinate space
    mc_verts = mc_verts / (grid_res - 1)  # Normalize to [0, 1]
    for i in range(3):
        mc_verts[:, i] = mc_verts[:, i] * (max_corner[i] - min_corner[i]) + min_corner[i]

    # mc_verts: (k x 3) array of vertices of the epsilon contour
    # mc_faces: (l x 3) array of faces of the epsilon contour
    return mc_verts, mc_faces

def postprocess_mesh_trimesh(mesh_path, output_path=None, target_faces=None):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                meshes = list(mesh.geometry.values())
                mesh = meshes[0]
                
                if len(meshes) > 1:
                    areas = [m.area for m in meshes]
                    mesh = meshes[np.argmax(areas)]
        
        mesh.merge_vertices()
        mesh.fix_normals()
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_infinite_values()
        mesh.fill_holes()
        
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda m: m.vertices.shape[0])
            mesh = largest
        
        # # Ensure watertightness by keeping only the largest connected component
        

        # Fix normals - ensure consistent orientation

        if target_faces is not None and mesh.faces.shape[0] > target_faces:
            # Calculate the target reduction ratio (between 0 and 1)
            # target_faces is the desired number of faces
            current_faces = mesh.faces.shape[0]
            target_ratio = target_faces / current_faces
            
            # Ensure ratio is valid (between 0 and 1)
            target_ratio = max(0.01, min(0.99, target_ratio))
            print(f"Simplifying mesh from {current_faces} to approx {target_faces} faces (ratio: {target_ratio:.4f})")
            mesh = mesh.simplify_quadric_decimation(target_ratio)
            # Recompute normals after decimation
            mesh.face_normals
            mesh.vertex_normals

        # V, F = mesh.vertices, mesh.faces
        # V, F = watertight(V, F, grid_res=512)
        # mesh = trimesh.Trimesh(vertices=V, faces=F)

        # Fix winding order to ensure consistent face orientation
        
        # Recompute face and vertex normals
        mesh.fix_normals()
        mesh.face_normals
        mesh.vertex_normals
        
        # Check if normals are inverted (most faces pointing inward)
        # by testing if the mesh volume is negative
        if hasattr(mesh, 'is_volume') and mesh.is_volume:
            if mesh.volume < 0:
                mesh.invert()

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
