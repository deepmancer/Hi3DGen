#!/usr/bin/env python3
"""
Hi3DGen Standalone Inference Script
Generate 3D meshes from images without Gradio interface.

Usage:
    python inference_hi3dgen.py --input /path/to/image.jpg --output /path/to/mesh.glb
    python inference_hi3dgen.py --input_dir /path/to/images/ --output_dir /path/to/meshes/
"""

import os
os.environ['SPCONV_ALGO'] = 'native'

import torch
import numpy as np
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import glob
import trimesh
from hi3dgen.pipelines import Hi3DGenPipeline
from mesh_postprocessing import postprocess_mesh


def estimate_normal(image, normal_predictor, resolution=1024, match_input_resolution=True, num_inference_steps=None):
    # Use the NiRNE predictor wrapper interface to produce a normal map (PIL.Image)
    kwargs = {
        'match_input_resolution': match_input_resolution,
        'num_inference_steps': num_inference_steps,
    }
    return normal_predictor(image, resolution=resolution, **kwargs)


def infer_image_single(
    input_path, output_path, hi3dgen_pipeline, normal_predictor, *,
    preprocess_resolution=1024, normal_resolution=1024, seed=42,
    ss_sampling_steps=50, ss_guidance_strength=3.0,
    slat_sampling_steps=6, slat_guidance_strength=3.0,
    skip_preprocess=False, skip_normal=False, save_normal=False,
    output_format='glb', enable_postprocessing=True, target_faces=200_000, 
    **kwargs,
):
    print(f"\nProcessing: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False

    try:
        image = Image.open(input_path)

        if not skip_preprocess:
            print("  Preprocessing image...")
            image = hi3dgen_pipeline.preprocess_image(image, resolution=preprocess_resolution)

        if skip_normal:
            print("  Using input as normal map...")
            normal_image = image
        else:
            print("  Generating normal map using NiRNE...")
            with torch.no_grad():
                normal_image = estimate_normal(image, normal_predictor, resolution=normal_resolution,
                                               match_input_resolution=True)

        if save_normal and not skip_normal:
            output_dir = os.path.dirname(output_path) or "."
            normal_path = os.path.join(output_dir, "surface_normal.png")
            os.makedirs(output_dir, exist_ok=True)
            normal_image.save(normal_path)
            print(f"  Normal map saved to: {normal_path}")

        print("  Generating 3D mesh...")
        with torch.no_grad():
            outputs = hi3dgen_pipeline.run(
                normal_image,
                seed=seed,
                formats=["mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )

        generated_mesh = outputs['mesh'][0]

        print("  Exporting mesh...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        
        if enable_postprocessing:
            temp_output = output_path + ".tmp" + os.path.splitext(output_path)[1]
            trimesh_mesh.export(temp_output)
            
            print("  Post-processing mesh...")
            try:
                processed_mesh_path = postprocess_mesh(
                    temp_output,
                    output_path,
                    target_faces=target_faces,
                )

                if os.path.exists(temp_output) and temp_output != processed_mesh_path:
                    os.remove(temp_output)
                
                if processed_mesh_path != output_path and os.path.exists(processed_mesh_path):
                    import shutil
                    shutil.move(processed_mesh_path, output_path)
                    
            except Exception as e:
                print(f"  Warning: Post-processing failed: {e}")
                if os.path.exists(temp_output):
                    import shutil
                    shutil.move(temp_output, output_path)
        else:
            trimesh_mesh.export(output_path)

        print(f"  Mesh saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False


def infer_image_multiview(
    image_paths, output_path, hi3dgen_pipeline, normal_predictor, *,
    preprocess_resolution=1024, normal_resolution=1024, seed=42,
    ss_sampling_steps=50, ss_guidance_strength=3.0,
    slat_sampling_steps=6, slat_guidance_strength=3.0,
    skip_preprocess=False, skip_normal=False, save_normal=False,
    output_format='glb', enable_postprocessing=True, target_faces=200_000,
    multiview_mode='stochastic',
    **kwargs,
):
    print(f"\nProcessing multiview ({len(image_paths)} views)")
    
    try:
        images = [Image.open(p) for p in image_paths]
        
        if not skip_preprocess:
            print("  Preprocessing images...")
            images = [hi3dgen_pipeline.preprocess_image(img, resolution=preprocess_resolution) for img in images]
        
        if skip_normal:
            print("  Using inputs as normal maps...")
            normal_images = images
        else:
            print("  Generating normal maps using NiRNE...")
            normal_images = []
            with torch.no_grad():
                for img in images:
                    normal_img = estimate_normal(img, normal_predictor, resolution=normal_resolution,
                                                 match_input_resolution=True)
                    normal_images.append(normal_img)
        
        if save_normal and not skip_normal:
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            for i, normal_img in enumerate(normal_images):
                normal_path = os.path.join(output_dir, f"surface_normal_view{i:02d}.png")
                normal_img.save(normal_path)
            print(f"  Normal maps saved to: {output_dir}/surface_normal_view*.png")
        
        print("  Generating 3D mesh from multiview...")
        with torch.no_grad():
            outputs = hi3dgen_pipeline.run_multi_image(
                normal_images,
                seed=seed,
                formats=["mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                mode=multiview_mode,
            )
        
        generated_mesh = outputs['mesh'][0]
        
        print("  Exporting mesh...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
        
        if enable_postprocessing:
            temp_output = output_path + ".tmp" + os.path.splitext(output_path)[1]
            trimesh_mesh.export(temp_output)
            
            print("  Post-processing mesh...")
            try:
                processed_mesh_path = postprocess_mesh(
                    temp_output,
                    output_path,
                    target_faces=target_faces,
                )
                
                if os.path.exists(temp_output) and temp_output != processed_mesh_path:
                    os.remove(temp_output)
                
                if processed_mesh_path != output_path and os.path.exists(processed_mesh_path):
                    import shutil
                    shutil.move(processed_mesh_path, output_path)
                    
            except Exception as e:
                print(f"  Warning: Post-processing failed: {e}")
                if os.path.exists(temp_output):
                    import shutil
                    shutil.move(temp_output, output_path)
        else:
            trimesh_mesh.export(output_path)
        
        print(f"  Mesh saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Error processing multiview images: {e}")
        import traceback
        traceback.print_exc()
        return False


def infer_directory_single(input_dir, output_dir, hi3dgen_pipeline, normal_predictor, **kwargs):
    print(f"Processing images from directory: {input_dir}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))

    if len(image_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    # Filter out images that already have output meshes
    images_to_process = []
    for image_path in image_paths:
        sample_id = os.path.splitext(os.path.basename(image_path))[0]
        sample_output_dir = os.path.join(output_dir, sample_id)
        output_path = os.path.join(sample_output_dir, f"shape_mesh.{kwargs.get('output_format','glb')}")
        
        if not os.path.exists(output_path):
            images_to_process.append(image_path)
    
    total_images = len(image_paths)
    already_processed = total_images - len(images_to_process)
    
    print(f"Total images: {total_images}")
    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {len(images_to_process)}")
    
    if len(images_to_process) == 0:
        print("All images have already been processed!")
        return

    success_count = 0
    for image_path in tqdm(images_to_process, desc="Processing images"):
        # Get sample_id from image filename (without extension)
        sample_id = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create sample-specific output directory
        sample_output_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Output mesh always named shape_mesh.glb
        output_path = os.path.join(sample_output_dir, f"shape_mesh.{kwargs.get('output_format','glb')}")
        
        if infer_image_single(image_path, output_path, hi3dgen_pipeline, normal_predictor, **kwargs):
            success_count += 1

    print(f"\n✓ Processing complete! {success_count}/{len(images_to_process)} meshes generated successfully")
    print(f"  Output directory: {output_dir}")


def infer_directory_multiview(input_dir, output_dir, hi3dgen_pipeline, normal_predictor, **kwargs):
    print(f"Processing multiview samples from directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input_dir contains subdirectories (multi-sample) or just images (single-sample)
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    if subdirs:
        # Multi-sample mode: each subfolder is a sample with multiple views
        
        # Filter out samples that already have output meshes
        samples_to_process = []
        for sample_id in sorted(subdirs):
            sample_output_dir = os.path.join(output_dir, sample_id)
            output_path = os.path.join(sample_output_dir, f"shape_mesh.{kwargs.get('output_format','glb')}")
            
            if not os.path.exists(output_path):
                samples_to_process.append(sample_id)
        
        total_samples = len(subdirs)
        already_processed = total_samples - len(samples_to_process)
        
        print(f"Total samples: {total_samples}")
        print(f"Already processed: {already_processed}")
        print(f"Remaining to process: {len(samples_to_process)}")
        
        if len(samples_to_process) == 0:
            print("All samples have already been processed!")
            return
        
        print(f"Found {len(samples_to_process)} samples to process (multi-sample mode)")
        success_count = 0
        
        for sample_id in tqdm(samples_to_process, desc="Processing samples"):
            sample_dir = os.path.join(input_dir, sample_id)
            image_paths = sorted([
                os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                if os.path.isfile(os.path.join(sample_dir, f)) and 
                os.path.splitext(f)[1] in image_extensions
            ])
            
            if len(image_paths) == 0:
                print(f"\n  Warning: No images found in {sample_dir}, skipping...")
                continue
            
            # Create sample-specific output directory
            sample_output_dir = os.path.join(output_dir, sample_id)
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Output mesh always named shape_mesh.glb
            output_path = os.path.join(sample_output_dir, f"shape_mesh.{kwargs.get('output_format','glb')}")
            
            if infer_image_multiview(image_paths, output_path, hi3dgen_pipeline, normal_predictor, **kwargs):
                success_count += 1
        
        print(f"\n✓ Processing complete! {success_count}/{len(samples_to_process)} meshes generated successfully")
    else:
        # Single-sample mode: all images in directory are views of one sample
        image_paths = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and 
            os.path.splitext(f)[1] in image_extensions
        ])
        
        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            return
        
        sample_id = os.path.basename(input_dir.rstrip('/'))
        
        # Create sample-specific output directory
        sample_output_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Output mesh always named shape_mesh.glb
        output_path = os.path.join(sample_output_dir, f"shape_mesh.{kwargs.get('output_format','glb')}")
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"Sample already processed: {output_path}")
            print("Skipping...")
            return
        
        print(f"Found {len(image_paths)} views (single-sample mode)")
        
        if infer_image_multiview(image_paths, output_path, hi3dgen_pipeline, normal_predictor, **kwargs):
            print(f"\n✓ Processing complete! Mesh generated successfully")
        else:
            print(f"\n✗ Processing failed!")
    
    print(f"  Output directory: {output_dir}")


def main(input, output, input_dir, output_dir, model_path, nirne_weights_dir, device,
            preprocess_resolution, normal_resolution, seed, output_format,
            ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps,
            skip_preprocess, skip_normal, save_normal, enable_postprocessing, target_faces,
            multiview_inference, multiview_mode):

    # Setup device
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        dev = torch.device('cpu')

    print(f"Using device: {dev}")

    # Load Hi3DGen pipeline
    print("Loading Hi3DGen pipeline...")
    try:
        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(model_path)
        if dev.type == "cuda":
            hi3dgen_pipeline.cuda()
        print("Hi3DGen pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading Hi3DGen pipeline: {e}")
        return

    # Load NiRNE predictor (unless skipping)
    normal_predictor = None
    if not skip_normal:
        print("Loading NiRNE normal predictor...")
        try:
            import sys
            nirne_path = os.path.join(os.path.dirname(__file__), 'NiRNE')
            if nirne_path not in sys.path:
                sys.path.insert(0, nirne_path)

            from nirne.pipeline_yoso_normal import YOSONormalsPipeline

            yoso_weight_path = os.path.join(nirne_weights_dir, 'NiRNE')
            if not os.path.exists(yoso_weight_path):
                print(f"Error: NiRNE weights not found at {yoso_weight_path}")
                print("Please download weights first using: python download_nirne_weights.py")
                return

            pipe = YOSONormalsPipeline.from_pretrained(
                yoso_weight_path,
                trust_remote_code=True,
                safety_checker=None,
                variant="fp16",
                torch_dtype=torch.float16,
                t_start=0
            ).to(dev)

            class NiRNEPredictor:
                def __init__(self, model):
                    self.model = model

                def to(self, device):
                    self.model.to(device)
                    return self

                @torch.no_grad()
                def __call__(self, img, resolution=1024, **kwargs):
                    from PIL import Image
                    import numpy as np

                    orig_size = img.size
                    if img.mode == 'RGBA':
                        rgb = img.convert('RGB')
                        alpha = img.split()[-1]
                        white_bg = Image.new('RGB', img.size, (255, 255, 255))
                        img = Image.composite(rgb, white_bg, alpha)

                    input_image_np = np.asarray(img)
                    H, W = float(input_image_np.shape[0]), float(input_image_np.shape[1])
                    k = float(resolution) / max(H, W)
                    new_H = int(np.round(H * k / 64.0)) * 64
                    new_W = int(np.round(W * k / 64.0)) * 64
                    img = img.resize((new_W, new_H), Image.Resampling.LANCZOS)

                    pipe_kwargs = {}
                    if 'num_inference_steps' in kwargs and kwargs['num_inference_steps'] is not None:
                        pipe_kwargs['num_inference_steps'] = kwargs['num_inference_steps']

                    match_input_resolution = kwargs.get('match_input_resolution', True)
                    pipe_out = self.model(img, match_input_resolution=match_input_resolution, **pipe_kwargs)

                    prediction = pipe_out.prediction[0]
                    normal_map = (prediction.clip(-1, 1) + 1) / 2
                    normal_map = (normal_map * 255).astype(np.uint8)
                    normal_map = Image.fromarray(normal_map)

                    if match_input_resolution:
                        normal_map = normal_map.resize(orig_size, Image.Resampling.LANCZOS)

                    return normal_map

            normal_predictor = NiRNEPredictor(pipe)
            print("NiRNE predictor loaded successfully!")
        except Exception as e:
            print(f"Error loading NiRNE predictor: {e}")
            import traceback
            traceback.print_exc()
            return

    # Dispatch work
    if input:
        if multiview_inference:
            parser.error("Single image input (--input) cannot be used with --multiview_inference")
        
        ok = infer_image_single(
            input, output, hi3dgen_pipeline, normal_predictor,
            preprocess_resolution=preprocess_resolution,
            normal_resolution=normal_resolution,
            seed=seed,
            ss_sampling_steps=ss_sampling_steps,
            ss_guidance_strength=ss_guidance_strength,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
            skip_preprocess=skip_preprocess,
            skip_normal=skip_normal,
            save_normal=save_normal,
            output_format=output_format,
            enable_postprocessing=enable_postprocessing,
            target_faces=target_faces,
        )
        if ok:
            print("\n✓ Processing complete!")
        else:
            print("\n✗ Processing failed!")

    elif input_dir:
        if multiview_inference:
            infer_directory_multiview(
                input_dir, output_dir, hi3dgen_pipeline, normal_predictor,
                preprocess_resolution=preprocess_resolution,
                normal_resolution=normal_resolution,
                seed=seed,
                ss_sampling_steps=ss_sampling_steps,
                ss_guidance_strength=ss_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                skip_preprocess=skip_preprocess,
                skip_normal=skip_normal,
                save_normal=save_normal,
                output_format=output_format,
                enable_postprocessing=enable_postprocessing,
                target_faces=target_faces,
                multiview_mode=multiview_mode,
            )
        else:
            infer_directory_single(
                input_dir, output_dir, hi3dgen_pipeline, normal_predictor,
                preprocess_resolution=preprocess_resolution,
                normal_resolution=normal_resolution,
                seed=seed,
                ss_sampling_steps=ss_sampling_steps,
                ss_guidance_strength=ss_guidance_strength,
                slat_sampling_steps=slat_sampling_steps,
                slat_guidance_strength=slat_guidance_strength,
                skip_preprocess=skip_preprocess,
                skip_normal=skip_normal,
                save_normal=save_normal,
                output_format=output_format,
                enable_postprocessing=enable_postprocessing,
                target_faces=target_faces,
            )


if __name__ == "__main__":
    # NOTE: Argument parser must be defined here per refactor requirements
    parser = argparse.ArgumentParser(description="Hi3DGen 3D Mesh Generation Inference")

    # Input/Output
    parser.add_argument("--input", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, help="Path to output mesh file")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing images")
    parser.add_argument("--output_dir", type=str, help="Path to output directory for meshes")

    # Model
    parser.add_argument("--model_path", type=str, default="weights/trellis-normal-v0-1",
                        help="Path to Hi3DGen model weights (default: weights/trellis-normal-v0-1)")
    parser.add_argument("--nirne_weights_dir", type=str, default="./weights",
                        help="Directory to cache NiRNE weights (default: ./weights)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run inference on")

    # Processing
    parser.add_argument("--preprocess_resolution", type=int, default=1024,
                        help="Resolution for image preprocessing (default: 1024)")
    parser.add_argument("--normal_resolution", type=int, default=1024,
                        help="Resolution for normal estimation (default: 768)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_format", type=str, default="glb",
                        choices=["glb", "obj", "ply", "stl"],
                        help="Output mesh format (default: glb)")

    # Sampler params
    parser.add_argument("--ss_guidance_strength", type=float, default=3.0,
                        help="Guidance strength for sparse structure generation (default: 3.0)")
    parser.add_argument("--ss_sampling_steps", type=int, default=50,
                        help="Sampling steps for sparse structure generation (default: 50)")
    parser.add_argument("--slat_guidance_strength", type=float, default=3.0,
                        help="Guidance strength for structured latent generation (default: 3.0)")
    parser.add_argument("--slat_sampling_steps", type=int, default=6,
                        help="Sampling steps for structured latent generation (default: 6)")

    # Options
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip image preprocessing (use if image is already preprocessed)")
    parser.add_argument("--skip_normal", action="store_true",
                        help="Skip normal estimation (use if input is already a normal map)")
    parser.add_argument("--save_normal", default=True,
                        help="Save intermediate normal map")
    
    # Post-processing
    parser.add_argument("--disable_postprocessing", action="store_true",
                        help="Disable mesh post-processing (enabled by default)")
    parser.add_argument("--target_faces", type=int, default=200_000,
                        help="Target number of faces for post-processed mesh (default: 300,000)")
    
    # Multiview
    parser.add_argument("--multiview_inference", action="store_true",
                        help="Enable multiview inference mode")
    parser.add_argument("--multiview_mode", type=str, default="multidiffusion",
                        choices=["stochastic", "multidiffusion"],
                        help="Multiview fusion mode (default: multidiffusion)")

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input_dir")
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input_dir")
    if args.input and not args.output:
        parser.error("Must specify --output when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("Must specify --output_dir when using --input_dir")


    # Call main with parsed arguments
    main(args.input, args.output, args.input_dir, args.output_dir, args.model_path,
         args.nirne_weights_dir, args.device,
         args.preprocess_resolution, args.normal_resolution, args.seed, args.output_format,
         args.ss_guidance_strength, args.ss_sampling_steps, args.slat_guidance_strength, args.slat_sampling_steps,
         args.skip_preprocess, args.skip_normal, args.save_normal,
         not args.disable_postprocessing, args.target_faces,
         args.multiview_inference, args.multiview_mode)