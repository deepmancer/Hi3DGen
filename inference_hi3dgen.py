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


def main():
    parser = argparse.ArgumentParser(description="Hi3DGen 3D Mesh Generation Inference")
    
    # Input/Output arguments
    parser.add_argument("--input", type=str, help="Path to input image file")
    parser.add_argument("--output", type=str, help="Path to output mesh file")
    parser.add_argument("--input_dir", type=str, help="Path to input directory containing images")
    parser.add_argument("--output_dir", type=str, help="Path to output directory for meshes")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="weights/trellis-normal-v0-1",
                        help="Path to Hi3DGen model weights (default: weights/trellis-normal-v0-1)")
    parser.add_argument("--nirne_weights_dir", type=str, default="./weights",
                        help="Directory to cache NiRNE weights (default: ./weights)")
    parser.add_argument("--normal_estimator", type=str, default="nirne",
                        choices=["nirne", "genpercept"],
                        help="Normal estimation model to use (default: genpercept)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run inference on")
    
    # Processing arguments
    parser.add_argument("--preprocess_resolution", type=int, default=1024,
                        help="Resolution for image preprocessing (default: 1024)")
    parser.add_argument("--normal_resolution", type=int, default=1024,
                        help="Resolution for normal estimation (default: 768)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_format", type=str, default="glb",
                        choices=["glb", "obj", "ply", "stl"],
                        help="Output mesh format (default: glb)")
    
    # Stage 1: Sparse Structure Generation
    parser.add_argument("--ss_guidance_strength", type=float, default=3.0,
                        help="Guidance strength for sparse structure generation (default: 3.0)")
    parser.add_argument("--ss_sampling_steps", type=int, default=50,
                        help="Sampling steps for sparse structure generation (default: 50)")
    
    # Stage 2: Structured Latent Generation
    parser.add_argument("--slat_guidance_strength", type=float, default=3.0,
                        help="Guidance strength for structured latent generation (default: 3.0)")
    parser.add_argument("--slat_sampling_steps", type=int, default=6,
                        help="Sampling steps for structured latent generation (default: 6)")
    
    # Options
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip image preprocessing (use if image is already preprocessed)")
    parser.add_argument("--skip_normal", action="store_true",
                        help="Skip normal estimation (use if input is already a normal map)")
    parser.add_argument("--save_normal", action="store_true",
                        help="Save intermediate normal map")
    
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
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load Hi3DGen pipeline
    print("Loading Hi3DGen pipeline...")
    try:
        hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained(args.model_path)
        if device.type == "cuda":
            hi3dgen_pipeline.cuda()
        print("Hi3DGen pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading Hi3DGen pipeline: {e}")
        return
    
    # Load normal predictor (unless skipping)
    normal_predictor = None
    if not args.skip_normal:
        if args.normal_estimator == "nirne":
            print("Loading NiRNE normal predictor...")
            try:
                # Import NiRNE components
                import sys
                nirne_path = os.path.join(os.path.dirname(__file__), 'NiRNE')
                if nirne_path not in sys.path:
                    sys.path.insert(0, nirne_path)
                
                from nirne.pipeline_yoso_normal import YOSONormalsPipeline
                from nirne.pipeline_nirne import NiRNEPipeline
                from nirne.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
                
                # Load from local weights directory
                yoso_weight_path = os.path.join(args.nirne_weights_dir, 'NiRNE')
                
                # Check if weights exist, if not provide helpful error
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
                ).to(device)
                
                # Create NiRNE Predictor wrapper
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
                        
                        # Handle RGBA images
                        orig_size = img.size
                        if img.mode == 'RGBA':
                            rgb = img.convert('RGB')
                            alpha = img.split()[-1]
                            white_bg = Image.new('RGB', img.size, (255, 255, 255))
                            img = Image.composite(rgb, white_bg, alpha)
                        
                        # Resize image
                        input_image_np = np.asarray(img)
                        H, W = float(input_image_np.shape[0]), float(input_image_np.shape[1])
                        k = float(resolution) / max(H, W)
                        new_H = int(np.round(H * k / 64.0)) * 64
                        new_W = int(np.round(W * k / 64.0)) * 64
                        img = img.resize((new_W, new_H), Image.Resampling.LANCZOS)
                        
                        # Generate normal map
                        pipe_kwargs = {}
                        if 'num_inference_steps' in kwargs and kwargs['num_inference_steps'] is not None:
                            pipe_kwargs['num_inference_steps'] = kwargs['num_inference_steps']
                        
                        match_input_resolution = kwargs.get('match_input_resolution', True)
                        pipe_out = self.model(img, match_input_resolution=match_input_resolution, **pipe_kwargs)
                        
                        # Apply mask and convert to image
                        prediction = pipe_out.prediction[0]
                        normal_map = (prediction.clip(-1, 1) + 1) / 2
                        normal_map = (normal_map * 255).astype(np.uint8)
                        normal_map = Image.fromarray(normal_map)
                        
                        # Resize back to original dimensions if needed
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
        
        elif args.normal_estimator == "genpercept":
            print("Loading GenPercept normal predictor...")
            try:
                # Import GenPercept components
                import sys
                genpercept_path = os.path.join(os.path.dirname(__file__), 'genpercept')
                if genpercept_path not in sys.path:
                    sys.path.insert(0, genpercept_path)
                
                from genpercept.compute_normal import GenPerceptNormalPipeline
                
                # Create GenPercept pipeline
                genpercept_pipeline = GenPerceptNormalPipeline(device=device)
                
                # Create GenPercept Predictor wrapper
                class GenPerceptPredictor:
                    def __init__(self, pipeline):
                        self.pipeline = pipeline
                    
                    def to(self, device):
                        self.pipeline.device = torch.device(device)
                        self.pipeline.pipe.to(device)
                        return self
                    
                    @torch.no_grad()
                    def __call__(self, img, resolution=1024, **kwargs):
                        # GenPercept compute_normal handles RGBA and resolution internally
                        normal_map = self.pipeline.compute_normal(img, processing_resolution=resolution)
                        return normal_map
                
                normal_predictor = GenPerceptPredictor(genpercept_pipeline)
                print("GenPercept predictor loaded successfully!")
            except Exception as e:
                print(f"Error loading GenPercept predictor: {e}")
                import traceback
                traceback.print_exc()
                return
    
    def process_single_image(input_path, output_path):
        """Process a single image and generate 3D mesh"""
        print(f"\nProcessing: {input_path}")
        
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            return False
        
        try:
            # Load input image
            image = Image.open(input_path)
            
            # Preprocess image (background removal, etc.)
            if not args.skip_preprocess:
                print("  Preprocessing image...")
                image = hi3dgen_pipeline.preprocess_image(image, resolution=args.preprocess_resolution)
            
            # Generate normal map
            if args.skip_normal:
                print("  Using input as normal map...")
                normal_image = image
            else:
                print(f"  Generating normal map using {args.normal_estimator}...")
                with torch.no_grad():
                    normal_image = normal_predictor(
                        image,
                        resolution=args.normal_resolution,
                        match_input_resolution=True,
                        data_type='object'
                    )
            
            # Save normal map if requested
            if args.save_normal and not args.skip_normal:
                normal_path = os.path.splitext(output_path)[0] + "_normal.png"
                os.makedirs(os.path.dirname(normal_path) or ".", exist_ok=True)
                normal_image.save(normal_path)
                print(f"  Normal map saved to: {normal_path}")
            
            # Generate 3D mesh
            print("  Generating 3D mesh...")
            with torch.no_grad():
                outputs = hi3dgen_pipeline.run(
                    normal_image,
                    seed=args.seed,
                    formats=["mesh"],
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": args.ss_sampling_steps,
                        "cfg_strength": args.ss_guidance_strength,
                    },
                    slat_sampler_params={
                        "steps": args.slat_sampling_steps,
                        "cfg_strength": args.slat_guidance_strength,
                    },
                )
            
            generated_mesh = outputs['mesh'][0]
            
            # Export mesh
            print("  Exporting mesh...")
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # Convert to trimesh and export
            trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)
            trimesh_mesh.export(output_path)
            
            print(f"  Mesh saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"  Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Process single image
    if args.input:
        success = process_single_image(args.input, args.output)
        if success:
            print("\n✓ Processing complete!")
        else:
            print("\n✗ Processing failed!")
    
    # Process directory of images
    elif args.input_dir:
        print(f"Processing images from directory: {args.input_dir}")
        
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            return
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        if len(image_paths) == 0:
            print(f"No images found in {args.input_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process each image
        success_count = 0
        for image_path in tqdm(image_paths, desc="Processing images"):
            # Construct output path maintaining subdirectory structure
            rel_path = os.path.relpath(image_path, args.input_dir)
            output_path = os.path.join(args.output_dir, rel_path)
            
            # Change extension to output format
            output_path = os.path.splitext(output_path)[0] + f".{args.output_format}"
            
            if process_single_image(image_path, output_path):
                success_count += 1
        
        print(f"\n✓ Processing complete! {success_count}/{len(image_paths)} meshes generated successfully")
        print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
