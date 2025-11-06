#!/usr/bin/env python3
"""
Download NiRNE weights from Hugging Face
"""

import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download NiRNE weights")
    parser.add_argument("--weights_dir", type=str, default="./weights",
                        help="Directory to save weights (default: ./weights)")
    args = parser.parse_args()
    
    # The model is hosted on Hugging Face as yoso-normal weights
    model_id = "Stable-X/yoso-normal-v1-8-1"
    output_dir = os.path.join(args.weights_dir, "NiRNE")
    
    print(f"Downloading NiRNE weights from Hugging Face...")
    print(f"  Model: {model_id}")
    print(f"  Destination: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the model from Hugging Face
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            force_download=False,
            resume_download=True
        )
        
        print(f"\n✓ Weights downloaded successfully to {output_dir}")
        print(f"  You can now run inference with --nirne_weights_dir {args.weights_dir}")
    except Exception as e:
        print(f"\n✗ Error downloading weights: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
