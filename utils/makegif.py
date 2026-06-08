import os
import sys
from PIL import Image

def main():
    """
    Utility script to quickly stitch a folder of images into a GIF.
    """
    if len(sys.argv) < 3:
        print("Usage: python make_gif.py <input_directory> <output_filename.gif> [fps]")
        print("Example: python make_gif.py ../part1/data/capture_output ../assets/new_movie.gif 20")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    # Default to 20 frames per second if not specified
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 20 
    duration_ms = int(1000 / fps)

    # Grab all images and sort them alphabetically/numerically
    image_files = sorted([
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Generating GIF at {fps} FPS...")
    
    # Load all images into memory
    frames = [Image.open(os.path.join(input_dir, f)) for f in image_files]
    
    # Stitch and save
    frames[0].save(
        output_file,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0 # 0 means loop infinitely
    )
    
    print(f"Success! GIF saved to {output_file}")

if __name__ == "__main__":
    main()