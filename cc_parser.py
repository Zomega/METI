from PIL import Image
import numpy as np
import os

def recover_cosmic_call_bits(image_path, scale_factor=8):
    """
    Loads an 8x scaled Cosmic Call image, sections it into original 1x pixels, 
    and recovers the original bit sequence.

    Args:
        image_path (str): Path to the 8x scale PNG image.
        scale_factor (int): The scaling factor (e.g., 8 for 8x scaled images).

    Returns:
        tuple: (list of int, tuple of int). The list contains the recovered 
               bit sequence (0 or 1), and the tuple contains the original 
               grid dimensions (width, height).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    try:
        # 1. Load the Image
        # Use 'L' mode (8-bit pixels, grayscale) to ensure we're dealing with
        # a single color channel.
        img = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Get the dimensions of the scaled image
    scaled_width, scaled_height = img.size

    # Calculate the dimensions of the original grid
    if scaled_width % scale_factor != 0 or scaled_height % scale_factor != 0:
        print(f"Error: Image dimensions ({scaled_width}x{scaled_height}) are not "
              f"evenly divisible by the scale factor {scale_factor}.")
        return None, None
        
    original_width = scaled_width // scale_factor
    original_height = scaled_height // scale_factor
    original_dims = (original_width, original_height)
    print(f"Loaded image: {image_path}")
    print(f"Scaled dimensions: {scaled_width}x{scaled_height}")
    print(f"Original grid dimensions: {original_width}x{original_height}")

    # Convert the image to a NumPy array for easier slicing
    np_img = np.array(img)

    recovered_bits = []

    # 2. Section the Image into Original Pixels
    # Iterate over the original grid coordinates (x, y)
    for y in range(original_height):
        for x in range(original_width):
            # Define the top-left corner of the 8x8 block for the current original pixel
            start_x = x * scale_factor
            start_y = y * scale_factor

            # Extract a single 8x8 block (representing one original pixel)
            # The slices are [row_start:row_end, col_start:col_end]
            pixel_block = np_img[start_y : start_y + scale_factor, 
                                 start_x : start_x + scale_factor]

            # 3. Recover the Original Bit
            # In the Cosmic Call images, the "pixels" are either fully white or fully black.
            # We can check the average value of the block to determine the bit:
            # - White (high value, e.g., 255) -> 0
            # - Black (low value, e.g., 0) -> 1
            # We assume a clear separation, e.g., any value <= 127 is "black" (1).

            # Get the mean pixel value of the 8x8 block
            mean_value = np.mean(pixel_block)

            # Assign bit: Black (low value) is typically 1 (active/signal), White (high value) is 0 (background)
            # Thresholding at the midpoint of 8-bit grayscale (255 / 2 = 127.5)
            bit = 1 if mean_value < 128 else 0
            
            recovered_bits.append(bit)

    print(f"Successfully recovered {len(recovered_bits)} bits.")
    return recovered_bits, original_dims

# --- Usage Example ---

# NOTE: Replace 'path/to/your/image.png' with the actual path to one of your Cosmic Call images.
image_file = 'CC1/104.png' 

# Run the function
bit_sequence, dimensions = recover_cosmic_call_bits(image_file, scale_factor=8)

if bit_sequence:
    # Print a sample of the recovered bits
    print("\n--- Sample of Recovered Bits ---")
    print(bit_sequence[:dimensions[0] * 3]) # Print first 3 rows
    
    # Optional: Reshape the flat sequence back into the 2D grid
    grid_array = np.array(bit_sequence).reshape(dimensions)
    print("\n--- Reshaped Grid (First 5x5) ---")
    print(grid_array[:5, :5])
    
    # The final flat sequence of bits is in 'bit_sequence'