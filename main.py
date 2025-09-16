import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image # For image processing
import os


def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    print(f"Loaded image with shape: {img_array.shape}")
    return img_array

# crop image
def crop_image(img_array, top, bottom, left, right):
   cropped_img = img_array[top:bottom, left:right] 
   return cropped_img

# resize function after crop_image
def resize_image(img_array, new_size=(200, 200)):
    """
    Resize image using PIL
    
    Explanation:
    - Convert NumPy array back to PIL Image
    - Use PIL's resize method (handles interpolation automatically)
    - Convert back to NumPy array for consistency
    """
    img_pil = Image.fromarray(img_array)
    resized_pil = img_pil.resize(new_size)
    resized_array = np.array(resized_pil)
    print(f"📏 Resized from {img_array.shape} to {resized_array.shape}")
    return resized_array

# Horizontal and Vertical Flip
def flip_horizontal(img_array):
    flipped = np.fliplr(img_array)
    return flipped

def flip_vertical(img_array):
    flipped = np.flipud(img_array)
    return flipped

# Change Brightness
def change_brightness(img_array, brightness_factor):
    
    """
      Broadcasting: adding a single number to entire array
    - Each pixel value gets increased by brightness_factor
    - np.clip() ensures values stay in valid range [0, 255]
    - astype() maintains proper data type for images
    """
    bright_img = img_array.astype(np.float32) + brightness_factor 
    # Clip values to valid pixel range and convert back to uint8
    bright_img = np.clip(bright_img, 0, 255).astype(np.uint8)
    
    print(f"💡 After brightness +{brightness_factor}: {bright_img.min()} to {bright_img.max()}")
    return bright_img


img_array = load_image('./images/G-wagon.jpg')
# Perform manipulations
cropped_img = crop_image(img_array, 50,800, 60, 1180)
resized_img = resize_image(img_array, new_size=(300, 300))
horiz_flip = flip_horizontal(img_array)
vert_flip = flip_vertical(img_array)
brighten_image = change_brightness(img_array, 100)


# Display original and processed images
fig, axes = plt.subplots(2, 3, figsize=(10, 6)) # 2 rows, 3 columns

plt.suptitle('Image Manipulation with NumPy & Matplotlib', 
             fontsize=16, fontweight='bold', y=0.98)
# print("fig:",fig)
# print("axes:",axes)
axes[0, 0].imshow(img_array)
axes[0, 0].set_title('Original Image') 

axes[0, 1].imshow(cropped_img)
axes[0, 1].set_title('Cropped Image')

axes[0, 2].imshow(resized_img)
axes[0, 2].set_title('Resized Image')

axes[1, 0].imshow(horiz_flip)
axes[1, 0].set_title('Horizontal Flip')

axes[1, 1].imshow(vert_flip)
axes[1, 1].set_title('Vertical Flip')

axes[1, 2].imshow(brighten_image)
axes[1, 2].set_title('Brightened Image')

for ax in axes.flatten():
    ax.axis('off')  # Hide axes ticks
plt.tight_layout()  # automatically adjusts subplot spacing to prevent overlapping and make the layout look neat.
plt.show()
