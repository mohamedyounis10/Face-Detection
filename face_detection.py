from skimage import io, color
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Methods
# Erosion
def erosion(image, kernel_size):
    height, width = image.shape
    
    # Create a copy of the image to store the eroded result
    eroded_image = np.zeros_like(image, dtype=np.uint8)
    
    # Calculate the offset (this is the size of half the kernel)
    offset = kernel_size // 2

    # Loop over the image pixels (excluding the border area)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            
            region = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            
            # Check if the kernel fits (i.e., all values in the region are 255)
            if np.all(region == 255):  # If all pixels in the region are white (skin)
                eroded_image[i, j] = 255  # Set the central pixel to 255
            else:
                eroded_image[i, j] = 0  # Otherwise, set the central pixel to 0
    
    return eroded_image

# Dilate
def dilate(mask, kernel_size=3):
    dilated_mask = np.copy(mask)
    
    k = kernel_size // 2
    
    for i in range(k, mask.shape[0] - k):
        for j in range(k, mask.shape[1] - k):
            # Get the region of the kernel around each pixel
            region = mask[i-k:i+k+1, j-k:j+k+1]
            if np.sum(region) > 0:
                dilated_mask[i, j] = 255
            else:
                dilated_mask[i, j] = 0
    return dilated_mask

# XOR 
def xor_operation(edges_thresholded, dilated_mask):
  # Ensure inputs are boolean arrays. This handles cases where they might be 0/1 integers.
  edges_thresholded = edges_thresholded.astype(bool)
  dilated_mask = dilated_mask.astype(bool)

  if edges_thresholded.shape != dilated_mask.shape:
      print("Error: Input arrays must have the same shape for XOR operation.")
      return None  # Or raise an exception if you prefer

  xor_result = np.zeros_like(edges_thresholded, dtype=np.uint8) # Pre-allocate for efficiency

  for i in range(edges_thresholded.shape[0]):
    for j in range(edges_thresholded.shape[1]): # Assuming 2D arrays. Adapt for higher dimensions
        xor_result[i, j] = 1 if edges_thresholded[i, j] ^ dilated_mask[i, j] else 0

  return xor_result
   
# Load the image
imagepath = r"C:\Users\moham\Desktop\Face Detection\1.webp"
my_image = io.imread(imagepath)
hsv_image = color.rgb2hsv(my_image) * 255
hsv_image = hsv_image.astype(np.uint8)

# Display the original image
plt.figure('Original Image')
plt.imshow(my_image)
plt.axis('off')
plt.show()

# Define lower and upper limits for skin color in HSV
lower_skin = np.array([0, 20], dtype=np.uint8)
upper_skin = np.array([20, 255], dtype=np.uint8)

# Initialize an empty mask with the same size as the image
skin_mask = np.zeros((hsv_image.shape[0], hsv_image.shape[1]))

# Loop through the image pixels
for i in range(hsv_image.shape[0]):
    for j in range(hsv_image.shape[1]):
        h, s, v = hsv_image[i, j]  # Get the H, S, V values of the pixel
        # Check if the pixel is within the skin color range
        if (lower_skin[0] <= h <= upper_skin[0]) and (lower_skin[1] <= s <= upper_skin[1]):
            skin_mask[i, j] = 255  # Set to 255 (white) for skin region
        else:
            skin_mask[i, j] = 0  # Set to 0 (black) for non-skin region

# Display the skin mask
plt.figure('Skin Mask')
plt.imshow(skin_mask, cmap='gray')
plt.axis('off')
plt.show()

# Perform erosion
kernel_size=13
eroded_mask = erosion(skin_mask, kernel_size)

# Display the eroded skin mask
plt.figure('Eroded Skin Mask')
plt.imshow(eroded_mask, cmap='gray')
plt.axis('off')
plt.show()

# Perform dilation
dilated_mask = dilate(eroded_mask, kernel_size=19)

# Display the dilated mask
plt.figure('Dilated Skin Mask')
plt.imshow(dilated_mask, cmap='gray')
plt.axis('off')
plt.show()

# Create a copy of the original image to work on
image_copy = my_image.copy()

# Now you can modify the copy without affecting the original image
image_copy[dilated_mask == 0] = 0

# Display the image after mask
plt.figure('My Image (After)')
plt.imshow(image_copy)
plt.axis('off')
plt.show()

# Apply Canny edge detection
edges = cv2.Canny(cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY), 100, 200)

# Display the edges
plt.figure('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# Increase edges using dilation
kernel = np.ones((3, 3), dtype=np.uint8)
edges = dilate(edges,3)

# Display the increased edges
plt.figure('Edge Detection (Increased)')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# Apply thresholding
threshold = 0.15
edges_thresholded = np.where(edges >= threshold, 1, 0)

# Display the thresholded edges
plt.figure('Thresholded Edges')
plt.imshow(edges_thresholded, cmap='gray')
plt.axis('off')
plt.show()
 
xor_result = xor_operation(edges, dilated_mask)

# Display the XOR result
plt.figure('XOR Result')
plt.imshow(xor_result, cmap='gray')
plt.axis('off')
plt.show()

# Find connected components
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(xor_result)

filtered_faces = []
hist_average_ratio = []
hist_average_ratio2 = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]

    aspect_ratio = w / float(h)
    hist_average_ratio.append(aspect_ratio)

    if (0.64 < aspect_ratio < 1.2  and w > 25 and h > 25 and area > 1000):
        hist_average_ratio2.append(aspect_ratio)
        filtered_faces.append((x, y, w, h))

for (x, y, w, h) in filtered_faces:
    cv2.rectangle(my_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

plt.hist(hist_average_ratio, bins=100)
plt.title('Aspect Ratios')
plt.show()

plt.hist(hist_average_ratio2, bins=100)
plt.title('Aspect Ratios')
plt.show()

# Display result
print('Number of Faces: ', len(filtered_faces))

plt.figure('Face Detection')
plt.imshow(my_image, cmap='gray')
plt.axis('off')
plt.show()