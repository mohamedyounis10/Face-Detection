# Face Detection Project

## Overview
This project implements a face detection system using custom image processing techniques. The goal is to detect faces in images by processing pixel-level data and applying various image manipulation methods. Key techniques include:

- Skin detection in HSV color space.
- Morphological operations (erosion and dilation).
- Edge detection using the Canny algorithm.
- Logical operations (XOR) for refining detected regions.
- Filtering based on dynamic thresholds for aspect ratio and size of the detected regions.

The system processes images to detect faces and highlights them with bounding boxes. Intermediate results such as masks, edges, and histograms of aspect ratios are also visualized for analysis.

---

## Features
- **Pixel-level Face Detection**: Detect faces based on custom processing techniques.
- **Morphological Transformations**: Uses custom kernels for erosion and dilation to refine the skin mask.
- **Edge Detection**: Applies the Canny algorithm to enhance image boundaries.
- **XOR Operation**: Refines detected face regions through XOR logical operations.
- **Geometric Filtering**: Filters regions based on dynamic aspect ratio and size thresholds.
- **Visualization**: Displays intermediate steps such as masks, edges, and histograms of aspect ratios.

---

## Images

### 1. **Presentation Slide Image**  

![Image Processing Presentation](https://github.com/user-attachments/assets/2ba06801-28b2-432d-bd49-fa171c741867)

---

### 2. **Example Outputs**  
After running the face detection script, various intermediate results will be produced. Below are the outputs of the process:

- **Original Image with Detected Faces**:  
  The original image with faces highlighted by bounding boxes.

  ![Original Image with Detected Faces](https://github.com/user-attachments/assets/1e5f2ab8-a063-4690-8f8a-e8e15e91a6f4)
---

## Requirements
To run this project, you'll need Python 3.7 or higher and the following dependencies:

- `numpy`
- `opencv-python`
- `scikit-image`
- `matplotlib`

To install these dependencies, run the following command:

```bash
pip install numpy opencv-python scikit-image matplotlib
```

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set the image path**:  
   Replace the image path in the code with the path to the image you want to process. Update the variable `imagepath` in the script:
   ```python
   imagepath = "./images/sample_image.webp"
   ```
   Ensure that the `images` folder contains your input images.

3. **Run the script**:  
   Run the Python script that processes the image:
   ```bash
   python face_detection.py
   ```

4. **View the results**:  
   After running the script, the following outputs will be displayed:
   - The original image with bounding boxes around detected faces.
   - A skin detection mask.
   - Eroded and dilated masks.
   - Canny edge detection result.
   - Refined face detection after XOR operation.
   - Histograms of aspect ratios of the detected regions.

---

## Outputs

The program will produce the following outputs:

1. **Original Image**: Displays the input image.
2. **Skin Mask**: A binary mask that highlights the skin regions in the image.
3. **Eroded and Dilated Masks**: Refined masks after applying morphological transformations.
4. **Edges**: Detected edges in the image using the Canny algorithm.
5. **XOR Result**: A refined mask after applying the XOR logical operation to eliminate false positives.
6. **Histograms**: Visualizations of aspect ratios of the detected regions to aid in filtering.
7. **Final Image**: The original image with bounding boxes around detected faces.

---

## Modifications

### Dropping Static Dependence
- The previous static threshold for filtering faces based on aspect ratio was replaced with dynamic thresholds. These thresholds are calculated based on band statistics from the detected regions' aspect ratios.
  ```python
  mean_aspect_ratio = np.mean(hist_average_ratio)
  std_aspect_ratio = np.std(hist_average_ratio)
  min_band = mean_aspect_ratio - std_aspect_ratio
  max_band = mean_aspect_ratio + std_aspect_ratio

  for i in range(1, num_labels):
      x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
      aspect_ratio = w / float(h)

      if (min_band < aspect_ratio < max_band and w > 25 and h > 25 and area > 1000):
          filtered_faces.append((x, y, w, h))
  ```

This change allows for better dynamic filtering of detected regions based on their geometric properties.

### Additional Notes
- Input images should contain visible faces and minimal background noise for optimal performance.
- You can adjust parameters such as kernel size and threshold values to fine-tune detection for different datasets or image conditions.
- For best results, ensure the images have good lighting and minimal occlusions (e.g., no overlapping faces).

---

## Limitations
- The method may not perform well in the following cases:
  - Complex backgrounds with no clear distinction between the face and background.
  - Low-quality images or images with poor lighting conditions.
  - Overlapping or occluded faces.
  - Images with non-human faces (e.g., animal faces).

---

## Future Enhancements
1. **Machine Learning Integration**: Replace static skin and geometric feature thresholds with machine learning models for better accuracy in detecting faces under varied conditions.
2. **Real-Time Detection**: Integrate with webcam or real-time video feed for live face detection.
3. **GUI for Visualization**: Create a graphical user interface (GUI) to simplify interaction and visualization of detection results.
4. **Optimizations for Speed**: Enhance performance for real-time processing and large-scale datasets.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the open-source contributors and the developers of Python libraries (e.g., OpenCV, NumPy, Scikit-image, Matplotlib) who made this project possible. Your contributions are greatly appreciated!

---
