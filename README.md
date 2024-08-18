# Depth Information Estimation

This project focuses on estimating 3D depth information from two images of the same object taken from different angles (left and right). The depth estimation is achieved through pixel-wise matching and window-based matching techniques.

## Features

- **Pixel-wise Matching:** Performs detailed comparison of corresponding pixels between the two images to estimate depth.
- **Window-based Matching:** Uses sliding windows to compare small regions of the images, improving robustness to noise and texture variations.
- **3D Depth Estimation:** Generates a depth map representing the distance of each point in the scene from the camera.

## Prerequisites

Ensure you have Python installed (preferably Python 3.12 or above).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Duy1230/Project-Module2-DepthEstimation.git
   cd Project-Module2-DepthEstimation
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run app.y
    ```
