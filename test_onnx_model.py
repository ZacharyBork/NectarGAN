import onnxruntime as ort
import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt

# Load ONNX model
model_path = pathlib.Path('E:/ML/pytorch-CycleGAN-and-pix2pix/checkpoints/h2t_v23/h2t_v23_epochlatest.onnx')
session = ort.InferenceSession(model_path.as_posix())

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load and preprocess image
image_path = pathlib.Path('E:/ML/pytorch-CycleGAN-and-pix2pix/results/h2t_v23/test_01/images/1457_3_0_real_A.png')
image = cv2.imread(image_path.as_posix())  # BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (256, 256))  # Match model input size
image = image.astype(np.float32) / 255.0  # Normalize to 0–1
image = (image - 0.5) * 2.0
image = np.transpose(image, (2, 0, 1))  # HWC → CHW
image = np.expand_dims(image, axis=0)  # Add batch dimension → (1, 3, H, W)


# Run inference
output = session.run([output_name], {input_name: image})[0]

# Postprocess output for visualization
output_img = output.squeeze().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
output_img = np.clip(output_img, 0, 1)  # Ensure it's in [0, 1] range
if output_img.dtype == np.uint8:
    output_img = output_img.astype(np.float32) / 255.0

# Display with matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(output_img)
plt.title("ONNX Model Output")
plt.axis("off")
plt.show()