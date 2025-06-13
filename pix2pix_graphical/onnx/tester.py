import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def test_model(model_path: str, image_path: str) -> None:
    model_path = Path(model_path)
    session = ort.InferenceSession(model_path.as_posix())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) * 2.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    output = session.run([output_name], {input_name: image})[0]

    output_img = output.squeeze().transpose(1, 2, 0)
    output_img = (output_img + 1) / 2.0
    output_img = np.clip(output_img, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(output_img)
    plt.title("ONNX Model Output")
    plt.axis("off")
    plt.show()
