import os
import time
import numpy as np
import tensorflow as tf

DATA_DIR = r"E:\disease\data\PlantVillage"
MODEL_DIR = r"E:\disease\models"
IMG_SIZE = 224
BATCH_SIZE = 1

model_paths = {
    "FP32": os.path.join(MODEL_DIR, "plant_disease_fp32.tflite"),
    "FP16": os.path.join(MODEL_DIR, "plant_disease_fp16.tflite"),
    "INT8": os.path.join(MODEL_DIR, "plant_disease_int8.tflite"),
}

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

def evaluate_tflite(model_path, model_name):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    times = []

    input_dtype = input_details[0]["dtype"]
    input_scale, input_zero_point = input_details[0]["quantization"]

    output_dtype = output_details[0]["dtype"]
    output_scale, output_zero_point = output_details[0]["quantization"]

    for images, labels in test_ds:
        image = images.numpy().astype(np.float32) / 255.0

        if input_dtype == np.uint8:
            image = image / input_scale + input_zero_point
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(input_dtype)

        start = time.time()
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        end = time.time()

        output = interpreter.get_tensor(output_details[0]["index"])

        if output_dtype == np.uint8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        pred = np.argmax(output[0])
        label = int(labels.numpy()[0])

        if pred == label:
            correct += 1
        total += 1
        times.append((end - start) * 1000)

    acc = correct / total
    avg_time = np.mean(times)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n[{model_name}]")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Accuracy: {acc:.4f}")
    print(f"Average inference time: {avg_time:.2f} ms")

for name, path in model_paths.items():
    if os.path.exists(path):
        evaluate_tflite(path, name)
    else:
        print(f"Missing model: {path}")