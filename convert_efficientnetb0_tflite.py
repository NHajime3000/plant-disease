import os
import tensorflow as tf

MODEL_DIR = r"E:\disease\models"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnetb0_baseline.keras")

print("TensorFlow version:", tf.__version__)
print("Loading model:", KERAS_MODEL_PATH)

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# ======================
# FP32 TFLite
# ======================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()

fp32_path = os.path.join(MODEL_DIR, "efficientnetb0_fp32.tflite")
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)

print("Saved FP32 TFLite:", fp32_path)

# ======================
# FP16 TFLite
# ======================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16 = converter.convert()

fp16_path = os.path.join(MODEL_DIR, "efficientnetb0_fp16.tflite")
with open(fp16_path, "wb") as f:
    f.write(tflite_fp16)

print("Saved FP16 TFLite:", fp16_path)

# ======================
# Show file sizes
# ======================
for path in [fp32_path, fp16_path]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"{os.path.basename(path)} size: {size_mb:.2f} MB")