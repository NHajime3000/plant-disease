import os
import tensorflow as tf

MODEL_DIR = r"E:\disease\models"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_mobilenetv2_baseline.keras")
DATA_DIR = r"E:\disease\data\PlantVillage"

IMG_SIZE = 224
BATCH_SIZE = 1

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

def representative_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    for images, _ in ds.take(100):
        images = normalization_layer(images)
        yield [tf.cast(images, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_int8 = converter.convert()

int8_path = os.path.join(MODEL_DIR, "plant_disease_int8.tflite")
with open(int8_path, "wb") as f:
    f.write(tflite_int8)

print("Saved INT8 TFLite:", int8_path)
print("INT8 model size:", os.path.getsize(int8_path) / (1024 * 1024), "MB")