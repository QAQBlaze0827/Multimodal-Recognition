from __future__ import annotations

import argparse
from pathlib import Path


EMOTIONS = ("neutral", "happy", "sad", "anger")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small video emotion model and export ONNX.")
    parser.add_argument("--data-dir", default="data/datasets/fer")
    parser.add_argument("--output", default="models/mini_xception_int8.onnx")
    parser.add_argument("--fp32-output", default="models/mini_xception_fp32.onnx")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Dataset must contain train/ and val/ folders. "
            "See TRAINING.md for the expected folder layout."
        )

    import tensorflow as tf
    import tf2onnx

    train_ds = make_dataset(tf, train_dir, args.batch_size, shuffle=True)
    val_ds = make_dataset(tf, val_dir, args.batch_size, shuffle=False)

    model = build_model(tf, learning_rate=args.learning_rate)
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    fp32_output = Path(args.fp32_output)
    fp32_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    spec = (tf.TensorSpec((None, 1, 48, 48), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(fp32_output))
    quantize_onnx(fp32_output, output)
    print(f"[train] exported FP32 ONNX: {fp32_output}")
    print(f"[train] exported int8 ONNX: {output}")


def make_dataset(tf, folder: Path, batch_size: int, shuffle: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="categorical",
        class_names=list(EMOTIONS),
        color_mode="grayscale",
        image_size=(48, 48),
        batch_size=batch_size,
        shuffle=shuffle,
    )

    def normalize(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = tf.transpose(images, [0, 3, 1, 2])
        return images, labels

    return ds.map(normalize).prefetch(tf.data.AUTOTUNE)


def build_model(tf, learning_rate: float):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(1, 48, 48), name="input")
    x = layers.Permute((2, 3, 1))(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for filters in (32, 64, 128, 256):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def quantize_onnx(fp32_output: Path, int8_output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_output),
        model_output=str(int8_output),
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
