import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import numpy as np
import matplotlib.pyplot as plt

# 1. تفعيل Mixed Precision إذا كان مدعوماً
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 2. تحميل وتحسين إدارة البيانات باستخدام tf.data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(50000).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .batch(64).prefetch(tf.data.AUTOTUNE)

# 3. أسماء الفئات
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 4. إنشاء النموذج باستخدام SeparableConv2D
def create_model():
    model = models.Sequential([
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', dtype='float32')  # التأكد من استخدام float32 عند mixed precision
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# 5. إنشاء النموذج
model = create_model()

# 6. إعدادات التدريب والتوقف المبكر
callbacks_list = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

# 7. تدريب النموذج باستخدام tf.data
history = model.fit(train_ds, epochs=30, validation_data=test_ds, callbacks=callbacks_list)

# 8. تقييم النموذج
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f'\nTest Accuracy: {test_acc:.4f}')
