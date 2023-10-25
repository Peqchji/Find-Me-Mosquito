import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
import optuna
from optuna.integration import TFKerasPruningCallback

# Constants
img_size = 224
epochs = 20
input_shape = (img_size, img_size, 3)

# Data Loading
img_path = os.path.join(os.getcwd(), "split_data")
train_set = tf.keras.utils.image_dataset_from_directory(
    img_path + "/train",
    image_size=(img_size, img_size),
    shuffle=True,
    label_mode='categorical',
    subset="training",
    validation_split=0.3,
    seed=42
)

test_set = tf.keras.utils.image_dataset_from_directory(
    img_path + "/test",
    image_size=(img_size, img_size),
    shuffle=False,
    label_mode='categorical',
    seed=42
)

val_set = tf.keras.utils.image_dataset_from_directory(
    img_path + "/val",
    image_size=(img_size, img_size),
    label_mode='categorical',
    shuffle=True,
    subset="validation",
    validation_split=0.3,
    seed=42
)

# Model Creation
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

class_amount = len(train_set.class_names)

output = Dense(class_amount, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Define the callbacks outside of the objective function
callbacks = [
    ModelCheckpoint("save_at_{epoch}.keras"),
    EarlyStopping(monitor="val_loss", patience=5),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
]

# Define the objective function for hyperparameter tuning
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    batch_size = int(trial.suggest_categorical("batch_size", [16, 32, 64]))  # Convert batch_size to int

    # Model Compilation with Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,  # Use the updated optimizer
        metrics=["accuracy"]
    )

    # Data Prefetching
    train_ds = train_set.prefetch(tf.data.AUTOTUNE)
    val_ds = val_set.prefetch(tf.data.AUTOTUNE)

    # Add the TFKerasPruningCallback to the list of callbacks
    pruning_callback = TFKerasPruningCallback(trial, "val_loss")
    trial_callbacks = callbacks + [pruning_callback]

    # Model Training
    history = model.fit(
        train_ds,
        epochs=epochs,
        verbose=1,
        callbacks=trial_callbacks,
        validation_data=val_ds,
        batch_size=batch_size
    )

    # Model Evaluation
    test_loss, test_accuracy = model.evaluate(test_set, verbose=1)

    # F1 Score Calculation
    true_labels = []
    predicted_labels = []

    for images, labels in test_set:
        true_labels.extend(np.argmax(labels, axis=1))
        predictions = model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))

    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return test_loss

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Print the best hyperparameters and result
best_trial = study.best_trial
print("Best trial:")
print("Value: ", best_trial.value)
print("Params: ")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

# Retrieve the best hyperparameters
best_learning_rate = best_trial.params["learning_rate"]
best_batch_size = int(best_trial.params["batch_size"])  # Convert batch_size to int

# Compile the model with the best learning rate
best_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=best_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=best_lr_schedule)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,  # Use the updated optimizer
    metrics=["accuracy"]
)

# Retrain the model with the best hyperparameters
history = model.fit(
    train_set,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,  # Reuse the global callbacks list
    validation_data=val_set
)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(test_set, verbose=1)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# F1 Score Calculation
true_labels = []
predicted_labels = []

for images, labels in test_set:
    true_labels.extend(np.argmax(labels, axis=1))
    predictions = model.predict(images)
    predicted_labels.extend(np.argmax(predictions, axis=1))

f1 = f1_score(true_labels, predicted_labels, average='weighted')
print('F1 Score:', f1)
