# File: train.py

import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_preprocessed_data(file_path):
    """Load preprocessed data from file."""
    with open(file_path, "rb") as f:
        X, y = pickle.load(f)
    return X, y

def build_model(input_shape):
    """Define the neural network model with improvements."""
    from keras.regularizers import l2

    model = Sequential()
    # Layer 1
    model.add(Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    # Layer 2
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    # Layer 3
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.3))
    # Output Layer
    model.add(Dense(3, activation='softmax'))  # 3-class output: H, E, C
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lr_schedule(epoch, lr):
    """Learning rate schedule: Reduce LR every 10 epochs."""
    return lr * 0.95 if epoch > 10 else lr

def plot_training(history):
    """Visualize training and validation accuracy/loss."""
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

def train_model(X, y):
    """Train the neural network with improvements."""
    # One-hot encode labels
    y = to_categorical(y, num_classes=3)

    # Split dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build and train the model
    model = build_model(X_train.shape[1])

    # Add callbacks for learning rate scheduling and early stopping
    callbacks = [
        LearningRateScheduler(lr_schedule, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_acc}")

    # Save the model in .keras format
    model.save("secondary_structure_model.keras")
    print("Model saved as secondary_structure_model.keras")

    # Plot training performance
    plot_training(history)

    return history, model

# Main Execution
if __name__ == "__main__":
    preprocessed_data_file = "CB513_features.pkl"  # Input file with features and labels
    X, y = load_preprocessed_data(preprocessed_data_file)
    train_model(X, y)
