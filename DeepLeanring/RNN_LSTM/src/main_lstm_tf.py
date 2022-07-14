from tensorflow_docs.vis import embed
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, History
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import pydotplus

from keras import layers
from keras.layers.recurrent import LSTM



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的信息


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 14

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

train_df.sample(10)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

### pre-train using Inception V3

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
keras.utils.plot_model(feature_extractor, to_file=r'./pre_model.png', show_shapes = True) 

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.

    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )

        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels


train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []
        self.losses = []
        self.val_loss = []
        self.val_acc= []

    def on_epoch_end(self, epoch, logs={}):
        self.accuracy.append(logs.get('accuracy'))
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def lstm():
    """
    Build a simple LSTM network. We pass 
    the extracted features from our CNN
    to this model predomenently.
    """
    print("Loading LSTM model.")

    class_vocab = label_processor.get_vocabulary()
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))

    x = LSTM(16, return_sequences=True, dropout=0.5)(frame_features_input)
    x = LSTM(8, return_sequences=False)(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    lstm_output = layers.Dense(len(class_vocab), activation='softmax')(x)

    lstm_model = keras.Model(frame_features_input, lstm_output)

    lstm_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return lstm_model


# Utility for running experiments.
def run_experiment():
    filepath = "./tmp/video_classifier"

    # checkpointer = ModelCheckpoint(
    #     filepath, save_weights_only=True, save_best_only=True, verbose=1
    # )

    history = LossHistory()
    my_callbacks = [
        ModelCheckpoint(filepath, save_weights_only=True, save_best_only=True, verbose=1),
        history,
]

    seq_model = lstm()
    keras.utils.plot_model(seq_model, to_file=r'./train_model_lstm.png', show_shapes=True)

    history_plus = seq_model.fit(
        train_data,
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=my_callbacks,
    )

  #  training_vis(history)
  
    plt.figure(num=1)
    plt.xlabel('epoch')
    plt.plot(history.accuracy)
    plt.plot(history.losses)
    plt.plot(history.val_loss)
    plt.legend(['accuracy','loss', 'val_loss'],loc='upper right')
#    plt.savefig("./trainplot/acc.png") 

    seq_model.load_weights(filepath)

    _, accuracy = seq_model.evaluate(test_data, test_labels)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history_plus, seq_model

_, sequence_model = run_experiment()

##########################################################################

def prepare_single_video(frames):
    frames = frames[None, ...]
#    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
#        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

#    return frame_features, frame_mask
    return frame_features


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
#    frame_features, frame_mask = prepare_single_video(frames)
    frame_features = prepare_single_video(frames)
#    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    probabilities = sequence_model.predict(frame_features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")


test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])



plt.show()
