import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

#from actual_final import generate_caption

# Load the pre-trained model
model = load_model('caption_model')

# model.summary()

# Load tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = f.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
vocab_size = len(tokenizer.word_index) + 1
max_length = 52 

# Load validation features
val_features = np.load('val_features.npy', allow_pickle=True).item()

# Load validation descriptions
with open('val_descriptions.json', 'r') as f:
    val_descriptions = json.load(f)


# Function to generate caption
def generate_caption(image, model, tokenizer, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([np.array([image]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    final_caption = in_text.split(' ', 1)[1] if len(in_text.split(' ', 1)) > 1 else in_text
    return final_caption


# Prepare the EfficientNetB0 model for Grad-CAM
base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
last_conv_layer_name = 'top_conv'  # Last convolutional layer in EfficientNetB0
grad_model = tf.keras.models.Model([base_model.input], [base_model.get_layer(last_conv_layer_name).output, base_model.output])


# Function to compute Grad-CAM
def compute_gradcam(image, grad_model, layer_name, class_idx=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([np.array([image])])
        if class_idx is None:
            class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (image.shape[1], image.shape[0]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


# Function to overlay Grad-CAM heatmap on image
def overlay_gradcam(original_image, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    resized_image = cv2.resize(original_image, (224, 224)).astype(np.float32) / 255.0
    
    # Blend original image with heatmap
    alpha = 0.75  # Transparency factor for the heatmap
    blended_image = cv2.addWeighted(heatmap, alpha, resized_image, 1 - alpha, 0)
    blended_image = np.uint8(255 * blended_image)
    return blended_image


# Preprocess image for EfficientNetB0
def preprocess_image(file_path):
    img_raw = tf.io.read_file(file_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_final = tf.image.resize(img_tensor, [224, 224])
    img_final = preprocess_input(img_final)
    return img_final


# Select a small random sample for Grad-CAM visualization
num_samples = 20
sample_keys = random.sample(list(val_features.keys()), num_samples)


for idx, img_key in enumerate(sample_keys):
    fig, ax = plt.subplots(figsize=(6, 4)) 
    # Load image
    img_path = f'coco2017/val2017/{img_key}'
    image = plt.imread(img_path)
    
    # Preprocess image for Grad-CAM
    processed_image = preprocess_image(img_path)

    # Generate caption
    caption = generate_caption(val_features[img_key], model, tokenizer, max_length)
    
    # Compute Grad-CAM
    cam = compute_gradcam(processed_image, grad_model, last_conv_layer_name)
    cam_image = overlay_gradcam(image, cam)
    
    # Display image
    ax.imshow(cam_image)
    ax.axis('off')

    # Add caption as text with background
    ax.add_patch(Rectangle((0, 0.85), 1, 0.15, color='black', alpha=0.5, transform=ax.transAxes))
    ax.text(0.5, 0.9, caption, fontsize=9, color='white', ha='center', va='center', transform=ax.transAxes)

    # Save the figure
    plt.savefig(f'output_gradcam_{img_key}.png', bbox_inches='tight')
    plt.close(fig) 
