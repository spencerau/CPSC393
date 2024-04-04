import json
import matplotlib.pyplot as plt

file_path = "history/training_history_unfreeze_3.json"
save_name = "assets/training_metrics_3.png"

# Load the training history
with open(file_path, 'r') as file:
    history = json.load(file)

# Create subplots
plt.figure(figsize=(21, 7))

# Plot training & validation accuracy values
plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation IoU values
plt.subplot(1, 3, 3)
plt.plot(history['top_k_categorical_accuracy'], label='Train Top K Categ Acc') 
plt.plot(history['val_top_k_categorical_accuracy'], label='Val Top K Categ Acc') 
plt.title('Model Top K Categ Acc')
plt.ylabel('Top K Categ Acc')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Save the figure
plt.savefig(save_name)
plt.close()
