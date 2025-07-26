# Image Classification of Cats, Dogs, and Pandas using Fine-Tuned VGG19 CNN Model

## Problem Statement
The aim of this project is to build an image classification system that can accurately identify cats, dogs, and pandas using deep learning. By fine-tuning the pretrained VGG19 model on a labeled dataset of animal images, the project demonstrates the effectiveness of transfer learning in solving multi-class classification tasks with high accuracy and minimal training time.

## Dataset Overview
The dataset used in this project consists of RGB images of three animal categories: cats, dogs, and pandas. The images are organized into labeled folders, with each class having its own directory. The dataset is split into training and testing sets to evaluate model performance.

- Classes: 3 — Cat, Dog, Panda

- Image Type: Colored (RGB) images

- Image Size (after preprocessing): 224 × 224 pixels

  ### Preprocessing Steps:

- Resizing to 224×224 (required for VGG19)

- Normalization using ImageNet mean and std values

- Optional: Augmentations (flip, rotation, etc.)

This structured and preprocessed dataset enables effective training of the VGG19 model for multi-class classification.

### VGG19 Model and Its Working

VGG19 is a deep Convolutional Neural Network (CNN) developed by the Visual Geometry Group (VGG) at the University of Oxford. It is known for its simplicity, depth, and use of small convolution filters. Originally trained on ImageNet, it can be fine-tuned for custom datasets like cats, dogs, and pandas using transfer learning.

### Architecture Overview:

- Total Layers: 19 (16 convolutional + 3 fully connected)

#### Convolution Layers:

- Uses 3×3 filters with stride=1 and padding=1

- Deeper layers capture complex patterns (edges → textures → objects)

- Activation Function: ReLU after each conv layer

- Pooling: MaxPooling (2×2) after certain blocks to reduce spatial dimensions

#### Fully Connected Layers:

- Two dense layers with 4096 neurons

- Final layer replaced for your 3-class classification

### Working Steps:
  
- The input image, resized to 224×224 RGB, is passed through multiple convolutional and pooling layers in the VGG19 model. These layers automatically extract meaningful features such as edges, textures, eyes, ears, and fur patterns that help distinguish between animals. The output from the final convolutional layer is flattened and fed into a series of fully connected (FC) layers, which learn to combine the extracted features. The last FC layer produces three class scores corresponding to cat, dog, and panda using a softmax activation function, and the class with the highest score is selected as the predicted label.

- In this project, transfer learning is applied by reusing the pretrained VGG19 weights from ImageNet. Only the final classification layer is modified and retrained on the specific animal image dataset. This approach significantly reduces training time and computational cost while improving accuracy, especially when working with smaller or limited datasets.

- VGG19 performs well in this task because it learns deep hierarchical features, captures complex patterns effectively, and generalizes well even with a small number of training epochs. The use of transfer learning further boosts performance by leveraging knowledge from a much larger dataset.
  
- The model was trained for 3 epochs using a batch size of 2400 for training and 600 for testing. The CrossEntropyLoss function was used as the loss criterion since it is well-suited for multi-class classification tasks. For optimization, either the Adam or SGD optimizer was employed to update the model weights efficiently. During each training step, batches of cat, dog, and panda images were loaded and passed through the VGG19 network to generate predictions.
  
- The loss between the predicted and true class labels was computed, and the model then performed backpropagation to adjust its internal weights. To monitor the overall training efficiency, the time.time() function was used to record the total training time, which provided a useful benchmark for understanding model performance and resource usage.

### Evaluation and Results
The model was trained over 3 epochs, and the results indicate steadily improving accuracy and decreasing loss:

#### Epoch 1:

Started with ~66.67% accuracy

Quickly climbed to 94.43% by the final batch

Sharp drop in loss, indicating good learning

#### Epoch 2:

Started at ~90.48%, reached 97.17% by final batch

Minimal loss across most batches (mostly 0.0000)

Minor spikes (e.g., batch 181: loss 7.9210) had negligible impact

#### Epoch 3:

Started at 94.05%, peaked at 97.85%

Few irregular spikes in loss (e.g., batch 361: 4.83, batch 401: 14.31)

Model remained highly stable overall

#### Accuracy Trend:
Across batches, training accuracy steadily improved:

- Epoch 1: 66.67% ➝ 94.43%

- Epoch 2: 90.48% ➝ 97.17%

- Epoch 3: 94.05% ➝ 97.75%

This clearly shows that the model learned quickly and generalized well, especially by the second epoch.

#### Prediction

<img width="425" height="417" alt="image" src="https://github.com/user-attachments/assets/10398d48-ad5f-4e4c-ac8d-ff34a6b95283" />

panda

#### Confusion Matrix

<img width="697" height="530" alt="image" src="https://github.com/user-attachments/assets/11feeb08-f387-499e-a766-2e9fde8ae87f" />
