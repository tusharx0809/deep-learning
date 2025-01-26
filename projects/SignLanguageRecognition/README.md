<h1><em>Sign Language Recognition Model Using TensorFlow Deep learning library and Scikit-learn</em>

<a href="https://www.kaggle.com/datasets/grassknoted/asl-alphabet"><h2>American Sign Language Dataset</h2></a>


<h2>This project demonstrates the development and implementation of a deep learning model for recognizing American Sign Language (ASL) gestures. Using TensorFlow and Scikit-learn, the model is trained on a dataset of hand sign images, with the goal of achieving high classification accuracy for real-time applications in sign language recognition.</h2>

<h2>Approach</h2>
<h3>Dataset and Preprocessing</h3>
<ul>
  <li>The dataset consists of hand gesture images representing the alphabet in American Sign Language.</li>
  <li>Data Split: The dataset was split into training and testing sets, with 85% of the data used for training and 15% for testing (test_size = 0.15).</li>
  <li>Image Size: The images were resized to 64x64 pixels for consistency across the dataset.</li>
</ul>

<h3>Model and Architecture</h3>
<ul>
  <li>Convolutional Neural Networks (CNN) were used to extract spatial features from the images.</li>
  <li>Dropout Layers were introduced to prevent overfitting and ensure the model generalizes well.</li>
  <li>Data Augmentation techniques such as random rotations, flips, and zooms were applied to enhance the training dataset, increasing the model's robustness.</li>
</ul>

<h3>Results</h3>
<ul>
  <li>The model was trained on a total of 73,950 images and tested on 13,050 images.</li>
  <li>Training Accuracy: 98.21%</li>
  <li>Test Accuracy: 98.92%</li>
  <li>The model demonstrated excellent performance, as indicated by the high accuracy on both the training and test sets.</li>
</ul>

<h3>Evaluation</h3>
<ul>
  <li>A Confusion Matrix was plotted to visualize the model’s performance across different ASL signs.</li>
  <li>A Classification Report was generated to provide metrics such as precision, recall, and F1-score for each ASL gesture.</li>
</ul>

<h3>Key Highlights</h3>
<ul>
  <li>High Accuracy: Achieved 98.21% accuracy on the training set and 98.92% on the test set.</li>
  <li>Data Augmentation: Applied techniques that enhanced model generalization.</li>
  <li>Overfitting Prevention: Dropout layers and data augmentation contributed to reduced overfitting.</li>
  <li>Evaluation Metrics: Comprehensive analysis using confusion matrix and classification report.</li>
</ul>

<h3>Dependencies</h3>
<ul>
  <li>TensorFlow</li>
  <li>Keras</li>
  <li>Scikit-learn</li>
  <li>Matplotlib</li>
  <li>NumPy</li>
  <li>OpenCV</li>
</ul>

<h3>Confusion Matrix</h3>

![ConfusionMatrix](ConfustionMatrix.JPG)