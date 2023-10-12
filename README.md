# Facial Recognition Project

## Project Overview
This project aims to build a facial recognition model to identify and compare faces in images. The implementation utilizes Support Vector Machines (SVM) for model training, OpenCV for image processing, and various other tools to analyze and compare facial features in different images. 

## Table of Contents
1. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
2. [Usage](#usage)
3. [Model Information](#model-information)
4. [Project Structure](#project-structure)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Requests
- Pillow

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your_username_/ProjectName.git
   ```
2. Install Python packages:
   ```sh
   pip install opencv-python scikit-learn numpy pandas joblib requests Pillow
   ```
   
## Usage
This section explains how to use the project after it has been set up successfully.

### Preprocessing Images
Images are loaded and preprocessed through a series of steps, including resizing, normalization, and flattening.

### Model Training
A SVM model is trained on preprocessed facial images and is capable of making predictions on new, unseen images.

### Comparing Faces
The trained model can be used to compare faces in two different images and predict whether they belong to the same person.

### Predicting Faces
Once the model is trained, it can predict the person in a given image.

Example:
```cmd
python main.py path/to/image1 path/to/image2 --local
```

## Model Information
The model used in this project is a Support Vector Machine (SVM) with a linear kernel. Further details about model training, selection, and evaluation can be found in the project notebooks.

## Project Structure
- `main.py`: Main script to run the facial recognition model.
- `model.py`: Contains functions related to image preprocessing.
- `svm_model.pkl`: Pre-trained SVM model.
- `label_encoder.pkl`: Label encoder for decoding model predictions.
- `/path_to/data`: Directory containing facial images for model training.

## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
- [OpenCV](https://opencv.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Joblib](https://joblib.readthedocs.io/)
