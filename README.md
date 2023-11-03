# FaceAlignment-Survey
This repository contains the code used in the Meher et al. Survey of face alignment methods. 
We have reimplemented different face alignment methods so that the readers can take the benefit of trying out different face alignment methods to better understand the process of face alignment. 

## Overview

### Face Alignment

Face alignment is a computer vision technique that involves standardizing the scale, rotation, and position of faces within images. The goal is to ensure that the facial features (eyes, nose, mouth, etc.) are aligned across different images, facilitating subsequent tasks such as facial recognition, expression analysis, and other facial analyses.

This process typically involves detecting specific facial landmarks — key points on a face, such as the corners of the eyes, the tip of the nose, and the corners of the mouth. Once detected, these landmarks can be used to apply transformations to the image to align the faces consistently. This usually involves steps like translation, scaling, and rotation based on the positions of the landmarks.

In the context of this project, face alignment plays a crucial role in preparing data for both Active Shape Models (ASM) and Active Appearance Models (AAM). Proper alignment ensures that the statistical models accurately capture the variance due to facial shapes and textures, rather than variances caused by orientation or position of the face within the image.

## Directories

### ASM

The `ASM` directory holds scripts related to the Active Shape Model. It includes tools for landmark detection, shape fitting, and model training.

For more information on the contents and how to run the scripts within the `ASM` directory, refer to its [README.md](ASM/README.md).

### AAM

The `AAM` directory contains scripts for the Active Appearance Model. It provides functionalities for texture mapping, model building, and image synthesis.

For details on usage and structure, see the [README.md](AAM/README.md) file in the `AAM` directory.

## Data

Datasets used by both ASM and AAM analyses are stored in the `Data` directory. This includes labeled facial images and other resources necessary for training and evaluating the models. Refer to the [README.md](Data/README.md) in the `Data` directory for instructions on how to organise and use the datasets.



## Getting Started

To begin using this project:

1. Clone the repository to your local machine.
2. Navigate to the subdirectory of interest and follow the instructions in the respective `README.md` file.
3. Download and organise the required datasets in the `Data` directory as described in its `README.md`.

## Contributing

Contributions to this project are welcome. To contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [Apache 2.0](LICENSE).

## Contact

If you have any questions or feedback, please open an issue in the GitHub repository or contact the maintainers at [torbjöurn.nordling@nordlinglab.org](mailto:your-email@example.com) or [jagmohan.meher@nordlinglab.org](mailto:jagmohan.meher@nordlinglab.org)