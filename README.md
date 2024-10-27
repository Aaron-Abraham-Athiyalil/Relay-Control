# Relay Control Application

**Author:** Aaron Abraham Athiyalil  
**Registration Number:** 2260822  
**Department:** Electrical and Electronics Engineering (EEE)

## Overview

The Relay Control Application is a user-friendly software tool designed for controlling multiple relays through a graphical interface. It incorporates machine learning predictions to automate relay toggling based on user-defined input parameters. This application is built using PyQt5, providing a responsive and aesthetic interface.

## Features

- **User Input:** Accepts real-time input for various parameters to predict relay states.
- **Relay Control:** Manually toggle each relay on or off.
- **Prediction Toggle:** Automatically controls relays based on predictions made by a trained machine learning model.
- **Responsive Design:** Adapts to various screen sizes for optimal user experience.

## Requirements

- Python 3.x
- PyQt5
- NumPy
- scikit-learn (for model loading)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aaron-Abraham-Athiyalil/Relay-Control.git
   cd Relay-Control
   ```

2. Install the required packages:
   ```bash
   pip install PyQt5 numpy scikit-learn
   ```

3. Ensure your machine learning model (`multi_output_model.joblib`) is in the project directory.

## Usage

1. Run the application:
   ```bash
   python testuipyqt.py
   ```

2. Enter values in the input fields for IR, IY, IB, VR, VY, and VB.

3. Click the **Start** button to enable manual control of relays or the **Predict** button to toggle relays based on the model's predictions.

4. Use the relay buttons (L1 to L8) to turn on or off each relay manually.

## Acknowledgments

- Thanks to the contributors of the PyQt5 library and its documentation for providing the foundation for this application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
