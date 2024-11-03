# Knee Osteoarthritis Prediction API

This is a Flask-based API for predicting knee osteoarthritis severity using a pre-trained deep learning model. The API allows users to upload images of their knee X-rays and receive predictions regarding the severity of osteoarthritis.

## Table of Contents

1. **Installation**
2. **Usage**
3. **API Endpoints**
4. **Model**
5. **Requirements**
6. **Contributing**
7. **License**

## 1. Installation

- **Clone this repository:**
  
  ```
  git clone https://github.com/abdurrehman022/KOASystemAPI.git
  cd KOASystemAPI
  ```

- **Install the required packages:**
  
  ```
  pip install -r requirements.txt
  ```

- **Ensure you have the model file** `kneeosteoarthritis_957.28.h5` **in the root directory of the project.**

## 2. Usage

- **Start the Flask server:**
  
  ```
  python app.py
  ```

- **The API will be available at** `http://localhost:8000`.

## 3. API Endpoints

### Predict

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Description:** Classifies the severity of knee osteoarthritis based on the provided X-ray image.
  
  - **Request:**
    - **Content-Type:** `multipart/form-data`
    - **Body:** 
      - `file`: An image file of the knee X-ray (PNG, JPEG).
  
  - **Response:**
    - **Success (200):**
      ```json
      {
          "prediction": "Healthy",
          "confidence": "85.65%"
      }
      ```
    - **Error (400):**
      ```json
      {
          "error": "No file part"
      }
      ```

### Index

- **Endpoint:** `/`
- **Method:** `GET`
- **Description:** A simple welcome message.

## 4. Model

The model used in this API is a pre-trained deep learning model for classifying knee osteoarthritis severity into three classes: 
- **Healthy**
- **Moderate**
- **Severe**

The model was trained on a dataset of knee X-rays and can provide confidence scores for its predictions.

## 5. Requirements

Ensure you have the following installed:

- **Python 3.9**
- **Flask**
- **TensorFlow**
- **PIL (Pillow)**
- **NumPy**

You can install all requirements by running:
  
```
pip install -r requirements.txt
```

## 6. Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.


## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
