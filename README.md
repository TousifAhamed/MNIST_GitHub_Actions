# MNIST Image Classification and Augmentation API 🚀

A FastAPI application that combines MNIST digit classification with advanced image augmentation capabilities. The model achieves >95% accuracy in a single epoch while maintaining less than 25,000 parameters. The application includes a modern web interface for interactive image augmentation with real-time previews and detailed descriptions.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)


## 🌟 Features

### 1. MNIST Classification
- Efficient CNN architecture (<25,000 parameters)
- High accuracy (>95%) in single epoch
- Batch normalization and dropout for regularization
- Optimized training process

### 2. Image Augmentation
- **Rotation**: Precise angle-based rotation with dimension preservation
- **Noise**: Controlled Gaussian noise addition
- **Brightness**: Adjustable brightness manipulation
- **Affine**: Combined geometric transformations (rotation, scale, shear, translation)

### 3. Modern Web Interface
- Interactive UI for image augmentation
- Real-time preview
- Parameter adjustment controls
- Detailed augmentation descriptions

### 4. Security Features
- JWT-based authentication
- Password hashing
- CORS protection
- Input validation

## 🛠️ Installation

1. Clone the repository: 
git clone https://github.com/TousifAhamed/MNIST_GitHub_Actions.git


2. Create and activate a virtual environment:
```bash
python -m venv mnist_env
source mnist_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Set the JWT signing secret (required for login tokens):
```bash
# macOS/Linux
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(64))")
# Windows PowerShell
$env:JWT_SECRET_KEY = python -c "import secrets; print(secrets.token_urlsafe(64))"
```
If `JWT_SECRET_KEY` is not set, the app generates a random key at startup and logs a warning. Tokens then stop working whenever the server restarts. In production, always set it, and keep it secret (never commit it).

## 🚀 Running the Application

1. Start the FastAPI server:
```bash
uvicorn app.api.endpoints:app --reload
```

2. Access the application:
- Main API: http://localhost:8000/
- Augmentation UI: http://localhost:8000/augmentation/augmentation-ui
- API Documentation: http://localhost:8000/docs

## 🧪 Running Tests

### Local Testing

Run all tests:
```bash
pytest
```

Run specific test categories:

## Model tests
pytest app/tests/test_model.py -v
## Augmentation tests
pytest app/tests/test_augmentation.py -v


### GitHub Actions

The repository includes automated testing via GitHub Actions, which tests:
1. Model parameter count (<25,000)
2. Model accuracy (>95%)
3. Image augmentation functionality
4. Cross-platform compatibility (Windows/Linux)

## 📊 Model Architecture
```
MNISTNet:
├── Conv1 (8 filters, 3x3)
│   ├── BatchNorm
│   ├── ReLU
│   └── MaxPool
├── Conv2 (16 filters, 3x3)
│   ├── BatchNorm
│   ├── ReLU
│   └── MaxPool
└── Fully Connected
```

## 🎨 Augmentation Types

1. **Rotation**
   - Range: -180° to 180°
   - Preserves image dimensions
   - Maintains color information

2. **Noise**
   - Gaussian noise addition
   - Controllable intensity (0.01-0.2)
   - Per-channel application

3. **Brightness**
   - Factor range: 0.5-2.0
   - Preserves image structure
   - Color-aware adjustment

4. **Affine**
   - Combined geometric transformations
   - Rotation + Scale + Shear + Translation
   - Dimension preservation

## 🔧 API Endpoints

### Classification
- `POST /predict`: Classify MNIST digit
  - Requires authentication
  - Accepts image file
  - Returns predicted digit

### Augmentation
- `GET /augmentation/augmentation-ui`: Interactive UI
- `POST /augmentation/augment`: Apply augmentation
  - Accepts image file
  - Augmentation type
  - Parameters

## 🔒 Security

1. Authentication required for predictions
2. File type validation
3. Parameter range validation
4. CORS protection
5. Rate limiting (configurable)

## 📈 Performance Metrics

- Model Parameters: 9,146
- Training Accuracy: >95%
- Training Time: Single epoch
- Memory Usage: Efficient (<100MB)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Email - tousifahamed11@gmail.com
Project Link: https://github.com/TousifAhamed/MNIST_GitHub_Actions.git

## 🙏 Acknowledgments

- PyTorch Team
- FastAPI
- torchvision
- PIL/Pillow
