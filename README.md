# CSI to Image Reconstruction

This project implements a system for reconstructing images from Channel State Information (CSI) using Wi-Fi signals. It consists of an embedded component (ESP32), a server for data collection and streaming (Python), and logic for model training.

## Project Structure

- **01_Embedded/**: ESP32 firmware code.
  - `01_cam/`: Camera module firmware (ESP32-CAM, ESP32-S3-CAM).
  - `02_csi/`: CSI collection firmware (TX, RX, Gateway).
- **02_Server/**: Backend server code.
  - `01_Data_Collection/`: Scripts for collecting CSI data.
  - `02_Streaming/`: Real-time streaming server using FastAPI and PyTorch.
- **03_Model_Training/**: Machine learning model training scripts.
  - `01_MoPoEVAE/`: Mixture-of-Product-of-Experts VAE implementation.
  - `02_VAE/`: Standard VAE implementation.

## Configuration

- **Embedded**: Update `sdkconfig.defaults` (WiFi) and source code (MAC addresses for TX, Server IP for Gateway/Camera) in `01_Embedded`.
- **Server**: Ensure the model checkpoint path is correct in `02_Server/02_Streaming/main.py`.

## Workflows

### 1. Data Collection & Training
Manage the full pipeline from raw signal gathering to model training.

1.  **Flash Firmware**: Build and flash the TX, RX, Gateway, and Camera firmware from `01_Embedded`.
2.  **Collect Data**: Run the collection server to save synchronized CSI and image data.
    ```bash
    cd 02_Server/01_Data_Collection
    python main.py
    ```
3.  **Train Model**: Run the training scripts in `03_Model_Training` using the collected dataset.
    ```bash
    cd 03_Model_Training/02_VAE
    python train.py
    ```

### 2. Real-time Streaming
Deploy the trained model for live image reconstruction.

1.  **Flash Firmware**: Ensure the Gateway is flashed with the streaming firmware.
2.  **Run Inference Server**: Execute the FastAPI server to process live CSI and serve the reconstructed images.
    ```bash
    cd 02_Server/02_Streaming
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
3.  **Visualize**: Open `http://localhost:8000` in your browser to see the real-time reconstruction.

## References

- **MoPoEVAE Implementation**: 
  - Based on the paper: [Sutter, T., Daunhawer, I., & Vogt, J. (2021). Generalized Multimodal ELBO](https://arxiv.org/abs/2105.02470).
  - Original Code: [thomassutter/MoPoE](https://github.com/thomassutter/MoPoE)
- **KL Approximation**:
  - Implementation from: [senya-ashukha/variational-dropout-sparsifies-dnn](https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn)


