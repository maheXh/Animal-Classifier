dataset : https://www.kaggle.com/datasets/puneet6060/intel-image-classification/code

results:

![9dd6a996-b1c8-41c2-affd-d814c6f341e5](https://github.com/maheXh/Landscape-Classifier/assets/122071980/b3688189-4776-4f73-b889-3b0d62bcd3ff)
![cb1237c0-feda-44c0-b444-921ff19963c7](https://github.com/maheXh/Landscape-Classifier/assets/122071980/be3bb649-8da2-4468-9a5b-e81072e6fe99)
![90590a82-0fcd-4431-9b8a-c53968764690](https://github.com/maheXh/Landscape-Classifier/assets/122071980/e43aca87-2e30-4faa-9e5d-67f3a3926e17)


### driver code


```
import cv2
import torch
import numpy as np
from torchvision import transforms

# Define mean and standard deviation for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Function to preprocess the image for model input
def preprocess_image(image):
    # Resize the image to match the input size of the model (64x64)
    resized_image = cv2.resize(image, (150, 150))
    # Convert the image to the format expected by PyTorch (RGB and float32)
    processed_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Normalize the image using the same mean and standard deviation as used during training
    processed_image = (processed_image - mean) / std
    # Convert the image to a PyTorch tensor and ensure it's of type float32
    tensor_image = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1)
    return tensor_image


# Function to predict label for a frame
def predict_frame_label(model, frame):
    # Preprocess the frame
    processed_frame = preprocess_image(frame)
    # Move the processed frame to the same device as the model
    processed_frame = processed_frame.to(device)
    # Perform inference
    output = model(processed_frame.unsqueeze(0))
    _, predicted = torch.max(output, 1)
    probabilities = torch.softmax(output, dim=1)
    return predicted.item(), probabilities[0][predicted.item()].item()

# Load the model
loaded_model = MyModule(6).to(device)
loaded_model.load_state_dict(torch.load('landscapeDetector.pth'))
loaded_model.eval()

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop to capture frames from the webcam
while(cap.isOpened()):
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform prediction on the frame
    predicted_label, probability = predict_frame_label(loaded_model, frame)
    
    # Print the predicted label and its probability on the frame
    label_text = f"Predicted Label: {predicted_label}, Probability: {probability:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()

```
