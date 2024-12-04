import torch
import torchvision
import requests
from PIL import Image
import streamlit as st

class ObjectPriceDetector:
    def __init__(self):
        # Load pre-trained object detection model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        # COCO class labels (common objects)
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
            'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
            'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Mock price database (replace with real API or database)
        self.price_database = {
            'laptop': 1000,
            'smartphone': 800,
            'chair': 150,
            'bicycle': 300,
            'book': 20,
            # Add more items
        }

    def detect_objects(self, image):
        """
        Detect objects in the image using pre-trained model
        """
        # Convert image to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        input_image = transform(image).unsqueeze(0)
        
        # Perform object detection
        with torch.no_grad():
            prediction = self.model(input_image)
        
        return prediction

    def get_object_prices(self, prediction, image):
        """
        Extract detected objects and their prices
        """
        prices = []
        boxes = prediction[0]['boxes'].detach().numpy()
        labels = prediction[0]['labels'].detach().numpy()
        scores = prediction[0]['scores'].detach().numpy()
        
        # Filter detections by confidence
        for i, score in enumerate(scores):
            if score > 0.7:  # 70% confidence threshold
                label = self.COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
                box = boxes[i]
                
                # Look up price
                price = self.price_database.get(label, "Price not available")
                
                prices.append({
                    'object': label,
                    'price': price,
                    'box': box,
                    'confidence': score
                })
        
        return prices

    def visualize_results(self, image, prices):
        """
        Draw bounding boxes and price information on image
        """
        import cv2
        import numpy as np
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for item in prices:
            box = item['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text with object and price
            label = f"{item['object']}: ${item['price']}"
            cv2.putText(img_cv, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return img_cv

def main():
    st.title("Object Price Detection App")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Initialize detector
        detector = ObjectPriceDetector()
        
        # Detect objects
        predictions = detector.detect_objects(image)
        
        # Get prices
        prices = detector.get_object_prices(predictions, image)
        
        # Visualize results
        result_image = detector.visualize_results(image, prices)
        
        # Display results
        st.image(result_image, caption="Detected Objects with Prices", use_column_width=True)
        
        # Show price list
        st.subheader("Detected Object Prices:")
        for item in prices:
            st.write(f"{item['object']}: ${item['price']} (Confidence: {item['confidence']:.2f})")

if __name__ == "__main__":
    main()
