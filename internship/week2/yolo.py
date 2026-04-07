from ultralytics import YOLO

# Load the latest YOLO26n model (nano version for speed)
model = YOLO("yolo26n.pt")

# Run inference on an image from a URL
results = model("image.jpeg")

# Display the results with bounding boxes
results[0].save(filename="detected.jpg")