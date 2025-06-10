from ultralytics import YOLO

dataset = 'dataset'
# Initialize the model
model = YOLO('yolo11m.pt')
# Train the model
results = model.train(
    data=f'{dataset}/data.yaml',
    epochs=50,           
    batch=8,              # Lower batch size to avoid overfitting and GPU OOM
    imgsz=640,            # Keep as is unless you know a better resolution for your task
    lr0=0.003,            # Lower initial LR to avoid divergence on small datasets
    lrf=0.01,             # Smaller decay for a more gradual LR reduction
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,          # Early stopping if no improvement
    cache=True            # Cache data in memory to speed up training
)
