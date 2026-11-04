import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from model import DigitCNN


# --------------------------------
# 1. Load the trained CNN model
# --------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DigitCNN().to(device)
model.load_state_dict(
    torch.load(
        "digit_cnn.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

print("Model loaded successfully!")


# --------------------------------
# 2. Image preprocessing
# --------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# --------------------------------
# 3. Create DigitVision GUI
# --------------------------------

root = tk.Tk()
root.title("DigitVision - Handwritten Digit Recognition")
root.geometry("430x600")
root.resizable(False, False)

title = tk.Label(
    root,
    text="DigitVision",
    font=("Arial", 24, "bold")
)
title.pack(pady=(12, 4))

instruction = tk.Label(
    root,
    text="Draw any digit from 0 to 9",
    font=("Arial", 12)
)
instruction.pack(pady=4)


# --------------------------------
# 4. Drawing canvas and image buffer
# --------------------------------

canvas_size = 280

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    cursor="cross",
    highlightthickness=2,
    highlightbackground="gray"
)
canvas.pack(pady=12)

drawing = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(drawing)


# --------------------------------
# 5. Drawing function
# --------------------------------

def paint(event):
    x, y = event.x, event.y
    radius = 9

    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=255
    )


canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", paint)


# --------------------------------
# 6. Prediction function
# --------------------------------

def predict_digit():
    bbox = drawing.getbbox()

    if bbox is None:
        messagebox.showinfo(
            "DigitVision",
            "Please draw a digit first!"
        )
        return

    # Crop the drawn digit to its bounding box
    cropped = drawing.crop(bbox)

    # Resize while preserving the digit's aspect ratio
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit in the center of a 28x28 image
    img = Image.new("L", (28, 28), 0)

    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2
    img.paste(cropped, (x, y))

    # Apply the same preprocessing used during training
    image_tensor = transform(img).unsqueeze(0).to(device)

    # Run model inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_digit = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_digit].item() * 100

        top_values, top_indices = torch.topk(probabilities, 3)

    result_label.config(
        text=(
            f"Predicted Digit: {predicted_digit}\n"
            f"Model Confidence: {confidence:.1f}%"
        )
    )

    alternatives = "Other predictions: "

    for i in range(1, 3):
        digit = top_indices[0, i].item()
        score = top_values[0, i].item() * 100
        alternatives += f"{digit} ({score:.1f}%)  "

    alternatives_label.config(text=alternatives)


# --------------------------------
# 7. Clear canvas function
# --------------------------------

def clear_canvas():
    global drawing, draw

    canvas.delete("all")

    drawing = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(drawing)

    result_label.config(
        text="Your prediction will appear here"
    )
    alternatives_label.config(text="")


# --------------------------------
# 8. Buttons
# --------------------------------

button_frame = tk.Frame(root)
button_frame.pack(pady=12)

predict_button = tk.Button(
    button_frame,
    text="Predict Digit",
    command=predict_digit,
    font=("Arial", 12, "bold"),
    width=14,
    bg="#198754",
    fg="white",
    cursor="hand2"
)
predict_button.grid(row=0, column=0, padx=8)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 12, "bold"),
    width=14,
    bg="#dc3545",
    fg="white",
    cursor="hand2"
)
clear_button.grid(row=0, column=1, padx=8)


# --------------------------------
# 9. Prediction result labels
# --------------------------------

result_label = tk.Label(
    root,
    text="Your prediction will appear here",
    font=("Arial", 15, "bold"),
    fg="blue",
    justify="center"
)
result_label.pack(pady=(12, 5))

alternatives_label = tk.Label(
    root,
    text="",
    font=("Arial", 10),
    fg="gray",
    wraplength=390
)
alternatives_label.pack(pady=5)


# --------------------------------
# 10. Start the application
# --------------------------------

root.mainloop()
