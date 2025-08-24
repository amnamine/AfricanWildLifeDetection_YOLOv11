import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("african.pt")

# --- Functions ---
def load_image():
    global img_path, tk_img, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        reset_image()  # clear old display
        img = Image.open(img_path)
        img.thumbnail((500, 500))
        tk_img = ImageTk.PhotoImage(img)
        img_display = canvas.create_image(250, 250, image=tk_img)

def predict():
    global img_path, tk_img, img_display
    if not img_path:
        return
    # Run YOLO prediction
    results = model(img_path)

    # Results[0].plot() gives OpenCV image with bounding boxes
    res_img = results[0].plot()  
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)  # convert to RGB for Tkinter
    res_img = Image.fromarray(res_img)
    res_img.thumbnail((500, 500))
    tk_img = ImageTk.PhotoImage(res_img)
    canvas.delete("all")
    img_display = canvas.create_image(250, 250, image=tk_img)

def reset_image():
    canvas.delete("all")

# --- GUI Setup ---
root = tk.Tk()
root.title("African Wildlife Detection with YOLO")
root.geometry("600x650")

canvas = tk.Canvas(root, width=500, height=500, bg="gray")
canvas.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

btn_load = tk.Button(frame, text="Load Image", command=load_image, width=15, bg="lightblue")
btn_load.grid(row=0, column=0, padx=5)

btn_predict = tk.Button(frame, text="Predict", command=predict, width=15, bg="lightgreen")
btn_predict.grid(row=0, column=1, padx=5)

btn_reset = tk.Button(frame, text="Reset", command=reset_image, width=15, bg="lightcoral")
btn_reset.grid(row=0, column=2, padx=5)

img_path = None
img_display = None
tk_img = None

root.mainloop()
