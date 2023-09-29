import tkinter as tk
from tkinter import ttk

def update_scale_label(scale_value):
    label_value.config(text=str(scale_value))

root = tk.Tk()
root.geometry("200x100")

frame = ttk.Frame(root)
frame.pack(fill="x", padx=(20, 0), pady=(5, 0))

ttk.Label(frame, text='Epoch:').pack(side="left")

scale_value = tk.IntVar()
scale = ttk.Scale(frame, variable=scale_value, from_=1, to=50, command=update_scale_label)
scale.pack(side="left", fill="x", expand=True, padx=5)

label_value = ttk.Label(frame)
label_value.pack(side="left")

root.mainloop()
