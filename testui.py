import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Load the multi-output model and make predictions based on user input
def test_multi_output_model(IR, IY, IB, VR, VY, VB):
    model = joblib.load('multi_output_model.joblib')
    input_data = np.array([[IR, IY, IB, VR, VY, VB]])
    predictions = model.predict(input_data)[0]

    # Update output labels and buttons
    for i in range(8):
        outputs[i].config(text=f'L{i + 1}: {"On" if predictions[i] == 1 else "Off"}')
        buttons[i].config(style="On.TButton" if predictions[i] == 1 else "Off.TButton")

# Function to retrieve input values and call the prediction function
def predict():
    try:
        IR = float(entry_IR.get())
        IY = float(entry_IY.get())
        IB = float(entry_IB.get())
        VR = float(entry_VR.get())
        VY = float(entry_VY.get())
        VB = float(entry_VB.get())

        test_multi_output_model(IR, IY, IB, VR, VY, VB)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Function to start/stop prediction
def toggle_prediction():
    if start_stop_button['text'] == "Start":
        start_stop_button.config(text="Stop", style="Stop.TButton")
        predict()  # Call predict once to get initial values
    else:
        start_stop_button.config(text="Start", style="Start.TButton")
        for i in range(8):
            outputs[i].config(text=f'L{i + 1}: Off')
            buttons[i].config(style="Off.TButton")  # Reset buttons to off style

# Create a tkinter window with modern styling
window = tk.Tk()
window.title("Relay Model Tester")
window.geometry("800x480")  # Adjust to fit your LCD dimensions
window.configure(bg='#2D2D2D')  # Dark background color for a modern look

# Modern color palette and fonts
primary_color = '#FFFFFF'  # White for text
secondary_color = '#424242'  # Grey for background elements
accent_color = '#00E5FF'   # Cyan for active buttons
start_color = '#76FF03'     # Lime Green for Start button
stop_color = '#FF1744'      # Red for Stop button
off_color = '#FFFFFF'        # White for inactive buttons
active_color = '#00FF00'     # Hacker Green for active buttons
entry_bg_color = '#2D2D2D'  # Background color for input fields
entry_fg_color = '#000000'   # Text color for input fields (black)

title_font = ('Helvetica Neue', 20, 'bold')
label_font = ('Helvetica Neue', 12)
entry_font = ('Helvetica Neue', 12)

# Style Configuration for Buttons and Labels
style = ttk.Style()
style.configure("TLabel", background="#2D2D2D", foreground=primary_color, font=label_font)
style.configure("TEntry", fieldbackground=entry_bg_color, foreground=entry_fg_color, font=entry_font, justify="center")
style.configure("Start.TButton", background=start_color, foreground='#000000', font=entry_font, borderwidth=0, relief="flat", padding=10)
style.configure("Stop.TButton", background=stop_color, foreground='#000000', font=entry_font, borderwidth=0, relief="flat", padding=10)
style.configure("On.TButton", background=active_color, foreground=primary_color, font=entry_font, borderwidth=0, relief="flat", padding=10)  # Active state
style.configure("Off.TButton", background=off_color, foreground=primary_color, font=entry_font, borderwidth=0, relief="flat", padding=10)  # Inactive state

# Button highlight colors
style.map("On.TButton", background=[('active', active_color), ('!active', active_color)])  # Set background for active state
style.map("Off.TButton", background=[('active', off_color), ('!active', off_color)])  # Set background for inactive state

# Department Label
dept_label = ttk.Label(window, text="Department of Electrical & Electronics, Christ University SOET", 
                       font=('Helvetica Neue', 12), background="#2D2D2D", foreground=primary_color)
dept_label.grid(row=0, column=0, columnspan=4, pady=5, padx=10)

# Title Label
title_label = ttk.Label(window, text="Relay Model Tester", font=title_font, background="#2D2D2D", foreground=primary_color)
title_label.grid(row=1, column=0, columnspan=4, pady=5)

# Input fields with labels
input_labels = ["IR", "IY", "IB", "VR", "VY", "VB"]
entries = []

for i, label in enumerate(input_labels):
    ttk.Label(window, text=f"{label}:", style="TLabel").grid(row=i + 2, column=0, padx=10, pady=5, sticky='e')
    entry = ttk.Entry(window, font=entry_font, width=10, justify="center")
    entry.grid(row=i + 2, column=1, padx=10, pady=5, sticky='w')
    entries.append(entry)

entry_IR, entry_IY, entry_IB, entry_VR, entry_VY, entry_VB = entries

# Start/Stop button
start_stop_button = ttk.Button(window, text="Start", style="Start.TButton", command=toggle_prediction)
start_stop_button.grid(row=8, column=0, columnspan=2, pady=10)

# Create labels and buttons for L1 to L8 in a single row
outputs = []
buttons = []
for i in range(8):
    output_label = ttk.Label(window, text=f'L{i + 1}: Off', style="TLabel")
    output_label.grid(row=9, column=i, padx=5, pady=5)
    outputs.append(output_label)

    button = ttk.Button(window, style="Off.TButton", width=5)
    button.grid(row=10, column=i, padx=5, pady=5)
    buttons.append(button)

# Run the tkinter event loop
window.mainloop()
