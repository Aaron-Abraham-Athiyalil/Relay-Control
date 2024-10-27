import joblib
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Function to load the multi-output model and make predictions based on user input
def test_multi_output_model(IR, IY, IB, VR, VY, VB):
    # Load the multi-output model
    model = joblib.load('multi_output_model.joblib')
    
    # Prepare input as a 2D array for prediction
    input_data = np.array([[IR, IY, IB, VR, VY, VB]])

    # Make predictions for all relays at once
    predictions = model.predict(input_data)[0]  # Get predictions for each relay

    # Display predictions in a message box
    prediction_text = "\n".join([f'Prediction for L{i+1}: {predictions[i]}' for i in range(len(predictions))])
    messagebox.showinfo("Relay Predictions", prediction_text)

# Function to retrieve input values and call the prediction function
def predict():
    try:
        # Get values from entry fields and convert to floats
        IR = float(entry_IR.get())
        IY = float(entry_IY.get())
        IB = float(entry_IB.get())
        VR = float(entry_VR.get())
        VY = float(entry_VY.get())
        VB = float(entry_VB.get())
        
        # Call the prediction function
        test_multi_output_model(IR, IY, IB, VR, VY, VB)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# Create a tkinter window
window = tk.Tk()
window.title("Relay Model Tester")

# Create and place input fields and labels for each value
tk.Label(window, text="Enter IR value:").grid(row=0, column=0)
entry_IR = tk.Entry(window)
entry_IR.grid(row=0, column=1)

tk.Label(window, text="Enter IY value:").grid(row=1, column=0)
entry_IY = tk.Entry(window)
entry_IY.grid(row=1, column=1)

tk.Label(window, text="Enter IB value:").grid(row=2, column=0)
entry_IB = tk.Entry(window)
entry_IB.grid(row=2, column=1)

tk.Label(window, text="Enter VR value:").grid(row=3, column=0)
entry_VR = tk.Entry(window)
entry_VR.grid(row=3, column=1)

tk.Label(window, text="Enter VY value:").grid(row=4, column=0)
entry_VY = tk.Entry(window)
entry_VY.grid(row=4, column=1)

tk.Label(window, text="Enter VB value:").grid(row=5, column=0)
entry_VB = tk.Entry(window)
entry_VB.grid(row=5, column=1)

# Create and place a button to trigger the prediction
predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=6, column=0, columnspan=2)

# Start the tkinter event loop
window.mainloop()
