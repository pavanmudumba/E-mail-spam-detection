import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
df = pd.read_csv("mail_data.csv")
data = df.where(pd.notnull(df), '')

# Convert 'Category' to binary values
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Define features and labels
X = data['Message']
Y = data['Category']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Initialize TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data
X_train_features = feature_extraction.fit_transform(X_train)

# Transform the test data
X_test_features = feature_extraction.transform(X_test)

# Convert labels to integer type
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Create the Tkinter app
app = tk.Tk()
app.title("Email Spam Detection")
app.geometry("400x300")

# Function to predict if an email is spam or ham
def predict_email():
    email_content = email_text.get("1.0", tk.END).strip()
    if not email_content:
        messagebox.showwarning("Input Error", "Please enter email content.")
        return
    
    # Transform the input email content and predict
    input_data_features = feature_extraction.transform([email_content])
    prediction = model.predict(input_data_features)

    # Display the result
    result = "Ham" if prediction[0] == 1 else "Spam"
    result_label.config(text=f"Result: {result}")

# Create UI elements
tk.Label(app, text="Enter Email Content:", font=("Arial", 12)).pack(pady=10)
email_text = tk.Text(app, height=8, width=40)
email_text.pack(pady=10)

check_button = tk.Button(app, text="Check if Spam", command=predict_email)
check_button.pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run the Tkinter app
app.mainloop()
