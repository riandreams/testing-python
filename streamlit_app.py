from flask import Flask, request, jsonify, send_file
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import os

app = Flask(__name__)

# Load the CSV file with names and genders
df = pd.read_csv('indonesian-names.csv')

# Assume your CSV has two columns: 'name' and 'gender'
names = df['name']
genders = df['gender']

# Convert names to character n-grams (e.g., ngram_range=(1, 3))
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))  # 'char' n-gram for better name recognition
X = vectorizer.fit_transform(names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, genders, test_size=0.2, random_state=42)

# Create a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Function to predict gender from a list of names
def predict_gender_from_names(name_list):
    results = []
    
    # For each name, make a prediction using the trained classifier
    for name in name_list:
        if not name or not isinstance(name, str):
            # If the name is None, empty, or not a string, assign 'Unknown'
            results.append({"name": name, "predicted_gender": "unknown"})
            continue
        
        name = name.strip()  # Remove extra spaces and newlines
        X_new = vectorizer.transform([name])  # Transform name into n-grams
        predicted_gender = classifier.predict(X_new)[0]
        
        # Map prediction results to 'male', 'female', or 'Unknown'
        if predicted_gender == 'm':
            gender_result = 'male'
        elif predicted_gender == 'f':
            gender_result = 'female'
        else:
            gender_result = 'unknown'
        
        # Append the result directly in JSON format
        results.append({"name": name, "predicted_gender": gender_result})
    
    return results

# Endpoint to upload the file, predict gender, and return a downloadable JSON file
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read the uploaded JSON file
    try:
        file_contents = file.read().decode('utf-8')
        names_data = json.loads(file_contents)  # Load JSON content
    except Exception as e:
        return jsonify({"error": f"Failed to process JSON file: {str(e)}"}), 400

    if 'names' not in names_data:
        return jsonify({"error": "'names' key not found in JSON file"}), 400
    
    name_list = names_data['names']
    
    # Predict genders from the names in the JSON file
    predictions = predict_gender_from_names(name_list)

    # Write predictions to a JSON file
    output_file = 'predicted_genders.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    # Send the file back to the client
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
