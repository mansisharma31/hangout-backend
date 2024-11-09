from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from flask_cors import CORS
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model_training.recommendation_test import generate_itineraries_with_age_group
# import sys
# import os

# # Add the parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_training')))

# # Now you can import the module
#from recommendation_test import generate_itineraries_with_age_group

# Initialize Flask app
app = Flask(__name__)
#CORS(app)

CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST"], "allow_headers": ["Content-Type"]}})

# Load the model and other necessary objects
model = tf.keras.models.load_model('../model_training/models/content_model.h5')
encoder_location = joblib.load('../model_training/models/encoder_location.pkl')
encoder_category = joblib.load('../model_training/models/encoder_category.pkl')
scaler = joblib.load('../model_training/models/scaler.pkl')
df = pd.read_csv('../model_training/places_data.csv')

# Your recommendation function (same as before)
# def generate_itineraries_with_age_group(location, budget, duration, age_group, categories=None, top_n=3):
#     # Encode location and category
#     encoded_location = encoder_location.transform([[location]])
#     encoded_category = encoder_category.transform([[categories]]) if categories else np.zeros((1, encoder_category.categories_[0].size))
    
#     # Normalize budget and duration
#     normalized_price_duration = scaler.transform([[budget, duration]])
    
#     # Combine all features for user input
#     user_features = np.hstack([encoded_location, encoded_category, normalized_price_duration])

#     # Predict using the loaded model
#     predictions = model.predict(user_features)[0]

#     # Filter places based on budget, duration, location, and age group
#     filtered_places = df[
#         (df['price'] <= budget) &
#         (df['duration'] <= duration) &
#         (df['location'] == location) &
#         ((df['category'] == categories) if categories else True)  # Category is optional
#     ]
    
#     # Filter places by age group
#     filtered_places = filtered_places[filtered_places['age'].apply(
#         lambda x: any(group in x for group in age_group))]

#     # Separate dining places from the rest
#     dining_places = filtered_places[filtered_places['categories'] == 'dining']
#     other_places = filtered_places[filtered_places['categories'] != 'dining']
    
#     # Check if a dining place is needed but unavailable
#     if duration >= 2 and dining_places.empty:
#         return {"error": "No dining places available for the specified duration and budget."}

#     all_itineraries = []

#     # Generate possible itineraries by including a dining place
#     for _, dining_place in dining_places.iterrows():
#         current_duration = dining_place['duration']
#         current_budget = dining_place['price']
#         selected_places = [dining_place]
        
#         remaining_duration = duration - current_duration
#         remaining_budget = budget - current_budget
        
#         # Loop over other places and fill the remaining duration and budget
#         for _, place in other_places.iterrows():
#             if current_duration + place['duration'] <= duration and current_budget + place['price'] <= budget:
#                 selected_places.append(place)
#                 current_duration += place['duration']
#                 current_budget += place['price']
            
#             if current_duration >= duration or current_budget >= budget:
#                 break
        
#         # If itinerary is valid, add it to all_itineraries
#         if selected_places:
#             all_itineraries.append(selected_places)
    
#     # Sort itineraries by model predictions
#     itineraries_with_scores = []
#     for itinerary in all_itineraries:
#         itinerary_score = sum(predictions[df[df['id'] == place['id']].index[0]] for place in itinerary)
#         itineraries_with_scores.append((itinerary, itinerary_score))
    
#     # Sort itineraries by score and return the top N
#     itineraries_with_scores.sort(key=lambda x: x[1], reverse=True)

#     # Structure itineraries for JSON response
#     recommendations = []
#     for itinerary, score in itineraries_with_scores[:top_n]:
#         itinerary_details = {
#             "score": score,
#             "places": [
#                 {
#                     "name": place["name"],
#                     "categories": place["categories"],
#                     "duration": place["duration"],
#                     "price": place["price"]
#                 }
#                 for _, place in itinerary
#             ],
#             "total_duration": sum(place["duration"] for _, place in itinerary),
#             "total_price": sum(place["price"] for _, place in itinerary)
#         }
#         recommendations.append(itinerary_details)
    
#     return recommendations
  # E.g., [{'score': 9.5, 'places': [...]}, ...]

# Define the API route
@app.route('/api/getRecommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()  # Get JSON data from request
    location = data.get('location')
    budget = float(data.get('budget'))
    duration = float(data.get('duration'))
    age_group = data.get('ageGroup')
    categories = data.get('categories', None)  # Optional
    print("backend")

    # Get recommendations
    recommendations = generate_itineraries_with_age_group(
        location, budget, duration, age_group, categories, top_n=3
    )

    # Send recommendations as JSON response
    return jsonify({"data": recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=3000)




# from flask import Flask, request, jsonify
# import sys
# import os

# # Add the parent directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_training')))

# # Now you can import the module
# from recommendation_test import generate_itineraries_with_age_group


# app = Flask(__name__)

# @app.route('/api/recommendations', methods=['POST'])
# def get_recommendations():
#     data = request.get_json()  # Get data from frontend form submission
    
#     location = data.get('location')
#     budget = float(data.get('budget'))
#     duration = float(data.get('duration'))
#     age_group = data.get('ageGroup').split(',')  # Assuming multiple age groups are separated by commas
    
#     # Generate recommendations using the imported function
#     itineraries = generate_itineraries_with_age_group(location, budget, duration, age_group)

#     # If itineraries are generated, return them as a response
#     if itineraries:
#         return jsonify(itineraries), 200
#     else:
#         return jsonify({"error": "No recommendations found"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # import numpy as np
# # import pandas as pd
# # import joblib

# # app = Flask(__name__)

# # # Load models and preprocessors
# # model = tf.keras.models.load_model('models/content_model.h5')
# # encoder_location = joblib.load('models/encoder_location.pkl')
# # encoder_category = joblib.load('models/encoder_category.pkl')
# # scaler = joblib.load('models/scaler.pkl')
# # df = pd.read_csv('places_data.csv')

# # # Generate itineraries function
# # def generate_itineraries_with_age_group(location, budget, duration, age_groups, category=None, top_n=3):
# #     # Encode and normalize input data, filter places, and rank itineraries...
# #     # [existing logic]

# #     # Prepare JSON-compatible result data
# #     result = []
# #     for itinerary, score in itineraries_with_scores[:top_n]:
# #         places = [{
# #             'name': place['name'],
# #             'category': place['category'],
# #             'duration': place['duration'],
# #             'price': place['price']
# #         } for place in itinerary]
# #         total_duration = sum(place['duration'] for place in itinerary)
# #         total_price = sum(place['price'] for place in itinerary)
# #         result.append({
# #             'places': places,
# #             'total_duration': total_duration,
# #             'total_price': total_price
# #         })
    
# #     return result

# # @app.route('/recommendations', methods=['POST'])
# # def recommendations():
# #     data = request.get_json()
# #     location = data['location']
# #     category = data['category']
# #     age_groups = data['age_groups']
# #     budget = float(data['budget'])
# #     duration = float(data['duration'])

# #     itineraries = generate_itineraries_with_age_group(location, budget, duration, age_groups, category, top_n=3)
    
# #     return jsonify({'itineraries': itineraries})

# # if __name__ == "__main__":
# #     app.run(debug=True)



# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # import numpy as np
# # import pandas as pd
# # import joblib

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load model and preprocessing objects
# # model = tf.keras.models.load_model('models/content_model.h5')
# # encoder_location = joblib.load('models/encoder_location.pkl')
# # encoder_category = joblib.load('models/encoder_category.pkl')
# # scaler = joblib.load('models/scaler.pkl')

# # # Load places data
# # df = pd.read_csv('places_data.csv')

# # # ID-to-label mapping from content_based_model.py
# # id_to_label = {item_id: idx for idx, item_id in enumerate(sorted(df['id'].unique()))}
# # label_to_id = {idx: item_id for item_id, idx in id_to_label.items()}

# # # Function to generate itineraries
# # def generate_itineraries_with_age_group(location, budget, duration, age_groups, category=None, top_n=3):
# #     # Transform location based on user input
# #     encoded_location = encoder_location.transform([[location]])
# #     encoded_category = encoder_category.transform([[category]]) if category else np.zeros((1, encoder_category.categories_[0].size))
    
# #     # Normalize budget and duration
# #     normalized_price_duration = scaler.transform([[budget, duration]])
    
# #     # Combine all features for user input
# #     user_features = np.hstack([encoded_location, encoded_category, normalized_price_duration])

# #     # Predict using the loaded model
# #     predictions = model.predict(user_features)[0]

# #     # Filter places based on budget, duration, location, and age group
# #     filtered_places = df[
# #         (df['price'] <= budget) &
# #         (df['duration'] <= duration) &
# #         (df['location'] == location) &
# #         ((df['category'] == category) if category else True)
# #     ]
    
# #     # Filter places by age group
# #     filtered_places = filtered_places[filtered_places['age'].apply(
# #         lambda x: any(group in x for group in age_groups))]

# #     # Separate dining places from the rest
# #     dining_places = filtered_places[filtered_places['category'] == 'dining']
# #     other_places = filtered_places[filtered_places['category'] != 'dining']
    
# #     # Ensure at least one dining place is included if duration >= 2 hours
# #     if duration >= 2 and dining_places.empty:
# #         return []

# #     all_itineraries = []

# #     # Generate possible itineraries by including a dining place
# #     for _, dining_place in dining_places.iterrows():
# #         current_duration = dining_place['duration']
# #         current_budget = dining_place['price']
# #         selected_places = [dining_place]
        
# #         remaining_duration = duration - current_duration
# #         remaining_budget = budget - current_budget
        
# #         # Loop over other places and fill the remaining duration and budget
# #         for _, place in other_places.iterrows():
# #             if current_duration + place['duration'] <= duration and current_budget + place['price'] <= budget:
# #                 selected_places.append(place)
# #                 current_duration += place['duration']
# #                 current_budget += place['price']
            
# #             if current_duration >= duration or current_budget >= budget:
# #                 break
        
# #         if current_duration <= duration and current_budget <= budget and selected_places:
# #             all_itineraries.append(selected_places)
    
# #     # Sort itineraries by model predictions
# #     itineraries_with_scores = []
# #     for itinerary in all_itineraries:
# #         itinerary_score = sum(predictions[df[df['id'] == place['id']].index[0]] for place in itinerary)
# #         itineraries_with_scores.append((itinerary, itinerary_score))
    
# #     # Sort itineraries by score and return the top N
# #     itineraries_with_scores.sort(key=lambda x: x[1], reverse=True)
    
# #     # Prepare the output data
# #     result = []
# #     for itinerary, score in itineraries_with_scores[:top_n]:
# #         places = [{
# #             'name': place['name'],
# #             'category': place['category'],
# #             'duration': place['duration'],
# #             'price': place['price']
# #         } for place in itinerary]
# #         total_duration = sum(place['duration'] for place in itinerary)
# #         total_price = sum(place['price'] for place in itinerary)
# #         result.append({
# #             'places': places,
# #             'total_duration': total_duration,
# #             'total_price': total_price
# #         })
    
# #     return result

# # @app.route('/recommendations', methods=['POST'])
# # def recommendations():
# #     data = request.get_json()
    
# #     location = data['location']
# #     category = data['category']
# #     age_groups = data['age_groups']
# #     budget = float(data['budget'])
# #     duration = float(data['duration'])

# #     itineraries = generate_itineraries_with_age_group(location, budget, duration, age_groups, category, top_n=3)
    
# #     return jsonify({'itineraries': itineraries})

# # if __name__ == "__main__":
# #     app.run(debug=True)




# # from flask import Flask, request, jsonify
# # from tensorflow.keras.models import load_model
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # import pickle  # To load your trained model


# # app = Flask(__name__)

# # # Load your trained model
# # model = load_model('../model_training/models/content_model.h5')
# # #model = pickle.load(open('model_training/models/content_model.h5', 'rb'))  # Assuming your model is saved as 'model.pkl'
# # encoder_location = pickle.load(open('../model_training/models/encoder_location.pkl', 'rb'))
# # encoder_category = pickle.load(open('../model_training/models/encoder_category.pkl', 'rb'))
# # scaler = pickle.load(open('../model_training/models/scaler.pkl', 'rb'))

# # # Sample data, replace this with your actual DataFrame
# # df = pd.read_csv("../model_training/places_data.csv")  # Replace with actual data

# # @app.route("/api/recommend", methods=["POST"])
# # def recommend():
# #     data = request.get_json()

# #     # Extract input data from frontend
# #     location = data.get("location")
# #     budget = int(data.get("budget"))
# #     duration = int(data.get("duration"))
# #     age_groups = data.get("age_groups")
# #     category = data.get("category", None)  # Optional parameter

# #     # Preprocess input for model
# #     encoded_location = encoder_location.transform([[location]])
# #     encoded_category = encoder_category.transform([[category]]) if category else np.zeros((1, encoder_category.categories_[0].size))
# #     normalized_price_duration = scaler.transform([[budget, duration]])
# #     user_features = np.hstack([encoded_location, encoded_category, normalized_price_duration])

# #     # Make predictions
# #     predictions = model.predict(user_features)[0]

# #     # Filter places based on budget, duration, and age group
# #     filtered_places = df[
# #         (df['price'] <= budget) &
# #         (df['duration'] <= duration) &
# #         (df['location'] == location) &
# #         ((df['category'] == category) if category else True)
# #     ]

# #     # Filter by age group
# #     filtered_places = filtered_places[filtered_places['age'].apply(
# #         lambda x: any(group in x for group in age_groups))]

# #     # Generate the best itinerary (same logic you had earlier)
# #     best_combination = find_best_combination(filtered_places, budget, duration)

# #     # Return the response to the frontend
# #     return jsonify({"recommendations": best_combination})

# # def find_best_combination(filtered_places, budget, duration):
# #     # Use your backtracking or any combination algorithm to find the best places
# #     best_combination = []
# #     # Your backtracking logic here
# #     return best_combination

# # if __name__ == "__main__":
# #     app.run(debug=True)

