from math import floor
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel
from typing import Dict
from supabase import create_client
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from time import sleep
from threading import Thread

#initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')

#load supabase credentials
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url,key)

#declare cached dataframes for automatic caching
cached_districts_df = pd.DataFrame()
cached_hotels_df = pd.DataFrame()
cached_restaurants_df = pd.DataFrame()
cached_tours_df = pd.DataFrame()

#declare data frames
districts_df = pd.DataFrame()
hotels_df = pd.DataFrame()
restaurants_df = pd.DataFrame()
tours_df = pd.DataFrame()

#define data models
class TravelInput (BaseModel):
    purpose: str 
    interests: str
    weather: str

class TripPreferencesRequest (BaseModel):
    nb_of_days: int                          
    budget_min: int
    budget_max: int
    activity_counts: Dict[str, Optional[int]]  #ex: {"hotel":3,"restaurant":2,"tour":3}, how many times go for each        
    budget_percentages: Dict[str, float] #percentage of budget given for each activity 

class UserTagsRequest (BaseModel):
    user_tags: list 

# Global variable to store top districts
top_districts: List[str] = []

# Fetch table data from Supabase and return as DataFrame
def fetch_table_data(table_name, columns):
    response = supabase.table(table_name).select(columns).execute()
    return pd.DataFrame(response.data)

# Load all required tables concurrently from Supabase
def load_data_from_supabase():
    global cached_districts_df, cached_hotels_df, cached_restaurants_df, cached_tours_df

    # Use threads to fetch all tables in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            "districts": executor.submit(fetch_table_data, "districts", "country,district,description,tags,embedding"),
            "hotels": executor.submit(fetch_table_data, "hotels", "name,country,district,price_per_day,rating,description,facilities"),
            "restaurants": executor.submit(fetch_table_data, "restaurants", "title,country,district,avg_price,rating,description,tag"),
            "tours": executor.submit(fetch_table_data, "tours", "title,country,district,duration,price,description")
        }

        # Store results in global cached variables
        cached_districts_df = futures["districts"].result()
        cached_hotels_df = futures["hotels"].result()
        cached_restaurants_df = futures["restaurants"].result()
        cached_tours_df = futures["tours"].result()

RELOAD_INTERVAL_MINUTES = 30 
def reload_function():
    while True:
        load_data_from_supabase()
        sleep(RELOAD_INTERVAL_MINUTES*60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    #start background thread at startup
    thread = Thread(target=reload_function,daemon=True)
    thread.start()
    yield


#initialize app
app = FastAPI(lifespan=lifespan)

# API endpoint to trigger quiz data loading
@app.post("/start-quiz")
def start_quiz():
    global districts_df, hotels_df, restaurants_df, tours_df
    districts_df = cached_districts_df.copy()
    hotels_df = cached_hotels_df.copy()
    restaurants_df =  cached_restaurants_df.copy()
    tours_df =  cached_tours_df.copy()
    return {
        "status":"success",
        "message": "Quiz data loaded",
        "data": {
            "districts": districts_df.to_dict(orient="records"),
            "hotels": hotels_df.to_dict(orient="records"),
            "restaurants": restaurants_df.to_dict(orient="records"),
            "tours": tours_df.to_dict(orient="records"),
        }
        }

@app.post("/budget_filter")
def budget_filter(inputs: TripPreferencesRequest):
    global hotels_df, restaurants_df, tours_df

    total_budget_min = inputs.budget_min
    total_budget_max = inputs.budget_max

    # Assume the user needs hotel for every day of the trip
    inputs.activity_counts["hotel"] = inputs.nb_of_days

    # Calculates min/max budget per activity item (e.g., per hotel/night, per restaurant meal, etc.)
    def budget_range(category: str):
        perc = inputs.budget_percentages.get(category, 0)
        count = inputs.activity_counts.get(category, 1)
        count = max(count,1)  
        min_budget = floor((total_budget_min * perc / 100) / count)
        max_budget = floor((total_budget_max * perc / 100) / count)
        return (min_budget, max_budget)

    hotel_budget_range = budget_range('hotel')
    restaurant_budget_range = budget_range('restaurant')
    tour_budget_range = budget_range('tour')

    # Filters the given DataFrame by a price column using a budget range
    def filter_by_budget(df, price_column, budget_range):
        return df[df[price_column].between(*budget_range)]

    hotels_df = filter_by_budget(hotels_df, 'price_per_day', hotel_budget_range)
    restaurants_df = filter_by_budget(restaurants_df, 'avg_price', restaurant_budget_range)
    tours_df = filter_by_budget(tours_df, 'price', tour_budget_range)

    return {
        "status": "success",
        "message": "Activities filtered based on budget",
        "data": {
            "hotels": hotels_df.to_dict(orient="records"),
            "restaurants": restaurants_df.to_dict(orient="records"),
            "tours": tours_df.to_dict(orient="records"),
        }
    }

@app.post("/tags_filter")
def tag_filter(inputs: UserTagsRequest):
    global districts_df

    # Count how many tags in the user's list exist in the district's tags
    def count_sum(user_list, tags_list):
        return sum(1 for x in user_list if x in tags_list)

    # Filter districts that share at least 2 tags with the user
    districts_df = districts_df[districts_df["tags"].apply(
        lambda column_tags: count_sum(inputs.user_tags, column_tags) >= 2
    )]

    return {
        "status": "success",
        "message": "districts filtered based on tags",
        "matched_districts_count": len(districts_df),
        "matched_districts": districts_df.to_dict(orient="records")
    }

@app.post("/recommend")
def recommend(input: TravelInput):
    global top_districts
    global districts_df

    # Build a sentence based on user input
    user_sentence = f"I want to travel for {input.purpose}. I am interested in {input.interests}. I prefer {input.weather} weather."

    # Get the embedding vector for the user's sentence
    user_embedding = model.encode(user_sentence)

    districts_scores = []

    # Loop over all districts and compute similarity with user preferences
    for district, embedding in zip(districts_df["district"], districts_df["embedding"]):
        score = cosine_similarity([user_embedding], [embedding])[0][0]
        districts_scores.append((district, score))

    # Sort districts by similarity score in descending order
    districts_scores.sort(key=lambda x: x[1], reverse=True)

    # Extract top 6 districts
    top_districts = [d[0] for d in districts_scores[:6]]

    # Return top districts with their similarity scores
    return {
        "status":"success",
        "message": "Top travel districts recommendations generated",
        "recommendations": [
            {"district": district, "score": round(score, 4)}
            for district, score in districts_scores[:6]
        ]
    }

