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
from threading import Thread,Event
import google.generativeai as genai
from fastapi.responses import StreamingResponse
import asyncio


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

#config the genai
api_key= os.getenv("google_api_key")
genai.configure(api_key=api_key)

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

#store all user inputs
user_inputs: Dict = {}
#event to check if caching is ready
cache_ready_event = Event()
# Fetch table data from Supabase and return as DataFrame
def fetch_table_data(table_name, columns):
    response = supabase.table(table_name).select(columns).execute()
    return pd.DataFrame(response.data)

# Load all required tables concurrently from Supabase
def load_data_from_supabase():
    global cached_districts_df, cached_hotels_df, cached_restaurants_df, cached_tours_df
    #so that while being updated the data isn't copied in start quiz
    cache_ready_event.clear()
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
        cache_ready_event.set()

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
    #wait till the data is loaded and cached
    cache_ready_event.wait()
    #make copies of the cached data for the user
    districts_df = cached_districts_df.copy()
    hotels_df = cached_hotels_df.copy()
    restaurants_df =  cached_restaurants_df.copy()
    tours_df =  cached_tours_df.copy()
    return {
        "status": "success",
        "message": "Quiz data loaded",
        "data": {
            "districts": districts_df.to_dict(orient="records"),
            "hotels": hotels_df.to_dict(orient="records"),
            "restaurants": restaurants_df.to_dict(orient="records"),
            "tours": tours_df.to_dict(orient="records")
        }
    }

@app.post("/budget_filter")
def budget_filter(inputs: TripPreferencesRequest):
    global hotels_df, restaurants_df, tours_df,user_inputs

    user_inputs["nb_of_days"]=inputs.nb_of_days
    user_inputs["budget_min"]=inputs.budget_min
    user_inputs["budget_max"]=inputs.budget_max
    user_inputs["activity_counts"]=inputs.activity_counts

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

    user_inputs["activities_budgets"]={"hotel_budget_range":hotel_budget_range,"restaurant_budget_range":restaurant_budget_range
                                       ,"tour_budget_range":tour_budget_range}

    # Filters the given DataFrame by a price column using a budget range
    def filter_by_budget(df, price_column, budget_range):
        return df[df[price_column].between(*budget_range)]

    filtered_hotels_df = filter_by_budget(hotels_df, 'price_per_day', hotel_budget_range)
    filtered_restaurants_df = filter_by_budget(restaurants_df, 'avg_price', restaurant_budget_range)
    filtered_tours_df = filter_by_budget(tours_df, 'price', tour_budget_range)

    if filtered_hotels_df.empty:
        return{
            "status": "repeat",
            "message": "no hotels found, choose other answers"
        }
    elif filtered_restaurants_df.empty:
        return{
            "status": "repeat",
            "message": "no restaurants found, choose other answers"
        }
    elif filtered_tours_df.empty:
        return{
            "status": "repeat",
            "message": "no tours found, choose other answers"
        }
    else:
        hotels_df= filtered_hotels_df
        restaurants_df= filtered_restaurants_df
        tours_df= filtered_tours_df
        return {
        "status": "success",
        "message": "Activities filtered based on budget",
        "data": {
            "hotels": hotels_df.to_dict(orient="records"),
            "restaurants": restaurants_df.to_dict(orient="records"),
            "tours": tours_df.to_dict(orient="records")
        }
    }
    
@app.post("/tags_filter")
def tag_filter(inputs: UserTagsRequest):
    global districts_df,hotels_df,tours_df,restaurants_df,user_inputs

    user_inputs["tags"]=inputs.user_tags

    # Count how many tags in the user's list exist in the district's tags
    def count_sum(user_list, tags_list):
        return sum(1 for x in user_list if x in tags_list)
    # Filter districts that share at least 2 tags with the user
    filtered_districts_df = districts_df[districts_df["tags"].apply(
        lambda column_tags: count_sum(inputs.user_tags, column_tags) >= 2
    )]

    if filtered_districts_df.empty:
        return{
            "status":"repeat",
            "message":"no districts found, choose other answers"
        }
    else:
        districts = filtered_districts_df["district"].to_list()
        filtered_hotels_df = hotels_df[hotels_df["district"].isin(districts)]
        filtered_restaurants_df = restaurants_df[restaurants_df["district"].isin(districts)]
        filtered_tours_df = tours_df[tours_df["district"].isin(districts)]
        
        if filtered_hotels_df.empty:
            return{
                "status": "repeat",
                "message": "no hotels found, choose other tags"
            }
        elif filtered_restaurants_df.empty:
            return{
                "status": "repeat",
                "message": "no restaurants found, choose other tags"
            }
        elif filtered_tours_df.empty:
            return{
                "status": "repeat",
                "message": "no tours found, choose other tags"
            }
        else:
            districts_df = filtered_districts_df
            hotels_df= filtered_hotels_df
            restaurants_df= filtered_restaurants_df
            tours_df= filtered_tours_df
            return {
            "status": "success",
            "message": "districts filtered based on tags, all data filtered based on districts",
            "matched_districts_count": len(districts),
            "data":{
                "districts": districts_df.to_dict(orient="records"),
                "hotels": hotels_df.to_dict(orient="records"),
                "restaurants": restaurants_df.to_dict(orient="records"),
                "tours": tours_df.to_dict(orient="records")
            }
        }        

@app.post("/recommend")
def recommend(inputs: TravelInput):
    global top_districts
    global districts_df,hotels_df,restaurants_df,tours_df,user_inputs

    user_inputs["purpose"]=inputs.purpose
    user_inputs["interests"]=inputs.interests
    user_inputs["weather"]=inputs.weather

    # Build a sentence based on user input
    user_sentence = f"I want to travel for {inputs.purpose}. I am interested in {inputs.interests}. I prefer {inputs.weather} weather."

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

    filtered_hotels_df = hotels_df[hotels_df["district"].isin(top_districts)]
    filtered_restaurants_df = restaurants_df[restaurants_df["district"].isin(top_districts)]
    filtered_tours_df = tours_df[tours_df["district"].isin(top_districts)]
    
    if filtered_hotels_df.empty:
        return{
            "status": "repeat",
            "message": "no hotels found, choose other answers"
        }
    elif filtered_restaurants_df.empty:
        return{
            "status": "repeat",
            "message": "no restaurants found, choose other answers"
        }
    elif filtered_tours_df.empty:
        return{
            "status": "repeat",
            "message": "no tours found, choose other answers"
        }
    else:
        districts_df= districts_df[districts_df["district"].isin(top_districts)]
        hotels_df= filtered_hotels_df
        restaurants_df= filtered_restaurants_df
        tours_df= filtered_tours_df
        return {
        "status": "success",
        "message": "Top travel districts recommendations generated, all data filtered based on recommended districts",
        "recommendation": [
            {"district":district, "score":round(score,4)}
            for district, score in districts_scores[:6]
        ],
        "data":{
            "districts": districts_df.to_dict(orient="records"),
            "hotels": hotels_df.to_dict(orient="records"),
            "restaurants": restaurants_df.to_dict(orient="records"),
            "tours": tours_df.to_dict(orient="records")
        }
    }

generative_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

async def chatbot_output_streamer():
    global user_inputs
    # Build the structured prompt
    prompt = f"""
    You are a travel planning AI. The user has given their trip preferences and we have filtered travel data for them. 
    Your task is to generate **3 complete itineraries**, each covering the full trip duration. 
    For each itinerary:
    - Pick exactly **one district** (different for each itinerary) from the provided data.
    - Pick exactly **one hotel** from that district for the entire stay.
    - Select restaurants and tours **only from that same district**.
    - Use the activity counts and budgets to decide how many restaurants/tours THE user will go to, and how much to spend daily.
    - The number of days is {user_inputs.get("nb_of_days")}.
    - The restaurants/tours must be spread across the days according to the activity_counts: {user_inputs.get("activity_counts")}.
    - Each activity must respect the given daily budget ranges: {user_inputs.get("activities_budgets")}.
    - Districts also have tags; consider both the tags and descriptions when matching the user’s preferences.
    - The user tags are: {user_inputs.get("tags")}.
    - User purpose: {user_inputs.get("purpose")}.
    - User interests: {user_inputs.get("interests")}.
    - Preferred weather: {user_inputs.get("weather")}.

    **Available Data** (pandas DataFrames converted to JSON):

    DISTRICTS:
    {districts_df.to_dict(orient="records")}

    HOTELS:
    {hotels_df.to_dict(orient="records")}

    RESTAURANTS:
    {restaurants_df.to_dict(orient="records")}

    TOURS:
    {tours_df.to_dict(orient="records")}

    **Rules:**
    1. One district per itinerary, hotel/restaurants/tours all from that district.
    2. Respect the activity counts (e.g., if restaurants=2 in activity_counts and nb_of_days=4, schedule restaurant visits for exactly 2 different days).
    3. Ensure each chosen restaurant/tour fits in its budget range for its category.
    4. Use variety: do not pick the same restaurant or tour twice in the same itinerary.
    5. The output must be **detailed day-by-day** with all activities.

    Return your answer as **3 separate itineraries**, each clearly labeled and formatted for easy reading.
    """

    # Stream the generation
    for chunk in generative_model.generate_content(prompt, stream=True):
        if chunk.text:
            yield chunk.text.replace("*", "") + "\n"


@app.get("/generate")
async def generate():
    return StreamingResponse(chatbot_output_streamer(),media_type="text/plain")