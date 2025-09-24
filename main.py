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
from time import sleep, time
from threading import Thread,Event
import google.generativeai as genai
from fastapi.responses import StreamingResponse
import asyncio
import time

#initialize embedding model
model = SentenceTransformer('model/trained_model')

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

class QuizSession:
    def __init__(self, session_id: str):
        import copy
        self.session_id = session_id
        # Copy the cached dataframes internally
        self.districts_df = copy.deepcopy(cached_districts_df)
        self.hotels_df = copy.deepcopy(cached_hotels_df)
        self.restaurants_df = copy.deepcopy(cached_restaurants_df)
        self.tours_df = copy.deepcopy(cached_tours_df)
        # Per-session variables
        self.top_districts: List[str] = []
        self.user_inputs: Dict = {}
        self.last_active = time.time()

sessions: Dict[str, QuizSession] = {}

def get_session(session_id: str) -> QuizSession:
    session = sessions.get(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")
    session.last_active = time.time()  # update last_active timestamp
    return session

#config the genai
api_key= os.getenv("google_api_key")
genai.configure(api_key=api_key)

class SessionRequest(BaseModel):
    session_id: str

#define data models
class TravelInput (BaseModel):
    session_id: str
    purpose: str 
    interests: str
    weather: str

class TripPreferencesRequest (BaseModel):
    session_id: str
    nb_of_days: int                          
    budget_min: int
    budget_max: int
    activity_counts: Dict[str, Optional[int]]  #ex: {"hotel":3,"restaurant":2,"tour":3}, how many times go for each        
    budget_percentages: Dict[str, float] #percentage of budget given for each activity 

class UserTagsRequest (BaseModel):
    session_id: str
    user_tags: list 

class GenerateRequest(BaseModel):
    session_id: str

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
from fastapi import Query

@app.post("/start-quiz")
def start_quiz(request: SessionRequest):
    session_id = request.session_id

    cache_ready_event.wait()  # wait for cached data

    # create a new session object
    session = QuizSession(session_id=session_id)
    sessions[session_id] = session

    return {
        "status": "success",
        "message": "Quiz session started",
        "session_id": session_id
    }


@app.post("/budget_filter")
def budget_filter(inputs: TripPreferencesRequest):
    session = get_session(inputs.session_id)
    session.user_inputs["nb_of_days"] = inputs.nb_of_days
    session.user_inputs["budget_min"] = inputs.budget_min
    session.user_inputs["budget_max"] = inputs.budget_max
    session.user_inputs["activity_counts"] = inputs.activity_counts

    total_budget_min = inputs.budget_min
    total_budget_max = inputs.budget_max

    inputs.activity_counts["hotel"] = inputs.nb_of_days

    def budget_range(category: str):
        perc = inputs.budget_percentages.get(category, 0)
        count = max(inputs.activity_counts.get(category, 1), 1)
        return floor((total_budget_min * perc / 100) / count), floor((total_budget_max * perc / 100) / count)

    hotel_budget_range = budget_range('hotel')
    restaurant_budget_range = budget_range('restaurant')
    tour_budget_range = budget_range('tour')

    session.user_inputs["activities_budgets"] = {
        "hotel_budget_range": hotel_budget_range,
        "restaurant_budget_range": restaurant_budget_range,
        "tour_budget_range": tour_budget_range
    }

    def filter_by_budget(df, price_column, budget_range):
        return df[df[price_column].between(*budget_range)]

    filtered_hotels_df = filter_by_budget(session.hotels_df, 'price_per_day', hotel_budget_range)
    filtered_restaurants_df = filter_by_budget(session.restaurants_df, 'avg_price', restaurant_budget_range)
    filtered_tours_df = filter_by_budget(session.tours_df, 'price', tour_budget_range)

    if filtered_hotels_df.empty:
        return {"status": "repeat", "message": "No hotels found for the given budget"}
    elif filtered_restaurants_df.empty:
        return {"status": "repeat", "message": "No restaurants found for the given budget"}
    elif filtered_tours_df.empty:
        return {"status": "repeat", "message": "No tours found for the given budget"}

    
    session.hotels_df = filtered_hotels_df
    session.restaurants_df = filtered_restaurants_df
    session.tours_df = filtered_tours_df

    return {
        "status": "success",
        "message": "Activities filtered based on budget",
        "data": {
            "hotels": session.hotels_df.to_dict(orient="records"),
            "restaurants": session.restaurants_df.to_dict(orient="records"),
            "tours": session.tours_df.to_dict(orient="records")
        }
    }
    
@app.post("/tags_filter")
def tag_filter(inputs: UserTagsRequest):
    # get the session object
    session = get_session(inputs.session_id)

    # store user tags in session
    session.user_inputs["tags"] = inputs.user_tags

    # helper function to count matching tags
    def count_sum(user_list, tags_list):
        return sum(1 for x in user_list if x in tags_list)

    # filter districts that share at least 2 tags with the user
    filtered_districts_df = session.districts_df[
        session.districts_df["tags"].apply(lambda column_tags: count_sum(inputs.user_tags, column_tags) >= 2)
    ]

    if filtered_districts_df.empty:
        return {
            "status": "repeat",
            "message": "no districts found, choose other answers"
        }

    # get matching districts
    districts = filtered_districts_df["district"].to_list()

    # filter other dataframes based on matched districts
    filtered_hotels_df = session.hotels_df[session.hotels_df["district"].isin(districts)]
    filtered_restaurants_df = session.restaurants_df[session.restaurants_df["district"].isin(districts)]
    filtered_tours_df = session.tours_df[session.tours_df["district"].isin(districts)]

    # check for empty results
    if filtered_hotels_df.empty:
        return {"status": "repeat", "message": "no hotels found, choose other tags"}
    if filtered_restaurants_df.empty:
        return {"status": "repeat", "message": "no restaurants found, choose other tags"}
    if filtered_tours_df.empty:
        return {"status": "repeat", "message": "no tours found, choose other tags"}

    # update session with filtered data
    session.districts_df = filtered_districts_df
    session.hotels_df = filtered_hotels_df
    session.restaurants_df = filtered_restaurants_df
    session.tours_df = filtered_tours_df

    return {
        "status": "success",
        "message": "districts filtered based on tags, all data filtered based on districts",
        "matched_districts_count": len(districts),
        "data": {
            "districts": session.districts_df.to_dict(orient="records"),
            "hotels": session.hotels_df.to_dict(orient="records"),
            "restaurants": session.restaurants_df.to_dict(orient="records"),
            "tours": session.tours_df.to_dict(orient="records")
        }
    }

@app.post("/recommend")
def recommend(inputs: TravelInput):
    # get session object
    session = get_session(inputs.session_id)

    # store user inputs in the session
    session.user_inputs["purpose"] = inputs.purpose
    session.user_inputs["interests"] = inputs.interests
    session.user_inputs["weather"] = inputs.weather

    # build a sentence based on user input
    user_sentence = f"I want to travel for {inputs.purpose}. I am interested in {inputs.interests}. I prefer {inputs.weather} weather."

    # get the embedding vector for the user's sentence
    user_embedding = model.encode(user_sentence)
    districts_scores = []

    # loop over all districts and compute similarity
    for district, embedding in zip(session.districts_df["district"], session.districts_df["embedding"]):
        score = cosine_similarity([user_embedding], [embedding])[0][0]
        districts_scores.append((district, score))

    # sort districts by similarity
    districts_scores.sort(key=lambda x: x[1], reverse=True)

    # extract top 6 districts
    session.top_districts = [d[0] for d in districts_scores[:6]]

    # filter other dataframes based on top districts
    filtered_hotels_df = session.hotels_df[session.hotels_df["district"].isin(session.top_districts)]
    filtered_restaurants_df = session.restaurants_df[session.restaurants_df["district"].isin(session.top_districts)]
    filtered_tours_df = session.tours_df[session.tours_df["district"].isin(session.top_districts)]

    # check each category separately
    if filtered_hotels_df.empty:
        return {"status": "repeat", "message": "No hotels found in top districts"}
    elif filtered_restaurants_df.empty:
        return {"status": "repeat", "message": "No restaurants found in top districts"}
    elif filtered_tours_df.empty:
        return {"status": "repeat", "message": "No tours found in top districts"}

    # update session dataframes
    session.districts_df = session.districts_df[session.districts_df["district"].isin(session.top_districts)]
    session.hotels_df = filtered_hotels_df
    session.restaurants_df = filtered_restaurants_df
    session.tours_df = filtered_tours_df

    return {
        "status": "success",
        "message": "Top travel districts recommendations generated, all data filtered based on recommended districts",
        "recommendation": [
            {"district": district, "score": round(score, 4)} 
            for district, score in districts_scores[:6]
        ],
        "data": {
            "districts": session.districts_df.to_dict(orient="records"),
            "hotels": session.hotels_df.to_dict(orient="records"),
            "restaurants": session.restaurants_df.to_dict(orient="records"),
            "tours": session.tours_df.to_dict(orient="records")
        }
    }

generative_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

async def chatbot_output_streamer(session_id: str):
    session = get_session(session_id)
    user_inputs = session.user_inputs

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
    {session.districts_df.to_dict(orient="records")}

    HOTELS:
    {session.hotels_df.to_dict(orient="records")}

    RESTAURANTS:
    {session.restaurants_df.to_dict(orient="records")}

    TOURS:
    {session.tours_df.to_dict(orient="records")}


    **Rules:**
    1. One district per itinerary, hotel/restaurants/tours all from that district.
    2. Respect the activity counts (e.g., if restaurants=2 in activity_counts and nb_of_days=4, schedule restaurant visits for exactly 2 different days).
    3. Ensure each chosen restaurant/tour fits in its budget range for its category.
    4. Use variety: do not pick the same restaurant or tour twice in the same itinerary.
    5. The output must be **detailed day-by-day** with all activities.
    6. Choose all the activities, restos, hotels, on your own, don't give a chance for the user to choose them you do it.

    Return your answer as **3 separate itineraries**, each clearly labeled and formatted for easy reading.

    **Important:**:Break your output into small readable segments. 
    At the end of each segment, append the special marker "!!" (double exclamation). 
    Do not put "!!" anywhere else in the text. 
    This will let the client know where each chunk ends.
    also don't write free time next to everything, and put the name of the tour that you recommended and provide some basic info about each
    activity,
    Don't repeat the check in to the same hotels each day, just say it once at the beginning of day one. 
    """
    loop = asyncio.get_event_loop()
    buffer = "" 

    # Stream the generation
    def generate_chunks_blocking():
        nonlocal buffer
        for chunk in generative_model.generate_content(prompt, stream=True):
            text = chunk.text or ""
            buffer += text
            while "!!" in buffer:
                part, buffer = buffer.split("!!", 1)
                if part.strip():
                    yield part.strip().replace("*","") + "\n" 
        # yield leftover text
        if buffer.strip():
            yield buffer.strip() 

    # Run the blocking generator in a thread and yield asynchronously
    gen = generate_chunks_blocking()
    for chunk in await loop.run_in_executor(None, lambda: gen):
        yield chunk

@app.post(path="/generate")
async def generate(input: GenerateRequest):
    return StreamingResponse(chatbot_output_streamer(input.session_id),media_type="text/event-stream")