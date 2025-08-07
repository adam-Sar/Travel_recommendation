import json
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

#initialize app
app = FastAPI()
#initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')

#load supabase credentials
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url,key)

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
    activity_counts: Dict[str, int]  #ex: {"hotel":3,"restaurant":2,"tour":3}, how many times go for each        
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
    global districts_df, hotels_df, restaurants_df, tours_df

    # Use threads to fetch all tables in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            "districts": executor.submit(fetch_table_data, "districts", "country,district,description,tags,embedding"),
            "hotels": executor.submit(fetch_table_data, "hotels", "name,country,district,price_per_day,rating,description,facilities"),
            "restaurants": executor.submit(fetch_table_data, "restaurants", "title,country,district,avg_price,rating,description,tag"),
            "tours": executor.submit(fetch_table_data, "tours", "title,country,district,duration,price,description")
        }

        # Store results in global variables
        districts_df = futures["districts"].result()
        hotels_df = futures["hotels"].result()
        restaurants_df = futures["restaurants"].result()
        tours_df = futures["tours"].result()

# API endpoint to trigger quiz data loading
@app.post("/start-quiz")
def start_quiz():
    load_data_from_supabase()
    return {"message": "Quiz data loaded"}
