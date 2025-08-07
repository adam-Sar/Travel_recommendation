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
