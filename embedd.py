import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
from supabase import create_client

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Function to generate embedding for a given text
def embedd(text):
    return model.encode(text, convert_to_tensor=True).cpu().numpy()

# Load Supabase credentials from environment variables
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase = create_client(url, key)

# Fetch district IDs and descriptions from the Supabase table
result = supabase.table("districts").select("id,description").execute()
rows = pd.DataFrame(result.data).reset_index(drop=True)

# Generate and update embeddings for each district description
for id, desc in rows.itertuples(index=False):
    embedded_desc = embedd(desc)
    list_data = embedded_desc.tolist()  # Convert numpy array to list
    supabase.table("districts").update({"embedding": list_data}).eq("id", id).execute()
