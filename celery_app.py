import pandas as pd
import numpy as np
import openai
import os
from celery import Celery
from dotenv import load_dotenv
import logging
import io
import redis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configure Celery to use Redis as the message broker
celery_app = Celery(
    'tasks',
    broker=os.environ.get("BROKER"),  # Redis broker URL
    backend=os.environ.get("BACKEND")  # Redis backend URL
)

# Configure consistent upload directory
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, mode=0o777, exist_ok=True)

# Initialize Redis connection
redis_client = redis.from_url(os.environ.get("REDIS_URL"))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

columns_to_merge = ['Res Status',
 'Person Sex',
 'Person Academic Interests',
 'Person Extra-Curricular Interest',
 'Sport1',
 'Sport2',
 'Sport3',
 'City',
 'State/Region',
 'Country',
 'School',
 'Person Race',
 'Person Hispanic']

def format_row(row):
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])])

def api_call(row):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=row
    )
    return response['data'][0]['embedding']

def generate_match_explanation(prospective_text, guide_text):
    """
    Generate a description for why the guide is a good match for the prospective student.
    """
    prompt = f"""
    A prospective student and a guide have been matched based on their profiles. Provide a concise explanation for why they are a good match.

    Prospective Student: {prospective_text}
    Guide: {guide_text}

    Explanation:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if you prefer
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        return "Error generating explanation."


@celery_app.task
def generate_embeddings_task(prospective_key, current_key):
    try:
        logging.info(f"Starting task with Redis keys - Prospective: {prospective_key}, Current: {current_key}")
        
        # Get file contents from Redis
        prospective_content = redis_client.get(prospective_key)
        current_content = redis_client.get(current_key)

        if not prospective_content or not current_content:
            raise FileNotFoundError("Could not retrieve files from Redis")

        # Convert bytes to DataFrame
        prospective_df = pd.read_csv(io.BytesIO(prospective_content))
        current_df = pd.read_csv(io.BytesIO(current_content))

        prospective_df['Text Query'] = prospective_df.apply(format_row, axis=1)
        current_df['Text Query'] = current_df.apply(format_row, axis=1)

        prospective_df['Embeddings'] = prospective_df['Text Query'].apply(api_call)
        current_df['Embeddings'] = current_df['Text Query'].apply(api_call)

        # Initialize suggestions, descriptions, and match scores
        for i in range(1, 4):
            prospective_df[f'suggestion_{i}'] = np.nan
            prospective_df[f'description_{i}'] = np.nan
            prospective_df[f'match_score_{i}'] = np.nan

        for i, row in prospective_df.iterrows():
            # Filter current students by gender and YOG
            filtered_current_df = current_df[
                (current_df["Person Sex"] == row["Person Sex"]) &
                (current_df["YOG"] == row["YOG"])
            ]

            if filtered_current_df.empty:
                continue

            # Calculate cosine similarity with each student in the filtered current_df
            similarities = filtered_current_df["Embeddings"].apply(
                lambda x: cosine_similarity(row["Embeddings"], x)
            )

            # Add the similarities as a new column
            filtered_current_df = filtered_current_df.assign(similarity=similarities)

            # Sort by similarity in descending order
            top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)

            for j, (_, match_row) in enumerate(top_matches.iterrows(), start=1):
                prospective_df.at[i, f"suggestion_{j}"] = match_row["Slate ID"]
                explanation = generate_match_explanation(row["Text Query"], match_row["Text Query"])
                prospective_df.at[i, f"description_{j}"] = explanation
                prospective_df.at[i, f"match_score_{j}"] = match_row["similarity"]

        # Include additional metadata and finalize columns
        prospective_df = prospective_df[[
            "Slate ID", "YOG", "Text Query",
            "suggestion_1", "description_1", "match_score_1",
            "suggestion_2", "description_2", "match_score_2",
            "suggestion_3", "description_3", "match_score_3"
        ]]

        # Save results back to Redis
        output = io.StringIO()
        prospective_df.to_csv(output, index=False)
        result_key = f"result_{prospective_key}"
        redis_client.setex(result_key, 3600, output.getvalue())

        return {"result_key": result_key}

    except Exception as e:
        logging.error(f"Error in generate_embeddings_task: {e}")
        raise


@celery_app.task
def delete_files(file_paths):
    # Deletes uploaded files to free up storage
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
