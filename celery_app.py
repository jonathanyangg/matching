import pandas as pd
import numpy as np
import openai
import os
import redis
import io
from celery import Celery
from dotenv import load_dotenv
import logging


load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configure Celery to use Redis as the message broker and backend
celery_app = Celery(
    'tasks',
    broker=os.environ.get("BROKER"),  # e.g., redis://localhost:6379/0
    backend=os.environ.get("BACKEND")  # e.g., redis://localhost:6379/1
)

# Initialize Redis connection
redis_client = redis.from_url(os.environ.get("REDIS_URL"))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Columns to merge for generating the text query
columns_to_merge_prospective = [
    'Res Status',
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
    'Person Hispanic'
]

columns_to_merge_current = [
    'Residential Status (Boarding or Day)',
    'Sex',
    'Academic Interests:',
    'Extra-curricular Interests (Clubs, affinity groups, hobbies, etc) ',
    'Fall co-curricular:',
    'Winter co-curricular:',
    'Spring co-curricular:',
    'Hometown',
    'State/Region',
    'Country',
    'School prior to attending Lawrenceville: ',
    'Race',
]


def format_row_prospective(row):
    """Format a DataFrame row by concatenating selected columns."""
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge_prospective if pd.notna(row[col])])

def format_row_current(row):
    """Format a DataFrame row by concatenating selected columns."""
    return ', '.join([f"{col}: {row[col]}" for col in columns_to_merge_current if pd.notna(row[col])])

def api_call(text):
    """Call OpenAI's embedding API for the provided text."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def generate_match_explanation(prospective_text, guide_text):
    """
    Generate a concise explanation for why a guide is a good match for a prospective student.
    """
    prompt = f"""
    A prospective student and a guide have been matched based on their profiles. Provide a concise explanation for why they are a good match.

    Prospective Student: {prospective_text}
    Guide: {guide_text}

    Explanation:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can also use "gpt-3.5-turbo" if preferred
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
    """
    Celery task to:
      1. Retrieve CSV file contents from Redis using the provided keys.
      2. Process the data by generating text queries, calling the OpenAI API for embeddings,
         and computing cosine similarities between prospective and current student embeddings.
      3. For each prospective student, identify the top three matches along with explanations.
      4. Store the resulting CSV back in Redis using a key based on the input key.
    """
    try:
        logging.info(f"Starting task with Redis keys - Prospective: {prospective_key}, Current: {current_key}")
        
        # Retrieve file contents from Redis
        prospective_content = redis_client.get(prospective_key)
        current_content = redis_client.get(current_key)

        if not prospective_content or not current_content:
            raise FileNotFoundError("Could not retrieve files from Redis")

        # Convert the retrieved bytes into DataFrames
        prospective_df = pd.read_csv(io.BytesIO(prospective_content))
        current_df = pd.read_csv(io.BytesIO(current_content))

        # Create a text query column for each DataFrame
        prospective_df['Text Query'] = prospective_df.apply(format_row_prospective, axis=1)
        current_df['Text Query'] = current_df.apply(format_row_current, axis=1)

        # Generate embeddings for each text query
        prospective_df['Embeddings'] = prospective_df['Text Query'].apply(api_call)
        current_df['Embeddings'] = current_df['Text Query'].apply(api_call)

        # Prepare columns for match suggestions, explanations, and scores
        for i in range(1, 4):
            prospective_df[f'suggestion_{i}'] = ""
            prospective_df[f'description_{i}'] = ""
            prospective_df[f'match_score_{i}'] = 0.0

        # For each prospective student, find the top 3 current student matches
        for i, row in prospective_df.iterrows():
            try:
                # Log the values before filtering
                logging.info(f"Prospective student gender: {row['Person Sex']}")
                logging.info(f"Current student genders sample: {current_df['Sex'].head().tolist()}")
                
                # Filter current students by matching gender and YOG (Year of Graduation)
                filtered_current_df = current_df[
                    (current_df["Sex"].str.upper().str[0] == row["Person Sex"][0].upper()) &
                    (current_df["Year of Graduation:"].astype(float) - float(row["YOG"]) == -1)
                ]
                
                # Log the filtering results
                logging.info(f"Found {len(filtered_current_df)} matches for gender and YOG")
                
                if filtered_current_df.empty:
                    logging.warning(f"No matches found for Slate ID: {row['Slate ID']}")
                    continue

                # Log similarity calculation
                similarities = filtered_current_df["Embeddings"].apply(
                    lambda x: cosine_similarity(row["Embeddings"], x)
                )
                logging.info(f"Calculated similarities. Max similarity: {similarities.max():.4f}")

                # Add more detailed logging for top matches
                filtered_current_df = filtered_current_df.assign(similarity=similarities)
                top_matches = filtered_current_df.sort_values(by="similarity", ascending=False).head(3)
                logging.info(f"Top 3 match scores: {top_matches['similarity'].tolist()}")

            except Exception as e:
                logging.error(f"Error processing student {row['Slate ID']}: {str(e)}")
                continue

            # Record the top matches and generate explanations
            for j, (_, match_row) in enumerate(top_matches.iterrows(), start=1):
                prospective_df.at[i, f"suggestion_{j}"] = match_row["Name (First Last):"]
                explanation = generate_match_explanation(row["Text Query"], match_row["Text Query"])
                prospective_df.at[i, f"description_{j}"] = explanation
                prospective_df.at[i, f"match_score_{j}"] = match_row["similarity"]

        # Finalize the result DataFrame with selected columns
        prospective_df = prospective_df[[
            "Slate ID", "YOG",
            "suggestion_1", "description_1", "match_score_1",
            "suggestion_2", "description_2", "match_score_2",
            "suggestion_3", "description_3", "match_score_3"
        ]]

        # Instead of saving to disk, store the CSV result back in Redis.
        output = io.StringIO()
        prospective_df.to_csv(output, index=False)
        result_key = f"result_{prospective_key}"  # e.g., "result_prospective_<unique_id>"
        redis_client.setex(result_key, 3600, output.getvalue())

        logging.info(f"Task completed successfully. Result stored in Redis under key: {result_key}")
        return {"result_key": result_key}
    
    except Exception as e:
        logging.error(f"Error in generate_embeddings_task: {e}")
        raise

@celery_app.task
def delete_files(file_paths):
    """
    Deletes files from the filesystem. (This task is still useful if you have any
    temporary files stored on disk that need cleanup.)
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
