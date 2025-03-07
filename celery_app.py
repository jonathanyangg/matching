import pandas as pd
import numpy as np
import openai
import os
import redis
import io
from celery import Celery
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor


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


def format_dataframe_columns(df, start_pos=3):
    """Merge columns from start_pos to the end into a single text column."""
    columns_to_merge = df.columns[start_pos:]
    
    # Create a new column with the merged values
    df['Text Query'] = df.apply(
        lambda row: ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])]),
        axis=1
    )
    return df

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
      3. For each prospective student, identify the top two matches along with explanations.
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
        prospective_df = format_dataframe_columns(prospective_df, 3)
        current_df = format_dataframe_columns(current_df, 3)

        prospective_df.iloc[:, 2] = prospective_df.iloc[:, 2].replace('PG', '12')

        # 1. First generate embeddings for ALL students
        logging.info("Generating embeddings for prospective students...")
        prospective_df['Embeddings'] = prospective_df['Text Query'].apply(api_call)
        
        logging.info("Generating embeddings for current students...")
        current_df['Embeddings'] = current_df['Text Query'].apply(api_call)

        # 2. Initialize result columns
        for i in range(1, 4):
            prospective_df[f'suggestion_{i}'] = ""
            prospective_df[f'description_{i}'] = ""
            prospective_df[f'match_score_{i}'] = 0.0

        # 3. Process each prospective student
        for i, row in prospective_df.iterrows():
            try:
                logging.info(f"Processing student {i+1}/{len(prospective_df)}")
                
                # First calculate ALL similarities
                similarities = current_df['Embeddings'].apply(
                    lambda x: cosine_similarity(row['Embeddings'], x)
                )
                
                # Then apply gender and grade filters
                mask = (
                    (current_df.iloc[:, 1].astype(str).str.upper().str[0] == row.iloc[1][0].upper()) &
                    (current_df.iloc[:, 2].astype(float) == float(row.iloc[2]))
                )
                
                # Get top matches from filtered results
                filtered_similarities = similarities[mask]
                if not filtered_similarities.empty:
                    top_matches = filtered_similarities.nlargest(3)
                    
                    # Store the matches
                    for j, (idx, score) in enumerate(top_matches.items(), 1):
                        prospective_df.at[i, f'suggestion_{j}'] = current_df.iloc[idx]['Name (First Last):']
                        prospective_df.at[i, f'match_score_{j}'] = score
                        # Generate explanation
                        explanation = generate_match_explanation(
                            row['Text Query'],
                            current_df.iloc[idx]['Text Query']
                        )
                        prospective_df.at[i, f'description_{j}'] = explanation
                
            except Exception as e:
                logging.error(f"Error processing student {row.iloc[0]}: {str(e)}")
                continue

        # 4. Select final columns for output
        result_df = prospective_df[[
            'Person Reference ID',  # First column
            'Grade',               # Third column
            'suggestion_1', 'description_1', 'match_score_1',
            'suggestion_2', 'description_2', 'match_score_2',
            'suggestion_3', 'description_3', 'match_score_3'
        ]]

        # After processing all students
        logging.info(f"Final result shape: {result_df.shape}")
        logging.info(f"Number of non-empty matches: {(result_df['suggestion_1'] != '').sum()}")
        logging.info(f"Sample of results:\n{result_df.head(2)}")

        # Save to Redis and return
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        result_key = f"result_{prospective_key}"
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

def batch_api_call(texts, batch_size=20):
    """Call OpenAI's embedding API for multiple texts in batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=batch
        )
        batch_embeddings = [item['embedding'] for item in response['data']]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def generate_explanations_in_parallel(matches, max_workers=4):
    """Generate match explanations in parallel."""
    prospective_texts = [match['prospective_text'] for match in matches]
    guide_texts = [match['guide_text'] for match in matches]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        explanations = list(executor.map(
            lambda args: generate_match_explanation(*args),
            zip(prospective_texts, guide_texts)
        ))
    
    return explanations
