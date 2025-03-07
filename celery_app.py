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

def calculate_similarities_vectorized(prospective_embedding, current_embeddings_array):
    """Calculate cosine similarities in a vectorized way."""
    # Convert to numpy arrays for faster computation
    prospective_embedding = np.array(prospective_embedding)
    current_embeddings_array = np.vstack(current_embeddings_array)
    
    # Calculate similarities in one operation
    similarities = np.dot(current_embeddings_array, prospective_embedding) / (
        np.linalg.norm(current_embeddings_array, axis=1) * np.linalg.norm(prospective_embedding)
    )
    
    return similarities

def format_dataframe_columns(df, start_pos=3):
    """Merge columns from start_pos to the end into a single text column."""
    columns_to_merge = df.columns[start_pos:]
    df['Text Query'] = df.apply(
        lambda row: ', '.join([f"{col}: {row[col]}" for col in columns_to_merge if pd.notna(row[col])]),
        axis=1
    )
    return df

def batch_api_call(texts, batch_size=20):
    """Call OpenAI's embedding API for multiple texts in batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            # Clean and validate the batch
            cleaned_batch = []
            for text in batch:
                # Handle None/null values
                if pd.isna(text):
                    text = ""
                
                # Convert to string if not already
                text = str(text)
                
                # Remove any problematic characters
                text = text.strip()
                
                # Ensure non-empty string
                if not text:
                    text = "no information provided"
                
                cleaned_batch.append(text)
            
            logging.info(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
            logging.info(f"Sample text from batch: {cleaned_batch[0][:100]}...")
            
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=cleaned_batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logging.error(f"Error in batch {i//batch_size + 1}: {e}")
            logging.error(f"Problematic batch content: {batch}")
            raise
    
    return all_embeddings

def generate_match_explanation(prospective_text, guide_text):
    """Generate a concise explanation for why a guide is a good match."""
    prompt = f"""
    A prospective student and a guide have been matched based on their profiles. Provide a concise explanation for why they are a good match.

    Prospective Student: {prospective_text}
    Guide: {guide_text}

    Explanation:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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

@celery_app.task
def generate_embeddings_task(prospective_key, current_key):
    """
    Optimized Celery task for student matching.
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

        # Log initial shapes
        logging.info(f"Prospective DF shape: {prospective_df.shape}")
        logging.info(f"Current DF shape: {current_df.shape}")

        # Create text queries
        prospective_df = format_dataframe_columns(prospective_df, 3)
        current_df = format_dataframe_columns(current_df, 3)

        # Replace 'PG' with '12'
        prospective_df.iloc[:, 2] = prospective_df.iloc[:, 2].replace('PG', '12')

        # Generate embeddings in batches
        logging.info("Generating embeddings for prospective students...")
        prospective_df['Embeddings'] = batch_api_call(prospective_df['Text Query'].tolist())
        
        logging.info("Generating embeddings for current students...")
        current_df['Embeddings'] = batch_api_call(current_df['Text Query'].tolist())

        # Pre-compute gender and grade filters
        current_df['Gender_First'] = current_df.iloc[:, 1].astype(str).str.upper().str[0]
        current_df['Grade_Float'] = current_df.iloc[:, 2].astype(float)
        current_embeddings_array = np.vstack(current_df['Embeddings'].values)

        # Initialize result columns
        result_columns = [f'{col}_{i}' for i in range(1, 3) 
                         for col in ['suggestion', 'description', 'match_score']]
        for col in result_columns:
            prospective_df[col] = ""

        # Process each prospective student
        total_students = len(prospective_df)
        for i, row in prospective_df.iterrows():
            try:
                logging.info(f"Processing student {i+1}/{total_students}")
                
                # Add this check
                if pd.isna(row['Embeddings']):
                    logging.warning(f"Skipping student {i+1} - no embeddings available")
                    continue
                
                # Calculate similarities vectorized
                similarities = calculate_similarities_vectorized(
                    row['Embeddings'],
                    current_embeddings_array
                )
                similarities = pd.Series(similarities, index=current_df.index)
                
                # Apply filters
                mask = (
                    (current_df['Gender_First'] == row.iloc[1][0].upper()) &
                    (current_df['Grade_Float'] == float(row.iloc[2]))
                )
                
                filtered_similarities = similarities[mask]
                logging.info(f"Found {len(filtered_similarities)} matches for gender and YOG")
                
                if not filtered_similarities.empty:
                    top_matches = filtered_similarities.nlargest(2)
                    
                    # Prepare matches for parallel explanation generation
                    matches_to_explain = [
                        {
                            'prospective_text': row['Text Query'],
                            'guide_text': current_df.iloc[idx]['Text Query']
                        }
                        for idx, _ in top_matches.items()
                    ]
                    
                    explanations = generate_explanations_in_parallel(matches_to_explain)
                    
                    # Bulk update matches
                    for j, ((idx, score), explanation) in enumerate(zip(top_matches.items(), explanations), 1):
                        update_data = {
                            f'suggestion_{j}': current_df.iloc[idx]['Name (First Last):'],
                            f'match_score_{j}': score,
                            f'description_{j}': explanation
                        }
                        prospective_df.loc[i, update_data.keys()] = update_data.values()

            except Exception as e:
                logging.error(f"Error processing student {row.iloc[0]}: {str(e)}")
                continue

        # Select final columns for output
        result_df = prospective_df[[
            'Person Reference ID',  # First column
            'Grade',               # Third column
            'suggestion_1', 'description_1', 'match_score_1',
            'suggestion_2', 'description_2', 'match_score_2',
        ]]

        # Log final results
        logging.info(f"Final result shape: {result_df.shape}")
        logging.info(f"Number of non-empty matches: {(result_df['suggestion_1'] != '').sum()}")
        logging.info(f"Sample of results:\n{result_df.head(2)}")

        # Save to Redis
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
    """Delete files from the filesystem."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Deleted file: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
