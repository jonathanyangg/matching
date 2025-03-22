import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
import pandas as pd
import redis
from celery import Celery
from dotenv import load_dotenv


# Constants and configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EXPLANATION_MODEL = "gpt-4"
EXPLANATION_MAX_TOKENS = 100
BATCH_SIZE = 20
MAX_WORKERS = 4
REDIS_CACHE_EXPIRY = 3600  # seconds (1 hour)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("matching_task.log"),
        logging.StreamHandler()
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Set debug level if environment variable is set
if os.environ.get("DEBUG_LOGGING") == "1":
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

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

def batch_api_call(texts, batch_size=BATCH_SIZE):
    """Call OpenAI's embedding API for multiple texts in batches.
    
    Args:
        texts: List of text strings to get embeddings for
        batch_size: Number of texts to process in each API call
        
    Returns:
        List of embedding vectors
    """
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
                
                # Convert to string and clean
                text = str(text).strip()
                
                # Ensure non-empty string
                if not text:
                    text = "no information provided"
                
                cleaned_batch.append(text)
            
            logger.info(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
            logger.info(f"Sample text from batch: {cleaned_batch[0][:100]}...")
            
            response = openai.Embedding.create(
                model=EMBEDDING_MODEL,
                input=cleaned_batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logger.error(f"Error in batch {i//batch_size + 1}: {e}")
            logger.error(f"Problematic batch content: {batch}")
            raise
    
    return all_embeddings

def generate_match_explanation(prospective_text, guide_text):
    """Generate a concise explanation for why a guide is a good match.
    
    Args:
        prospective_text: Text description of prospective student
        guide_text: Text description of guide student
        
    Returns:
        A string explanation of the match
    """
    prompt = f"""
    A prospective student and a guide have been matched based on their profiles. Provide a concise explanation for why they are a good match.

    Prospective Student: {prospective_text}
    Guide: {guide_text}

    Explanation:
    """
    try:
        response = openai.ChatCompletion.create(
            model=EXPLANATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=EXPLANATION_MAX_TOKENS
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return "Error generating explanation."

def generate_explanations_in_parallel(matches, max_workers=MAX_WORKERS):
    """Generate match explanations in parallel.
    
    Args:
        matches: List of dictionaries with 'prospective_text' and 'guide_text' keys
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of explanation strings
    """
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
        logger.info(f"Starting task with Redis keys - Prospective: {prospective_key}, Current: {current_key}")
        
        # Retrieve file contents from Redis
        prospective_content = redis_client.get(prospective_key)
        current_content = redis_client.get(current_key)

        if not prospective_content or not current_content:
            raise FileNotFoundError("Could not retrieve files from Redis")

        # Convert the retrieved bytes into DataFrames
        prospective_df = pd.read_csv(io.BytesIO(prospective_content))
        current_df = pd.read_csv(io.BytesIO(current_content))

        # Log initial shapes
        logger.info(f"Prospective DF shape: {prospective_df.shape}")
        logger.info(f"Current DF shape: {current_df.shape}")

        # Create text queries
        prospective_df = format_dataframe_columns(prospective_df, 3)
        current_df = format_dataframe_columns(current_df, 3)

        # Replace 'PG' with '12'
        prospective_df.iloc[:, 2] = prospective_df.iloc[:, 2].replace('PG', '12')

        # Generate embeddings in batches
        logger.info("Generating embeddings for prospective students...")
        prospective_df['Embeddings'] = batch_api_call(prospective_df['Text Query'].tolist())
        
        logger.info("Generating embeddings for current students...")
        current_df['Embeddings'] = batch_api_call(current_df['Text Query'].tolist())

        # Pre-compute gender and grade filters
        current_df['Gender_First'] = current_df.iloc[:, 1].astype(str).str.upper().str[0]
        current_df['Grade_Float'] = current_df.iloc[:, 2].astype(float)
        current_embeddings_array = np.vstack(current_df['Embeddings'].values)

        # Initialize result columns
        result_columns = [f'{col}_{i}' for i in range(1, 4) 
                         for col in ['suggestion', 'description', 'match_score']]
        for col in result_columns:
            prospective_df[col] = ""
            
        # Create a dictionary to keep track of how many times each guide has been selected
        # Key is the guide's index in current_df, value is the count of selections
        guide_selection_counts = {}
        
        # Process each prospective student
        total_students = len(prospective_df)
        for i, row in prospective_df.iterrows():
            try:
                logger.info(f"Processing student {i+1}/{total_students}: ID={row.iloc[0]}, Gender={row.iloc[1]}, Grade={row.iloc[2]}")
                
                # Add this check - fix for array embeddings
                if isinstance(row['Embeddings'], list) and len(row['Embeddings']) > 0:
                    # Embeddings exist and are valid
                    pass
                elif not isinstance(row['Embeddings'], list) and pd.notna(row['Embeddings']):
                    # Embeddings exist as a non-list (could be numpy array)
                    pass
                else:
                    logger.warning(f"Skipping student {i+1} - no embeddings available for ID={row.iloc[0]}")
                    continue
                
                # Calculate similarities vectorized
                similarities = calculate_similarities_vectorized(
                    row['Embeddings'],
                    current_embeddings_array
                )
                similarities = pd.Series(similarities, index=current_df.index)
                logger.debug(f"Calculated {len(similarities)} similarity scores. Max score: {similarities.max():.4f}")
                
                # Extract gender and grade safely for logging
                gender_value = str(row.iloc[1]).upper()[0] if pd.notna(row.iloc[1]) else ''
                grade_value = float(row.iloc[2]) if pd.notna(row.iloc[2]) and row.iloc[2] != '' else 0
                logger.info(f"Filtering guides for student ID={row.iloc[0]}: Gender={gender_value}, Grade={grade_value}")
                
                # Apply filters
                mask = (
                    (current_df['Gender_First'] == str(row.iloc[1]).upper()[0] if pd.notna(row.iloc[1]) else '') &
                    (current_df['Grade_Float'] == float(row.iloc[2]) if pd.notna(row.iloc[2]) and row.iloc[2] != '' else 0)
                )
                logger.debug(f"Initial filter matched {mask.sum()} guides with same gender and grade")
                
                # Create a copy to avoid SettingWithCopyWarning
                mask_copy = mask.copy()
                
                # Exclude guides who have been selected twice already
                excluded_count = 0
                for guide_idx, count in guide_selection_counts.items():
                    if count >= 2 and guide_idx in mask_copy.index:
                        mask_copy.loc[guide_idx] = False
                        excluded_count += 1
                
                if excluded_count > 0:
                    logger.info(f"Excluded {excluded_count} guides who were already selected twice")
                
                filtered_similarities = similarities[mask_copy]
                logger.info(f"Found {len(filtered_similarities)} eligible matches for student ID={row.iloc[0]}")
                
                if filtered_similarities.empty:
                    logger.warning(f"No suitable matches found for student ID={row.iloc[0]}. Continuing to next student.")
                    continue
                    
                if len(filtered_similarities) < 3:
                    logger.warning(f"Only {len(filtered_similarities)} matches found for student ID={row.iloc[0]} - fewer than the requested 3 matches")
                
                top_matches = filtered_similarities.nlargest(3)
                logger.info(f"Top match score for student ID={row.iloc[0]}: {top_matches.iloc[0]:.4f}")
                
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
                    
                    # Update the guide selection count
                    guide_selection_counts[idx] = guide_selection_counts.get(idx, 0) + 1
                    logger.info(f"Guide {current_df.iloc[idx]['Name (First Last):']} selected {guide_selection_counts[idx]} time(s)")

            except Exception as e:
                logger.error(f"Error processing student {row.iloc[0]}: {str(e)}")
                logger.error(f"Student data: Gender={row.iloc[1] if pd.notna(row.iloc[1]) else 'None'}, Grade={row.iloc[2] if pd.notna(row.iloc[2]) else 'None'}")
                import traceback
                logger.error(f"Stack trace: {traceback.format_exc()}")
                continue
        
        # Log the final guide selection counts
        logger.info("Final guide selection counts:")
        for idx, count in guide_selection_counts.items():
            logger.info(f"Guide {current_df.iloc[idx]['Name (First Last):']} was selected {count} time(s)")

        # Get the 1st and 3rd columns by position
        first_col = prospective_df.columns[0]  # 'Person Reference ID'
        third_col = prospective_df.columns[2]  # 'Grade'
        
        # Select the 1st column, 3rd column, and the last 9 columns (match results)
        result_df = prospective_df[[
            first_col,
            third_col,
            'suggestion_1', 'description_1', 'match_score_1',
            'suggestion_2', 'description_2', 'match_score_2',
            'suggestion_3', 'description_3', 'match_score_3',
        ]]

        # Log final results
        logger.info(f"Final result shape: {result_df.shape}")
        logger.info(f"Number of non-empty matches: {(result_df['suggestion_1'] != '').sum()}")
        logger.info(f"Sample of results:\n{result_df.head(2)}")

        # Save to Redis
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        result_key = f"result_{prospective_key}"
        redis_client.setex(result_key, 3600, output.getvalue())
        
        logger.info(f"Task completed successfully. Result stored in Redis under key: {result_key}")
        return {"result_key": result_key}

    except Exception as e:
        logger.error(f"Error in generate_embeddings_task: {e}")
        raise

@celery_app.task
def delete_files(file_paths):
    """Delete files from the filesystem."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
