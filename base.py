from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory, send_file
from celery_app import generate_embeddings_task, delete_files
import os
import pandas as pd
import logging
from dotenv import load_dotenv
import io
import redis
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Use simple ephemeral storage
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], mode=0o777, exist_ok=True)

# Initialize Redis connection
redis_client = redis.from_url(os.environ.get("REDIS_URL"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/match_students', methods=['POST'])
def match_students():
    try:
        prospective_file = request.files.get("prospective_students_file")
        current_file = request.files.get("current_students_file")

        if not prospective_file or not current_file:
            logging.error("Both files are required for matching.")
            return jsonify({"error": "Both files are required!"}), 400

        # Read files and store in Redis
        prospective_content = prospective_file.read()
        current_content = current_file.read()

        # Generate a custom unique ID that we will also use as the Celery task ID
        upload_id = str(uuid.uuid4())
        prospective_key = f"prospective_{upload_id}"
        current_key = f"current_{upload_id}"

        # Store file contents in Redis with 1-hour expiration
        redis_client.setex(prospective_key, 3600, prospective_content)
        redis_client.setex(current_key, 3600, current_content)

        logging.info(f"Files stored in Redis with upload ID: {upload_id}")

        # Use apply_async and pass upload_id as the Celery task id
        task = generate_embeddings_task.apply_async(
            args=[prospective_key, current_key],
            task_id=upload_id
        )
        logging.info(f"Celery Task ID: {task.id}")

        # Return the task id (which is also the upload_id) to the client
        return render_template("loading.html", task_id=task.id)

    except Exception as e:
        logging.error(f"Error in match_students: {e}")
        return jsonify({"error": "Failed to process student matching"}), 500

@app.route('/task_status/<task_id>')
def task_status(task_id):
    try:
        task = generate_embeddings_task.AsyncResult(task_id)
        logging.info(f"Checking status for Task ID: {task_id}. Current state: {task.state}")

        if task.state == 'SUCCESS':
            result = task.result
            if not result or 'result_key' not in result:
                logging.error("No result key returned from the task result.")
                return jsonify({"status": "FAILURE", "error": "No result key in task result"}), 500

            # Get the data from Redis
            result_key = result['result_key']
            csv_data = redis_client.get(result_key)
            if not csv_data:
                logging.error("Could not retrieve CSV data from Redis.")
                return jsonify({"status": "FAILURE", "error": "Could not retrieve results"}), 500

            # Save the CSV data to a file
            filename = f"matches_{task_id}.csv"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'w') as f:
                f.write(csv_data.decode('utf-8'))

            # Schedule cleanup after one hour
            delete_files.apply_async(args=[[file_path]], countdown=3600)
            
            return jsonify({"status": "SUCCESS", "redirect_url": url_for('results', filename=filename)})

        if task.state == 'FAILURE':
            # Build Redis keys using the same task_id (which is our upload_id)
            prospective_key = f"prospective_{task_id}"
            current_key = f"current_{task_id}"
            result_key = f"result_prospective_{task_id}"
            
            for key in [prospective_key, current_key, result_key]:
                if redis_client.exists(key):
                    redis_client.delete(key)
                    logging.info(f"Cleaned up Redis key: {key}")
            
            error_info = str(task.info)
            logging.error(f"Task failed with error: {error_info}")
            return jsonify({"status": "FAILURE", "error": error_info}), 500

        return jsonify({"status": task.state})

    except Exception as e:
        logging.error(f"Error in task_status: {e}")
        return jsonify({"error": "Failed to retrieve task status"}), 500

@app.route('/results')
def results():
    filename = request.args.get("filename")
    if not filename:
        logging.error("No filename specified in results request.")
        return "No file specified!", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        logging.error(f"Requested file not found: {file_path}")
        return "File not found!", 404

    # Read the results file to extract descriptions
    results_df = pd.read_csv(file_path)
    results_data = results_df.to_dict(orient="records")
    return render_template("results.html", results_data=results_data, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prospective_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "prospective_students.csv")
        current_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "current_students.csv")

        if not os.path.exists(file_path):
            logging.error(f"File not found for download: {file_path}")
            return jsonify({"error": "File not found"}), 404

        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

        # Cleanup the files after download
        for path in [file_path, prospective_file_path, current_file_path]:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Deleted file: {path}")

        return response

    except Exception as e:
        logging.error(f"Error during file download: {e}")
        return jsonify({"error": "Failed to download file"}), 500

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    try:
        # Revoke the Celery task using the task_id (which is also our upload_id)
        task = generate_embeddings_task.AsyncResult(task_id)
        task.revoke(terminate=True)
        
        # Build Redis keys from the same task_id
        prospective_key = f"prospective_{task_id}"
        current_key = f"current_{task_id}"
        
        for path in [prospective_key, current_key]:
            if redis_client.exists(path):
                redis_client.delete(path)
                logging.info(f"Deleted Redis key: {path}")
        
        return jsonify({"status": "cancelled"})
    except Exception as e:
        logging.error(f"Error cancelling task: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
