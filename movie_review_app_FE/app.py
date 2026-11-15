import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, render_template, request, jsonify, session
import re
import json
import uuid
from datetime import datetime
from inference import predict_sentiment
from summarizer import generate_dual_summary
from lib_file import lib_path
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'movie_review'

# Create directory for storing temporary review data
TEMP_DATA_DIR = './temp_review_data'
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Movie data with their review files
MOVIES = {
    'inception': {
        'name': 'Interstellar',
        'file': 'Text Files/interstellar_reviews.txt',
    },
    'avatar': {
        'name': 'Avatar',
        'file': 'Text Files/avatar_review.txt',
    },
    'Openheimer': {
        'name': 'Openheimer',
        'file': 'Text Files/oppenheimer_reviews.txt',
    }
}

# File storage functions
def save_reviews_to_file(session_id, positive_reviews, negative_reviews, movie_name):
    """Save positive and negative reviews to a JSON file"""
    file_path = os.path.join(TEMP_DATA_DIR, f"{session_id}.json")
    data = {
        'session_id': session_id,
        'movie_name': movie_name,
        'positive_reviews': positive_reviews,
        'negative_reviews': negative_reviews,
        'created_at': datetime.now().isoformat()
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved reviews to file: {file_path}")
    return file_path

def load_reviews_from_file(session_id):
    """Load positive and negative reviews from JSON file"""
    file_path = os.path.join(TEMP_DATA_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded reviews from file: {file_path}")
        return data
    return None

def delete_reviews_file(session_id):
    """Delete the reviews JSON file for a session"""
    file_path = os.path.join(TEMP_DATA_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted reviews file: {file_path}")
        return True
    return False

def read_movie_reviews(movie_key):
    """Read reviews from the movie's text file and clean numbering"""
    file_path = MOVIES[movie_key]['file']
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        reviews = []
        lines = content.split('\n')
        current_review = []

        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                if current_review:
                    review_text = ' '.join(current_review).strip()
                    review_text = re.sub(r'^\d+\.\s*', '', review_text)
                    if review_text:
                        reviews.append(review_text)
                    current_review = []
                line_without_number = re.sub(r'^\d+\.\s*', '', line.strip())
                if line_without_number:
                    current_review.append(line_without_number)
            else:
                if line.strip():
                    current_review.append(line.strip())

        if current_review:
            review_text = ' '.join(current_review).strip()
            review_text = re.sub(r'^\d+\.\s*', '', review_text)
            if review_text:
                reviews.append(review_text)

        if not reviews:
            raw_reviews = re.split(r'\n\s*\n', content)
            for review in raw_reviews:
                review = review.strip()
                if review:
                    review = re.sub(r'^\d+\.\s*', '', review)
                    reviews.append(review)

        return reviews

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def analyze_reviews(reviews):
    """Analyze all reviews and separate positive and negative"""
    results = {
        'positive': [],
        'negative': []
    }

    for review in reviews:
        sentiment = predict_sentiment(review)
        if sentiment == "Positive":
            results['positive'].append(review)
        else:
            results['negative'].append(review)

    return results

@app.route('/')
def index():
    """Home page - generate unique session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        print(f"New session created: {session['session_id']}")
    return render_template('index.html', movies=MOVIES)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze reviews for selected movie"""
    movie_key = request.json.get('movie_key')
    session_id = session.get('session_id')

    if not session_id:
        session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']

    if not movie_key or movie_key not in MOVIES:
        return jsonify({'error': 'Invalid movie selection'}), 400

    # Delete previous reviews file for this session (when new movie selected)
    delete_reviews_file(session_id)

    # Read reviews
    reviews = read_movie_reviews(movie_key)

    if not reviews:
        return jsonify({'error': 'No reviews found for this movie'}), 404

    # Analyze sentiment
    analysis = analyze_reviews(reviews)

    # Save to file instead of session
    movie_name = MOVIES[movie_key]['name']
    save_reviews_to_file(
        session_id,
        analysis['positive'],
        analysis['negative'],
        movie_name
    )

    return jsonify({
        'movie_name': movie_name,
        'positive_count': len(analysis['positive']),
        'negative_count': len(analysis['negative']),
        'positive_reviews': analysis['positive'][:20],
        'negative_reviews': analysis['negative'][:20]
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate summaries for positive and negative reviews"""
    print("=" * 50)
    print("Summarize endpoint called")

    session_id = session.get('session_id')

    if not session_id:
        print("No session ID found")
        return jsonify({'error': 'No analysis found. Please analyze a movie first.'}), 400

    # Load reviews from file instead of session
    data = load_reviews_from_file(session_id)

    if not data:
        print(f"No review data found for session: {session_id}")
        return jsonify({'error': 'No reviews found. Please analyze a movie first.'}), 400

    positive_reviews = data.get('positive_reviews', [])
    negative_reviews = data.get('negative_reviews', [])
    movie_name = data.get('movie_name', 'Movie')

    print(f"Movie: {movie_name}")
    print(f"Positive reviews count: {len(positive_reviews)}")
    print(f"Negative reviews count: {len(negative_reviews)}")

    if not positive_reviews and not negative_reviews:
        print("No reviews found in data")
        return jsonify({'error': 'No reviews found. Please analyze a movie first.'}), 400

    try:
        # Generate summaries
        summaries = generate_dual_summary(positive_reviews, negative_reviews)

        print(f"Summaries generated successfully")
        print(f"Positive summary length: {len(summaries.get('positive', ''))}")
        print(f"Negative summary length: {len(summaries.get('negative', ''))}")

        return jsonify({
            'movie_name': movie_name,
            'positive_summary': summaries.get('positive', 'No summary available'),
            'negative_summary': summaries.get('negative', 'No summary available')
        })

    except Exception as e:
        print(f"Error in summarize endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Error generating summary: {str(e)}'
        }), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the session and delete stored reviews"""
    session_id = session.get('session_id')
    if session_id:
        delete_reviews_file(session_id)
    session.clear()
    # Generate new session ID for next analysis
    session['session_id'] = str(uuid.uuid4())
    print("Session reset and files deleted")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
