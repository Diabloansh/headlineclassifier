import json
import re
import ssl

# Fix SSL issue if nltk tries to download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Using fallback stopword list due to: {e}")
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

input_file = '/Users/anshmadan/Desktop/headline-classifier/hierarchical-classifier/Dataset/BERT_training_3000_v2.json'
output_file = '/Users/anshmadan/Desktop/headline-classifier/hierarchical-classifier/Dataset/BERT_training_3000_v2_preprocessed.json'

with open(input_file, 'r') as f:
    data = json.load(f)

for item in data:
    text = item['text']
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Stopword removal
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    
    # Update text
    item['text'] = ' '.join(filtered_words)

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Preprocessing complete. Saved {len(data)} records to {output_file}")
