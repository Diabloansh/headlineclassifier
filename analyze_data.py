import json
from collections import Counter
import re

train_path = "/Users/anshmadan/Desktop/headline-classifier/hierarchical-classifier/Dataset/BERT_training_V3.json"
test_path = "/Users/anshmadan/Desktop/headline-classifier/hierarchical-classifier/Dataset/TSLabels.json"

def analyze_dataset(path, name):
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"--- Analysis for {name} ---")
    print(f"Total samples: {len(data)}")
    
    label_counts = Counter()
    topic_counts = Counter()
    lengths = []
    text_content = []
    
    for item in data:
        if item.get("framework1_feature1") == 1:
            label = "Central (F1)"
        elif item.get("framework1_feature2") == 1:
            label = "Peripheral (F2)"
        elif item.get("framework1_feature3") == 1:
            label = "Neutral (F3)"
        else:
            label = "Unknown"
        label_counts[label] += 1
        
        topic = item.get("topic", "Unknown")
        topic_counts[topic] += 1
        
        text = item.get("text", "")
        lengths.append(len(text.split()))
        text_content.append(text.lower())
        
    print(f"Label Distribution: {label_counts}")
    print(f"Topic Distribution: {topic_counts}")
    print(f"Average Words per Sentence: {sum(lengths)/len(lengths) if lengths else 0:.2f}")
    
    all_words = " ".join(text_content)
    words = re.findall(r'\b\w+\b', all_words)
    stopwords = {"the", "a", "an", "is", "of", "and", "in", "to", "that", "this", "it", "for", "on", "with", "as", "by", "are", "have", "from", "your", "has", "can", "or", "people", "not"}
    filtered_words = [w for w in words if w not in stopwords]
    most_common = Counter(filtered_words).most_common(20)
    print(f"Most common words: {most_common}\n")

analyze_dataset(train_path, "BERT_training_V3.json")
analyze_dataset(test_path, "TSLabels.json")
