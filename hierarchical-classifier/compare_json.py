import json
import sys

def compare():
    with open('Dataset/Cross annotation/Anagha.json', 'r') as f:
        anagha = json.load(f)

    with open('Dataset/Cross annotation/Ansh.json', 'r') as f:
        ansh = json.load(f)

    # Use the text itself as the key to avoid ID mismatches
    anagha_dict = {item['text'].strip(): item for item in anagha}
    ansh_dict = {item['text'].strip(): item for item in ansh}

    diffs = []
    
    # Check all keys in Anagha
    for text_key in anagha_dict:
        if text_key not in ansh_dict:
            diffs.append(f"Headline in Anagha but NOT in Ansh: {text_key}")
            continue
            
        a_item = anagha_dict[text_key]
        b_item = ansh_dict[text_key]
        
        # compare labels
        f1a = a_item.get('framework1_feature1', 0)
        f2a = a_item.get('framework1_feature2', 0)
        f3a = a_item.get('framework1_feature3', 0)
        
        f1b = b_item.get('framework1_feature1', 0)
        f2b = b_item.get('framework1_feature2', 0)
        f3b = b_item.get('framework1_feature3', 0)
        
        if (f1a, f2a, f3a) != (f1b, f2b, f3b):
            diffs.append(f"LABEL MISMATCH for '{text_key[:40]}...' -> Anagha: {(f1a, f2a, f3a)}, Ansh: {(f1b, f2b, f3b)}")

    # Check for keys in Ansh not in Anagha
    for text_key in ansh_dict:
        if text_key not in anagha_dict:
            diffs.append(f"Headline in Ansh but NOT in Anagha: {text_key}")

    print(f"Total differences found: {len(diffs)}")
    for d in diffs[:20]: # show first 20 to avoid overwhelming
        print(d)
    if len(diffs) > 20:
        print(f"...and {len(diffs) - 20} more.")

if __name__ == '__main__':
    compare()
