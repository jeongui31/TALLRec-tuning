import json
import pandas as pd
f = open('../data/movie/u.data', 'r')
data = f.readlines()
f = open('../data/movie/u.item', 'r', encoding='ISO-8859-1')
movies = f.readlines()
f = open('../data/movie/u.user', 'r')
users = f.readlines()

movie_names = [_.split('|')[1] for _ in movies]  # movie_names[0] = 'Toy Story (1995)'
user_ids = [_.split('|')[0] for _ in users]  # user_ids[0] = '1'
movie_ids = [_.split('|')[0] for _ in movies]  # movie_ids[0] = '1'
interaction_dicts = dict()  
for line in data:
    user_id, movie_id, rating, timestamp = line.split('\t')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
        }
    interaction_dicts[user_id]['movie_id'].append(movie_id)
    interaction_dicts[user_id]['rating'].append(int(rating))  # Preserve the original rating as an integer
    interaction_dicts[user_id]['timestamp'].append(timestamp)

with open('item_mapping.csv', 'w') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['movie_id', 'movie_name'])
    for i, name in enumerate(movie_names):
        writer.writerow([i + 1, name])

sequential_interaction_list = []
seq_len = 10
for user_id in interaction_dicts:
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'])
    temp = sorted(temp, key=lambda x: x[2])
    result = zip(*temp)
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'] = [list(_) for _ in result]
    for i in range(10, len(interaction_dicts[user_id]['movie_id'])):
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_id'][i-seq_len:i], interaction_dicts[user_id]['rating'][i-seq_len:i], interaction_dicts[user_id]['movie_id'][i], interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )
    
sequential_interaction_list = sequential_interaction_list[-10000:]  # 10000 records

import csv
# Save the CSV file for baselines
with open('../data/movie_rating/train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list)*0.8)])
with open('../data/movie_rating/valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.8):int(len(sequential_interaction_list)*0.9)])
with open('../data/movie_rating/test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp'])
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.9):])
    
def csv_to_json(input_path, output_path):
    data = pd.read_csv(input_path)

    # Drop rows with NaN values
    data = data.dropna()

    json_list = []
    for index, row in data.iterrows():
        # Convert history_movie_id and history_rating from string representation of lists to actual lists
        row['history_movie_id'] = eval(row['history_movie_id'])
        row['history_rating'] = eval(row['history_rating'])
        L = len(row['history_movie_id'])
        
        categories = {
            "strongly dislike": [],
            "dislike": [],
            "neutral": [],
            "like": [],
            "strongly like": []
        }
        
        # Correct categorization of ratings
        for i in range(L):
            movie_name = movie_names[int(row['history_movie_id'][i]) - 1]
            rating = int(row['history_rating'][i])  # Ensure rating is an integer
            
            if rating == 1:
                categories["strongly dislike"].append(movie_name)
            elif rating == 2:
                categories["dislike"].append(movie_name)
            elif rating == 3:
                categories["neutral"].append(movie_name)
            elif rating == 4:
                categories["like"].append(movie_name)
            elif rating == 5:
                categories["strongly like"].append(movie_name)
            else:
                print(f"Warning: Unexpected rating value {rating} for movie {movie_name}")

        # Convert categories to strings for input
        input_parts = []
        for category, movies in categories.items():
            movie_str = ", ".join([f"\"{movie}\"" for movie in movies]) if movies else "[]"
            input_parts.append(f"{category}: [{movie_str}]")
        input_str = "\n".join(input_parts)
        
        target_movie = movie_names[int(row['movie_id']) - 1]
        target_preference = int(row['rating'])
        target_preference_str = "Yes." if target_preference > 3 else "No."
        target_movie_str = f"\"{target_movie}\""
        
        json_list.append({
            "instruction": "Given the user's preference ratings from strongly dislike to strongly like, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\".",
            "input": f"{input_str}\nWhether the user will like the target movie {target_movie_str}?",
            "output": target_preference_str,
        })
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

# Generate the JSON file for the TALLRec
csv_to_json('../data/movie_rating/train.csv', '../data/movie_rating/train.json')
csv_to_json('../data/movie_rating/valid.csv', '../data/movie_rating/valid.json')
csv_to_json('../data/movie_rating/test.csv', '../data/movie_rating/test.json')
