import pandas as pd

# splits = {'train': 'NF-ToN-IoT-v2-train.csv', 'test': 'NF-ToN-IoT-v2-test.csv'}
# df = pd.read_csv("hf://datasets/Nora9029/NF-ToN-IoT-v2/" + splits["train"])



# print(f"Number of rows in the dataset: {len(df)}")
# # Save the filtered DataFrame back to a CSV file
# df.to_csv("./intrusion_data/NF-ToN-IoT-v2-train.csv", index=False)
# print("Filtered dataset saved to 'filtered_dataset.csv'")

# lets shuffle and split the dataset in 2 to reduce its size for testing purpose
df = pd.read_csv("./intrusion_data/NF-ToN-IoT-v2-train.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame
df.to_csv("./intrusion_data/NF-ToN-IoT-v2-train-shuffled.csv", index=False)