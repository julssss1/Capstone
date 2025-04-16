import pickle

# Replace 'filename.pkl' with your actual file name
with open('svm_sign_model_normalized.pkl', 'rb') as file:
    data = pickle.load(file)

# Check what type of data it is
print("Type of data:", type(data))

# If it's a dictionary, print the keys
if isinstance(data, dict):
    print("Keys:", data.keys())

# If it's a list or tuple, show the first few items
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    print("First item:", data[0])

# Print a sample
print("Sample content:")
print(data)