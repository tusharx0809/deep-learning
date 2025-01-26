import kagglehub

# Download latest version
path = kagglehub.dataset_download("abhinavkrjha/dog-breed-classification")

print("Path to dataset files:", path)