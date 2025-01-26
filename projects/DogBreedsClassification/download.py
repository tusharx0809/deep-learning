import kagglehub

# Download latest version
path = kagglehub.dataset_download("eward96/dog-breed-images")

print("Path to dataset files:", path)