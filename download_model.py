import requests
import os

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    file_id = "1F0O0eQi8rNnICfQXm5UIdHtYJ0PiDMXv"
    destination = "model.pkl"

    if not os.path.exists(destination):
        print("Downloading model from Google Drive...")
        download_file_from_google_drive(file_id, destination)
        print("Model downloaded!")
    else:
        print("Model already exists. Skipping download.")
