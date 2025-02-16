import requests

path_to_image = r".\images.jpg"
images = {"image": open(path_to_image, "rb")}

response = requests.post("http://127.0.0.1:8000/find_objects", files=images)
print(response.json())
