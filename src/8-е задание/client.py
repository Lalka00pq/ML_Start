import requests

path_to_image = r"C:\Users\User\Desktop\ML_projects\ML_Start_Caisar\ML_Start\src\7-е задание\images.jpg"
images = {"image": open(path_to_image, "rb")}

response = requests.post("http://127.0.0.1:8000/resize_image", files=images)
print(response.json())
