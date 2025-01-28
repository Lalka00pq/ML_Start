import requests

path_to_image = r"D:\Repositories\arscis-deep-learning\src\.test_datasets\wagons_numbers\vmtp1_20210610051511_0515_1_00244000.jpg"
images = {"image": open(path_to_image, "rb")}

response = requests.post("http://127.0.0.1:8000/resize_image", files=images)
print(response.json())
