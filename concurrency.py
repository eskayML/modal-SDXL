import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from uuid import uuid4

def send_post_request(url, data, save_folder):
    response = requests.post(url, json=data)
    if response.status_code == 200:
        # Generate a unique filename using UUID
        filename = f"{uuid4()}.jpg"  # Assuming the images are JPEGs
        filepath = os.path.join(save_folder, filename)
        
        # Save the content as an image file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return filename
    else:
        return None

def main():
    url = "https://eskayml--stable-diffusion-xl-model-web-inference.modal.run/"
    data = {"prompt": 'toy, children running in the snow'}
    num_requests = 10
    save_folder = "downloaded_images"  # Folder to save the images

    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_post_request, url, data, save_folder) for _ in range(num_requests)]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Image saved as: {result}")
                else:
                    print("Failed to save image")
            except Exception as e:
                print(f"Request failed with error: {str(e)}")

if __name__ == "__main__":
    main()