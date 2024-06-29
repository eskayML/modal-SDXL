import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_post_request(url, data):
    response = requests.post(url, json=data)
    return response

def main():
    url = "https://eskayml--stable-diffusion-xl-model-web-inference.modal.run"
    data = {
        "prompt": 'toy man running in a field, wearing a batman costume'
    }
    num_requests = 10

    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_post_request, url, data) for _ in range(num_requests)]

        for future in as_completed(futures):
            try:
                response = future.result()
                print(f"Request completed with status code: {response.status_code}")
            except Exception as e:
                print(f"Request failed with error: {str(e)}")

if __name__ == "__main__":
    main()