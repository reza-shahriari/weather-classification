import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
from io import BytesIO
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def download_weather_images(search_term, output_dir, num_images=100):
    os.makedirs(output_dir, exist_ok=True)
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Use DuckDuckGo images instead of Google (more reliable for scraping)
    search_url = f"https://duckduckgo.com/?q={search_term}&iax=images&ia=images"
    driver.get(search_url)
    
    # Wait for images to load
    time.sleep(3)
    
    # Click to show more images
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    
    # Get image URLs
    img_elements = driver.find_elements(By.CSS_SELECTOR, 'img.tile--img__img')
    img_urls = [img.get_attribute('src') for img in img_elements if img.get_attribute('src')]
    
    driver.quit()
    
    def download_image(url, path):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img.save(path, 'JPEG')
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False

    successful_downloads = 0
    for idx, url in enumerate(img_urls):
        if successful_downloads >= num_images:
            break
            
        file_path = os.path.join(output_dir, f"{search_term.replace(' ', '_')}_{idx}.jpg")
        if download_image(url, file_path):
            successful_downloads += 1
            print(f"Downloaded {successful_downloads}/{num_images} images")
    
    return successful_downloads

def create_weather_dataset():
    weather_conditions = [
        'sunny weather landscape',
        'rainy weather landscape',
        'cloudy weather landscape',
        'snowy weather landscape',
        'foggy weather landscape',
        'stormy weather landscape'
    ]
    
    base_dir = 'dataset'
    os.makedirs(base_dir, exist_ok=True)
    
    for condition in weather_conditions:
        condition_dir = os.path.join(base_dir, condition.split()[0])
        print(f"\nCollecting images for {condition}...")
        downloaded = download_weather_images(condition, condition_dir, num_images=100)
        print(f"Successfully downloaded {downloaded} images for {condition}")

if __name__ == "__main__":
    create_weather_dataset()
