import asyncio
import aiohttp
import aiofiles
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import logging
import ssl

# Set up logging
logging.basicConfig(filename='scraper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Base URLs
BASE_URLS = [
    "https://www.almeezan.qa/LawView.aspx?opt&LawID={}&language=ar",
    "https://www.almeezan.qa/RulingView.aspx?opt&RulID={}&language=ar",
    "https://www.almeezan.qa/ViewAgreement.aspx?opt&agrID={}&language=ar",
    "https://www.almeezan.qa/OpinionView.aspx?opt&OpID={}&language=ar",
]

# Semaphore to limit concurrent requests
MAX_CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Create a custom SSL context that doesn't verify certificates
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def fetch_url(session, url, folder_type, number):
    file_path = f"{folder_type}/{number}.txt"
    
    # Check if file already exists
    if os.path.exists(file_path):
        logging.info(f"File {file_path} already exists, skipping.")
        return

    try:
        async with semaphore:
            async with session.get(url, ssl=ssl_context) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    text_content = soup.get_text()
                    
                    async with aiofiles.open(file_path, mode='w', encoding='utf-8') as f:
                        await f.write(text_content)
                    logging.info(f"Successfully saved content from {url} to {file_path}")
                else:
                    logging.warning(f"Failed to fetch {url}. Status code: {response.status}")
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Connection error for {url}: {str(e)}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout error for {url}")
    except Exception as e:
        logging.error(f"Error processing {url}: {str(e)}")

async def main():
    start_time = time.time()
    
    # Create folders if they don't exist
    for folder in ['LawView', 'RulingView', 'ViewAgreement', 'OpinionView']:
        os.makedirs(folder, exist_ok=True)

    # Configure client session with SSL context and longer timeout
    timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for base_url in BASE_URLS:
            folder_type = urlparse(base_url).path.split('/')[1].split('.')[0]
            for number in tqdm(range(100001)):  # 0 to 10,000
                url = base_url.format(number)
                task = asyncio.ensure_future(fetch_url(session, url, folder_type, number))
                tasks.append(task)
        
        await asyncio.gather(*tasks)

    end_time = time.time()
    logging.info(f"Script completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
