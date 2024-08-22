# Web Scraper for Almeezan.qa

This project contains a Python script for scraping content from the Almeezan.qa website. It's designed to efficiently fetch and save content from multiple URL patterns, handling Arabic text and using asynchronous programming for improved performance.

## Author

Dr. Mohammed Al-Jefairi  
Email: maljefairi@sidramail.com  
GitHub: [maljefairi](https://github.com/maljefairi)

## Features

- Asynchronous web scraping using `asyncio` and `aiohttp`
- Multicore usage for improved performance
- Rate limiting to avoid overwhelming the target server
- Error handling and logging
- Ability to resume from where it stopped if interrupted
- Proper handling of Arabic text (UTF-8 encoding)
- Progress tracking

## Requirements

- Python 3.7+
- aiohttp
- aiofiles
- beautifulsoup4
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/maljefairi/almeezanScraper.git
   cd almeezanScraper
   ```

2. Install the required packages:
   ```
   pip install aiohttp aiofiles beautifulsoup4 tqdm
   ```

## Usage

Run the script using Python:

```
python scraper.py
```

The script will create folders for each URL type and save the content of each page as a text file. Progress and any errors will be logged to `scraper.log`.

## Disclaimer

This script is for educational purposes only. Always check a website's terms of service and robots.txt file before scraping. Use responsibly and ethically.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/maljefairi/almeezanScraper/issues) if you want to contribute.

## Show your support

Give a ⭐️ if this project helped you!
