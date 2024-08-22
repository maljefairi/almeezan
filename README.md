# almeezan

almeezan is a comprehensive Python-based tool designed to scrape, clean, and archive legal content from Almeezan.qa, Qatar's legal portal. This project aims to facilitate access to public legal information for research and educational purposes.

## Author

Dr. Mohammed Al-Jefairi  
Email: maljefairi@sidramail.com  
GitHub: [maljefairi](https://github.com/maljefairi)

## Features

- Asynchronous web scraping using `asyncio` and `aiohttp`
- Multi-core utilization for enhanced performance
- Rate limiting to prevent server overload
- Comprehensive error handling and logging
- Ability to resume interrupted scraping sessions
- UTF-8 encoding for proper Arabic text handling
- Real-time progress tracking
- Post-scraping content cleaning and formatting

## Project Structure

The project consists of two main scripts:

1. `scraper.py`: Handles the web scraping process
2. `cleaner.py`: Processes and cleans the scraped data

### Cleaner Functions

The `cleaner.py` script includes the following key functions:

- `clean_file(file_path)`: Processes a single file, removing unwanted content and duplicates.
- `scan_and_clean_folders()`: Iterates through all scraped files in specified folders and applies the cleaning process.

The cleaning process includes:
- Removal of specific unwanted patterns (e.g., metadata, print buttons, disclaimers)
- Elimination of excessive whitespace and newlines
- Removal of exact duplicates while preserving the original order of content
- Preservation of important content, including titles and unique information

## Requirements

- Python 3.7+
- aiohttp
- aiofiles
- beautifulsoup4
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/maljefairi/almeezan.git
   cd almeezan
   ```

2. Install the required packages:
   ```
   pip install aiohttp aiofiles beautifulsoup4 tqdm
   ```

## Usage

1. Run the scraper script:
   ```
   python scraper.py
   ```
   This will create folders for each content type and save the raw scraped data as text files.

2. After scraping, run the cleaner script:
   ```
   python cleaner.py
   ```
   This will process the scraped files, removing duplicates and unwanted content while preserving important information.

Progress and any errors will be logged to `scraper.log` and `cleaner.log` respectively.

## Disclaimer and Legal Considerations

This project is intended for educational and research purposes only. Please note:

1. **Educational Tool**: almeezan is developed as a learning resource to demonstrate web scraping techniques, particularly for Arabic content.

2. **Public Data**: Almeezan.qa is a governmental site providing public legal information. While this data is publicly available, users should adhere to the website's terms of service.

3. **Ethical Use**: Users must follow ethical web scraping practices, including respecting the website's robots.txt file and implementing reasonable request rates.

4. **No Warranty**: This script is provided "as is" without any warranties. The author is not responsible for any misuse or consequences arising from the use of this script.

5. **Legal Compliance**: Users are responsible for ensuring their use of this script and the obtained data complies with all applicable laws and regulations.

Always review the website's terms of service before scraping. Use this tool responsibly and ethically.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/maljefairi/almeezan/issues) if you want to contribute.

## Support

If you find this project helpful, please give it a ⭐️ on GitHub!