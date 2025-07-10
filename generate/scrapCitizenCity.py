import csv, os, time, re, logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not available. Using fallback data only.")

# Pastikan folder log tersedia
os.makedirs("scrapper", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scrapper/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EthicalIndonesiaCityScraper:
    def __init__(self):
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Educational Research Bot 1.0 (Indonesian Demographics Study)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'id,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            })
        else:
            self.session = None

        # Rate limiting settings
        self.request_delay = 2.0
        self.last_request_time = 0

        # Data storage
        self.cities_data = []

        # Ethical guidelines
        self.max_retries = 3
        self.timeout = 30

    def check_robots_txt(self, base_url: str) -> bool:
        try:
            robots_url = urljoin(base_url, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            user_agent = self.session.headers.get('User-Agent', '*')
            can_fetch = rp.can_fetch(user_agent, base_url)

            logger.info(f"Robots.txt check for {base_url}: {'Allowed' if can_fetch else 'Disallowed'}")
            return can_fetch

        except Exception as e:
            logger.warning(f"Could not check robots.txt for {base_url}: {e}")
            return True  # Assume allowed if can't check

    def rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def make_request(self, url: str, retries: int = 0):
        self.rate_limit()

        if not HAS_REQUESTS or not self.session:
            logger.warning("Requests library not available, skipping web request")
            return None

        try:
            logger.debug(f"Making request to: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response

        except Exception as e:
            if retries < self.max_retries:
                wait_time = (retries + 1) * 2  # Exponential backoff
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                return self.make_request(url, retries + 1)
            else:
                logger.error(f"Request failed after {self.max_retries} retries: {e}")
                return None

    def extract_city_data_from_wikipedia(self) -> List[Dict]:
        logger.info("Extracting city data from Wikipedia Indonesia...")

        url = "https://id.wikipedia.org/wiki/Daftar_kota_di_Indonesia_menurut_jumlah_penduduk"

        # Check robots.txt
        if not self.check_robots_txt("https://id.wikipedia.org"):
            logger.error("Robots.txt disallows scraping Wikipedia")
            return []

        response = self.make_request(url)
        if not response:
            return []

        content = response.text
        cities_data = []

        table_pattern = r'<tr[^>]*>.*?</tr>'
        table_rows = re.findall(table_pattern, content, re.DOTALL)

        for row in table_rows:
            city_data = self._parse_wikipedia_row(row)
            if city_data:
                cities_data.append(city_data)
                logger.debug(f"Extracted: {city_data.get('city_name', 'Unknown')}")

        logger.info(f"Successfully extracted {len(cities_data)} cities from Wikipedia")
        return cities_data

    def _parse_wikipedia_row(self, row_html: str) -> Optional[Dict]:
        try:
            city_match = re.search(r'title="Kota ([^"]+)"[^>]*>([^<]+)</a>', row_html)
            if not city_match:
                return None

            city_name = city_match.group(2).strip()

            province_match = re.search(r'title="([^"]*(?:Jawa|Sumatra|Kalimantan|Sulawesi|Papua|Bali|Nusa Tenggara|Maluku|Aceh|Riau|Jambi|Bengkulu|Lampung|Bangka|Gorontalo|Banten)[^"]*)"', row_html)
            province = province_match.group(1) if province_match else "Unknown"

            pop_matches = re.findall(r'(\d{1,3}(?:\.\d{3})*)', row_html)
            population_2024 = None

            for pop_str in pop_matches:
                pop_num = int(pop_str.replace('.', ''))
                if 10000 <= pop_num <= 50000000:  # Reasonable city population range
                    population_2024 = pop_num
                    break

            if not population_2024:
                return None

            return {
                'city_name': city_name,
                'province': province,
                'population_2024': population_2024,
                'data_source': 'Wikipedia Indonesia',
                'extraction_date': datetime.now().strftime('%Y-%m-%d')
            }

        except Exception as e:
            logger.debug(f"Error parsing row: {e}")
            return None

    def save_to_csv(self, data: List[Dict], filename: str = '../dataset/citizenCity.csv'):
        if not data:
            logger.error("No data to save!")
            return False

        logger.info(f"Saving {len(data)} city records to {filename}")

        # Define CSV columns in Indonesian
        fieldnames = [
            'kota',
            'provinsi',
            'populasi_2024',
            'tipe_kota',
            'perkiraan_kepadatan',
            'wilayah',
            'sumber_data',
            'tanggal_ekstraksi',
            'kualitas_data'
        ]

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Sort by population (descending)
                sorted_data = sorted(data, key=lambda x: x.get('population_2024', 0), reverse=True)

                for city in sorted_data:
                    # Enrich data with additional fields
                    enriched_city = self._enrich_city_data(city)

                    # Map to Indonesian column names
                    csv_row = {
                        'kota': enriched_city.get('city_name', ''),
                        'provinsi': enriched_city.get('province', ''),
                        'populasi_2024': enriched_city.get('population_2024', ''),
                        'tipe_kota': enriched_city.get('city_type', ''),
                        'perkiraan_kepadatan': enriched_city.get('population_density_estimate', ''),
                        'wilayah': enriched_city.get('region', ''),
                        'sumber_data': enriched_city.get('data_source', ''),
                        'tanggal_ekstraksi': enriched_city.get('extraction_date', ''),
                        'kualitas_data': enriched_city.get('data_quality', '')
                    }
                    writer.writerow(csv_row)

            logger.info(f"Successfully saved data to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return False

    def _enrich_city_data(self, city: Dict) -> Dict:
        enriched = city.copy()

        # Add city type if not present
        if 'type' not in enriched:
            enriched['city_type'] = 'Kota'
        else:
            enriched['city_type'] = enriched.get('type', 'Kota')

        # Estimate population density (rough estimate based on typical Indonesian city sizes)
        population = enriched.get('population_2024', 0)
        if population > 2000000:
            density_estimate = "Very High (>5000/km²)"
        elif population > 1000000:
            density_estimate = "High (2000-5000/km²)"
        elif population > 500000:
            density_estimate = "Medium (1000-2000/km²)"
        elif population > 100000:
            density_estimate = "Low (500-1000/km²)"
        else:
            density_estimate = "Very Low (<500/km²)"

        enriched['population_density_estimate'] = density_estimate

        # Determine region based on province
        province = enriched.get('province', '').lower()
        if any(island in province for island in ['jawa', 'jakarta', 'banten', 'yogyakarta']):
            region = 'Java'
        elif any(island in province for island in ['sumatra', 'sumatera', 'aceh', 'riau', 'jambi', 'bengkulu', 'lampung', 'bangka']):
            region = 'Sumatra'
        elif any(island in province for island in ['kalimantan']):
            region = 'Kalimantan'
        elif any(island in province for island in ['sulawesi']):
            region = 'Sulawesi'
        elif any(island in province for island in ['papua']):
            region = 'Papua'
        elif any(island in province for island in ['bali']):
            region = 'Bali'
        elif any(island in province for island in ['nusa tenggara']):
            region = 'Nusa Tenggara'
        elif any(island in province for island in ['maluku']):
            region = 'Maluku'
        elif any(island in province for island in ['gorontalo']):
            region = 'Sulawesi'
        else:
            region = 'Other'

        enriched['region'] = region

        # Data quality assessment
        if enriched.get('data_source', '').startswith('BPS'):
            quality = 'High'
        elif 'Wikipedia' in enriched.get('data_source', ''):
            quality = 'Medium'
        else:
            quality = 'Low'

        enriched['data_quality'] = quality

        return enriched

    def generate_summary_report(self, data: List[Dict]):
        if not data:
            logger.error("No data available for summary report")
            return

        print("\n" + "="*80)
        print("INDONESIAN CITIES POPULATION DATA SUMMARY")
        print("="*80)

        # Basic statistics
        total_cities = len(data)
        total_population = sum(city.get('population_2024', 0) for city in data)
        avg_population = total_population / total_cities if total_cities > 0 else 0

        print(f"Total Cities: {total_cities:,}")
        print(f"Total Urban Population: {total_population:,}")
        print(f"Average City Population: {avg_population:,.0f}")

        # Top 10 cities
        print(f"\nTop 10 Most Populous Cities:")
        print("-" * 80)
        sorted_cities = sorted(data, key=lambda x: x.get('population_2024', 0), reverse=True)

        for i, city in enumerate(sorted_cities[:10], 1):
            pop = city.get('population_2024', 0)
            province = city.get('province', 'Unknown')
            city_type = city.get('type', 'Kota')
            print(f"{i:2d}. {city['city_name']:<25} {pop:>12,} ({province}) [{city_type}]")

        # Regional distribution
        print(f"\nRegional Distribution:")
        print("-" * 40)
        regions = {}
        for city in data:
            enriched = self._enrich_city_data(city)
            region = enriched['region']
            if region not in regions:
                regions[region] = {'count': 0, 'population': 0}
            regions[region]['count'] += 1
            regions[region]['population'] += city.get('population_2024', 0)

        for region, stats in sorted(regions.items(), key=lambda x: x[1]['population'], reverse=True):
            print(f"  {region:<15}: {stats['count']:3d} cities, {stats['population']:>12,} people")

        # Data sources
        print(f"\nData Sources:")
        print("-" * 30)
        sources = {}
        for city in data:
            source = city.get('data_source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1

        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} cities")

        print(f"\nData extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    def run_scraping(self):
        logger.info("Starting ethical Indonesian city population data collection...")

        # Try to scrape from Wikipedia only
        wikipedia_data = []
        try:
            wikipedia_data = self.extract_city_data_from_wikipedia()
        except Exception as e:
            logger.error(f"Wikipedia scraping failed: {e}")

        # Use Wikipedia data if available and sufficient
        if wikipedia_data and len(wikipedia_data) > 50:
            logger.info(f"Using Wikipedia data ({len(wikipedia_data)} cities)")
            self.cities_data = wikipedia_data
        else:
            logger.error("No sufficient city data was collected from Wikipedia!")
            self.cities_data = []

        # Save to CSV
        if self.cities_data:
            success = self.save_to_csv(self.cities_data)
            if success:
                self.generate_summary_report(self.cities_data)
            return success
        else:
            logger.error("No city data was collected!")
            return False

def main():
    os.makedirs("../dataset", exist_ok=True)
    scraper = EthicalIndonesiaCityScraper()
    success = scraper.run_scraping()

    if success:
        print(f"\nData collection completed successfully!")
        print(f"File saved as: citizenCity.csv")
        print(f"Check scraper.log for detailed logs")
    else:
        print(f"\nData collection failed!")
        print(f"Check scraper.log for error details")

if __name__ == "__main__":
    main()