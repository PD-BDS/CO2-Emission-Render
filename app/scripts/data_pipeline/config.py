from pathlib import Path
import logging

# Ensure database directory exists
Path("database").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(filename='logs/data_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Constants
DB_PATH = "database/co2_emission.db"
BASE_URL = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
RELEVANT_COLUMNS = [
    'Minutes1DK', 'ProductionGe100MW', 'ProductionLt100MW', 'SolarPower',
    'OffshoreWindPower', 'OnshoreWindPower', 'Exchange_Sum', 'CO2Emission'
]
