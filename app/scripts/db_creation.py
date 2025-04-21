import sqlite3
from pathlib import Path

DB_PATH = "database/co2_emission.db"
Path("database").mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

# 1. Create Aggregated Hourly Data Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS aggregated_data (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    TimeStamp TEXT UNIQUE NOT NULL,
    ProductionGe100MW REAL NOT NULL,
    ProductionLt100MW REAL NOT NULL,
    SolarPower REAL NOT NULL,
    OffshoreWindPower REAL NOT NULL,
    OnshoreWindPower REAL NOT NULL,
    Exchange_Sum REAL NOT NULL,
    CO2Emission REAL NOT NULL,
    data_added_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')

# 2. Create Engineered Features Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS engineered_features (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    TimeStamp TEXT UNIQUE NOT NULL,
    CO2_lag_1 REAL NOT NULL,
    CO2_lag_2 REAL NOT NULL,
    CO2_lag_3 REAL NOT NULL,
    CO2_lag_4 REAL NOT NULL,
    CO2_lag_5 REAL NOT NULL,
    CO2_rolling_mean_rolling_window_6 REAL NOT NULL,
    CO2_rolling_std_rolling_window_6 REAL NOT NULL,
    CO2_rolling_mean_rolling_window_12 REAL NOT NULL,
    CO2_rolling_std_rolling_window_12 REAL NOT NULL,
    data_added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (TimeStamp) REFERENCES aggregated_data(TimeStamp) ON DELETE CASCADE
);
''')

# 3. Create Model Metadata Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS model_table (
    Model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model_name TEXT NOT NULL,
    Hidden_size INTEGER,
    Num_layers INTEGER,
    Dropout_rate REAL,
    Learning_rate REAL,
    Version TEXT,
    Trained_on TEXT,
    Model_path TEXT,
    Model_hash TEXT,
    Created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
''')

# 4. Create Model Training Set Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS model_training_sets (
    model_id INTEGER NOT NULL,
    time_frame TEXT NOT NULL,
    PRIMARY KEY (model_id, time_frame),
    FOREIGN KEY (model_id) REFERENCES model_table(Model_id) ON DELETE CASCADE
);
''')

# 5. Create Predictions Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Model_id INTEGER NOT NULL,
    TimeStamp TEXT NOT NULL,
    Prediction REAL NOT NULL,
    Actual REAL,
    FOREIGN KEY (Model_id) REFERENCES model_table(Model_id) ON DELETE CASCADE,
    FOREIGN KEY (TimeStamp) REFERENCES aggregated_data(TimeStamp) ON DELETE CASCADE
);
''')

# 6. Create Model Evaluations Table (for tracking evaluation metrics of models)
cursor.execute('''
CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    dataset_label TEXT NOT NULL,
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    RMSE REAL,
    MAE REAL,
    MSE REAL,
    R2 REAL,
    MAPE REAL,
    Pseudo_accuracy REAL,
    FOREIGN KEY (model_id) REFERENCES model_table(Model_id) ON DELETE CASCADE
);
''')

# 7. Create New Data Log Table to Track Data Updates
cursor.execute('''
CREATE TABLE IF NOT EXISTS new_data_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    time_added DATETIME DEFAULT CURRENT_TIMESTAMP,
    record_count INTEGER NOT NULL,
    log_description TEXT
);
''')

# Optional: Create indexes for performance optimization
cursor.execute("CREATE INDEX IF NOT EXISTS idx_aggregated_timestamp ON aggregated_data(TimeStamp);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(TimeStamp);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_training_sets ON model_training_sets(model_id, time_frame);")

# Commit and close the connection
conn.commit()
conn.close()

print("✅ Database schema created in 'app/database/co2_emission.db'")
