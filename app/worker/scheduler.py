import schedule, time, subprocess, logging
logging.basicConfig(level=logging.INFO)

def run_pipeline():
    subprocess.run(["python", "scripts/data_pipeline/d_pipeline.py"])
    subprocess.run(["python", "scripts/prediction_pipeline/predict.py"])

schedule.every(6).hours.do(run_pipeline)
logging.info("🔁 Scheduler started...")
while True:
    schedule.run_pending()
    time.sleep(60)