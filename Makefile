install:
	pip install -r requirements.txt

train:
	python src/models/train.py --data_path data/processed/processed.csv --output_dir outputs --model random_forest

predict:
	python -c "import pandas as pd; from src.models.predict import predict_dataframe; df=pd.read_csv('data/processed/processed.csv').drop(columns=['readmitted']); print(predict_dataframe(df))"

test:
	pytest src/tests/ -v
