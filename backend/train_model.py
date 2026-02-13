import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["Su", "Sy", "E", "G", "mu", "Ro"]


def load_material_data(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")
	df = pd.read_csv(csv_path)
	missing = set(FEATURE_COLUMNS + ["Use", "Material"]) - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in CSV: {sorted(missing)}")
	df["Use"] = df["Use"].astype(int)
	return df


def train_model(df: pd.DataFrame, test_size: float, random_state: int):
	x = df[FEATURE_COLUMNS]
	y = df["Use"].astype(int)
	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)
	model = GradientBoostingClassifier(random_state=random_state)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	metrics = {
		"accuracy": accuracy_score(y_test, y_pred),
		"f1": f1_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred),
		"recall": recall_score(y_test, y_pred),
	}
	if len(set(y_test)) > 1:
		metrics["roc_auc"] = roc_auc_score(y_test, y_pred)
	else:
		metrics["roc_auc"] = None

	report = classification_report(y_test, y_pred)
	return model, metrics, report


def material_recommend_for_selection(input_values, model, material_df):
	inp = pd.DataFrame([input_values])
	pred = model.predict(inp)
	if pred[0] == 0:
		return "No material suitable for selection"

	filtered_df = material_df[
		(material_df["Su"] == input_values["Su"])
		& (material_df["Sy"] == input_values["Sy"])
		& (material_df["E"] == input_values["E"])
		& (material_df["G"] == input_values["G"])
		& (material_df["mu"] == input_values["mu"])
		& (material_df["Ro"] == input_values["Ro"])
	]
	materials = list(filtered_df[filtered_df["Use"] == 1]["Material"].unique())
	return f"These materials are suitable for selection: {materials}"


def parse_args():
	parser = argparse.ArgumentParser(description="Train material selection model.")
	parser.add_argument(
		"--data",
		default="data/material.csv",
		help="Path to material.csv (default: data/material.csv)",
	)
	parser.add_argument(
		"--model-out",
		default="model/material_gbc.joblib",
		help="Output path for trained model bundle",
	)
	parser.add_argument(
		"--test-size",
		type=float,
		default=0.30,
		help="Test split size (default: 0.30)",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=0,
		help="Random state for reproducibility (default: 0)",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	csv_path = Path(args.data)
	model_path = Path(args.model_out)
	model_path.parent.mkdir(parents=True, exist_ok=True)

	df = load_material_data(csv_path)
	model, metrics, report = train_model(df, args.test_size, args.random_state)

	print("Metrics:")
	for key, value in metrics.items():
		print(f"  {key}: {value}")
	print("\nClassification report:\n")
	print(report)

	bundle = {
		"model": model,
		"feature_columns": FEATURE_COLUMNS,
	}
	joblib.dump(bundle, model_path)
	print(f"\nSaved model bundle to: {model_path}")


if __name__ == "__main__":
	main()
