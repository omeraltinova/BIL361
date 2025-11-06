import csv
import os
from pathlib import Path


def standardize_kaggle_csv(input_csv_path: Path) -> None:
    """
    Read the Kaggle test CSV and rewrite it to match the common schema: `label,text`.
    - Drops leading index column if present (empty header)
    - Renames `sonuc` -> `label`
    - Reorders columns to `label,text`
    - Does NOT modify data values
    """
    temp_csv_path = input_csv_path.with_suffix(".tmp.csv")

    with input_csv_path.open("r", encoding="utf-8", newline="") as infile, \
            temp_csv_path.open("w", encoding="utf-8", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        try:
            header = next(reader)
        except StopIteration:
            # Empty file; write nothing but keep it as is
            return

        # Determine column indices
        # Expected current header like: ["", "text", "sonuc"]
        header_lower = [h.lower() if h is not None else "" for h in header]

        # Handle optional empty first header (index column)
        has_leading_index = len(header_lower) >= 1 and header_lower[0].strip() == ""

        # Find positions of text and label-like columns
        # "sonuc" is the label column in the Kaggle file; keep data values unchanged
        try:
            text_idx = header_lower.index("text")
        except ValueError:
            raise RuntimeError("Beklenen 'text' sütunu bulunamadı")

        # Accept both 'sonuc' and 'label' as label column names
        label_idx = None
        for candidate in ("sonuc", "label"):
            if candidate in header_lower:
                label_idx = header_lower.index(candidate)
                break
        if label_idx is None:
            raise RuntimeError("Beklenen 'sonuc' veya 'label' sütunu bulunamadı")

        # Write standardized header
        writer.writerow(["label", "text"])

        # Stream rows and write only the two needed fields, preserving content
        for row in reader:
            # Skip completely empty lines (if any)
            if not row or all(cell == "" for cell in row):
                continue

            # Ensure row has enough columns by padding
            if len(row) <= max(text_idx, label_idx):
                row = list(row) + [""] * (max(text_idx, label_idx) + 1 - len(row))

            label_value = row[label_idx]
            text_value = row[text_idx]

            # Do NOT alter label_value or text_value; just reorder
            writer.writerow([label_value, text_value])

    # Replace original file atomically
    os.replace(temp_csv_path, input_csv_path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    input_csv = repo_root / "Test" / "kaggle-test.csv"
    standardize_kaggle_csv(input_csv)


