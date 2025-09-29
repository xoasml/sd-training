import csv
import os

csv_path = r"C:/Users/xoasm/Documents/sd-training/training/dataset/dataset_fixed.csv"
output_dir = r"C:/Users/xoasm/Documents/sd-training/training/dataset_txt"

os.makedirs(output_dir, exist_ok=True)

with open(csv_path, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        image_file = row["image"]
        caption = row["text"]

        # 파일명만 추출 (경로 제거)
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        txt_path = os.path.join(output_dir, base_name + ".txt")

        print(f"생성 중: {txt_path}")  # 디버깅용 출력

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption.strip())

print(f"✅ 변환 완료: {output_dir}")