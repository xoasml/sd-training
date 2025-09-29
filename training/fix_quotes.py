import os

# 원본 CSV 경로 입력
input_path = r"C:\Users\xoasm\Documents\sd-training\training\dataset\dataset.csv"
output_path = r"C:\Users\xoasm\Documents\sd-training\training\dataset\dataset_fixed.csv"

def fix_quotes_in_csv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    # 굽은 따옴표 → 직선 따옴표로 변환
    fixed_data = data.replace("“", '"').replace("”", '"')

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write(fixed_data)

    print(f"[완료] 굽은 따옴표를 직선 따옴표로 변환했습니다.\n결과 파일: {output_file}")

if __name__ == "__main__":
    if os.path.exists(input_path):
        fix_quotes_in_csv(input_path, output_path)
    else:
        print(f"[에러] 파일을 찾을 수 없습니다: {input_path}")