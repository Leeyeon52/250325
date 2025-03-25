import pandas as pd

# 원본 파일 경로
train_path = "train.csv"
test_path = "test.csv"
sample_submission_path = "sample_submission.csv"

# 데이터 로드
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_submission_path)

# 첫 1000개 데이터만 슬라이싱
df_train_1000 = df_train.head(1000)
df_test_1000 = df_test.head(1000)
df_sample_1000 = df_sample.head(1000)

# 새로운 파일로 저장
new_train_path = "train_1000.csv"
new_test_path = "test_1000.csv"
new_submission_path = "sample_submission_1000.csv"

df_train_1000.to_csv(new_train_path, index=False)
df_test_1000.to_csv(new_test_path, index=False)
df_sample_1000.to_csv(new_submission_path, index=False)

print(f"New files created: {new_train_path}, {new_test_path}, {new_submission_path}")
