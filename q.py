import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 파일 경로 설정
train_path = "train.csv"
test_path = "test.csv"
sample_submission_path = "sample_submission.csv"
output_path = "submission.csv"

# 파일 존재 여부 확인
def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {path}. 파일을 확인하세요.")

for path in [train_path, test_path, sample_submission_path]:
    check_file_exists(path)

# 데이터 로드
print("📂 데이터 로드 중...")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample = pd.read_csv(sample_submission_path)

# URL 컬럼명 변경 (자동 감지)
def rename_url_column(df):
    for col in df.columns:
        if 'url' in col.lower():
            df.rename(columns={col: 'URL'}, inplace=True)
            return df
    raise KeyError("❌ 'URL' 컬럼을 찾을 수 없습니다. 파일을 확인하세요.")

try:
    print("🔍 URL 컬럼명 변경 중...")
    df_train = rename_url_column(df_train)
    df_test = rename_url_column(df_test)
except KeyError as e:
    print(e)
    exit(1)

# 결측값 및 중복 제거
print("🗑️ 결측값 및 중복 제거 중...")
df_train = df_train.dropna().drop_duplicates()
df_test = df_test.dropna().drop_duplicates()

# 특징(X)와 타겟(y) 분리
if 'malicious' not in df_train.columns:
    raise KeyError("❌ 'malicious' 컬럼이 train.csv에 없습니다. 파일을 확인하세요.")
X_texts = df_train['URL'].astype(str)
y = df_train['malicious'].astype(int)

# TF-IDF 벡터화
print("📊 TF-IDF 벡터화 중...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
X_tfidf = vectorizer.fit_transform(X_texts)
test_tfidf = vectorizer.transform(df_test['URL'].astype(str))

# 데이터 분할
print("✂️ 데이터 분할 중...")
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습 및 하이퍼파라미터 튜닝
rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV로 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200, 300],  # 트리의 개수
    'max_depth': [10, 20, 50, None],  # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],  # 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]  # 리프 노드가 되기 위한 최소 샘플 수
}

# GridSearchCV 객체 정의 및 학습
print("🔍 GridSearchCV 학습 중...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("GridSearchCV 학습 완료")

# 최적의 모델 사용
best_rf_model = grid_search.best_estimator_

# 검증 데이터 평가
print("🧪 검증 데이터 평가 중...")
y_pred_rf = best_rf_model.predict(X_val)
print("검증 데이터 정확도:", accuracy_score(y_val, y_pred_rf))

# 교차 검증 점수 확인
print("📈 교차 검증 점수 확인 중...")
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"교차 검증 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 테스트 데이터 예측
print("📝 테스트 데이터 예측 중...")
y_test_pred = best_rf_model.predict(test_tfidf)

# 제출 파일 컬럼 확인 및 정리
if 'id' not in df_sample.columns:
    df_sample.insert(0, 'id', range(1, len(df_sample) + 1))
if 'malicious' not in df_sample.columns:
    df_sample['malicious'] = y_test_pred.astype(int)
else:
    df_sample['malicious'] = y_test_pred.astype(int)

# 컬럼 순서 조정
df_sample = df_sample[['id', 'malicious']]

# 제출 파일 생성
print("📁 제출 파일 생성 중...")
df_sample.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 제출 파일 생성 완료: {output_path}")

