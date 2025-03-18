import os
import re
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# 1. 텍스트 데이터 추출 함수 (Extract)
def extract_text(**kwargs):
    # 실제 파일 경로가 존재하면 해당 파일에서 데이터를 읽고, 없으면 예시 텍스트 사용
    file_path = '/path/to/input_text.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # 예시 텍스트 데이터
        text = "Hello World! This is a sample text data for ETL processing."
    logging.info("텍스트 데이터 추출 완료")
    return text

# 2. 텍스트 데이터 변환 함수 (Transform)
def transform_text(**kwargs):
    ti = kwargs['ti']
    text = ti.xcom_pull(task_ids='extract_text_task')
    # 텍스트 전처리: 소문자 변환, 특수문자 제거, 단어 토큰화
    text_lower = text.lower()
    text_clean = re.sub(r'[^a-z\s]', '', text_lower)
    tokens = text_clean.split()
    logging.info("텍스트 데이터 변환 완료: %s", tokens)
    return tokens

# 3. 텍스트 데이터 검증 함수 (Validation)
def validate_text(**kwargs):
    ti = kwargs['ti']
    tokens = ti.xcom_pull(task_ids='transform_text_task')
    # 데이터 검증: 토큰 리스트가 비어있지 않은지, 최소 단어 수가 3개 이상인지 확인
    if not tokens:
        raise ValueError("텍스트 데이터 검증 실패: 토큰 리스트가 비어있습니다.")
    if len(tokens) < 3:
        raise ValueError("텍스트 데이터 검증 실패: 단어 수가 부족합니다.")
    logging.info("텍스트 데이터 검증 완료: 단어 수 %d", len(tokens))
    return tokens

# 4. 텍스트 데이터 적재 함수 (Load)
def load_text(**kwargs):
    ti = kwargs['ti']
    tokens = ti.xcom_pull(task_ids='transform_text_task')
    # 처리된 텍스트 데이터를 로컬 파일에 저장
    output_path = '/path/to/processed_text.txt'
    processed_text = " ".join(tokens)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    logging.info("텍스트 데이터 적재 완료: %s", output_path)

# Airflow 기본 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email': ['alert@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의: 매일 실행되도록 스케줄 설정
dag = DAG(
    'etl_text_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

# 각 태스크 정의
extract_text_task = PythonOperator(
    task_id='extract_text_task',
    python_callable=extract_text,
    provide_context=True,
    dag=dag
)

transform_text_task = PythonOperator(
    task_id='transform_text_task',
    python_callable=transform_text,
    provide_context=True,
    dag=dag
)

validate_text_task = PythonOperator(
    task_id='validate_text_task',
    python_callable=validate_text,
    provide_context=True,
    dag=dag
)

load_text_task = PythonOperator(
    task_id='load_text_task',
    python_callable=load_text,
    provide_context=True,
    dag=dag
)

# 태스크 의존성: extract → transform → validate → load 순서로 실행
# extract_text_task >> transform_text_task >> validate_text_task >> load_text_task