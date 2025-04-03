from datasets import load_dataset
from transformers import BartForSequenceClassification, AutoTokenizer, BartForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# 1. 데이터셋 로드 (예시로 tner/bc5cdr 사용)
dataset = load_dataset("tner/bc5cdr")

# 2. BART 모델과 토크나이저 로드
model_name = "facebook/bart-large"
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5개의 라벨 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. label2id와 id2label 정의
label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}

id2label = {v: k for k, v in label2id.items()}  # ID를 텍스트로 변환

# 모델에 label2id, id2label 전달
model = BartForSequenceClassification.from_pretrained(model_name, num_labels=5, label2id=label2id, id2label=id2label)

# 4. 데이터 전처리 함수 정의
def preprocess_function(examples):
    sentences = [' '.join(sublist) for sublist in examples['tokens']]  # 'tokens'를 문장으로 변환
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=1000, is_split_into_words=True)

    # 라벨 변환
    all_labels = []
    for tags in examples['tags']:
        # 각 라벨을 숫자 ID로 변환
        label_ids = [label2id.get(tag, 0) for tag in tags]
        all_labels.append(label_ids)

    inputs["labels"] = all_labels
    return inputs

# 5. 데이터셋 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 6. 데이터셋 분할
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

# 7. DataCollatorForTokenClassification 사용
data_collator = BartForQuestionAnswering(tokenizer)

# 8. compute_metrics 함수 정의
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)  # 예측값 중 가장 높은 값을 가져옵니다.
    labels = labels.flatten()

    # 정확도 계산
    accuracy = accuracy_score(labels, preds.flatten())
    return {"accuracy": accuracy}

# 9. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir='./results',  # 결과 폴더
    evaluation_strategy="epoch",  # 에폭 단위로 평가
    learning_rate=1e-4,  # 학습률
    per_device_train_batch_size=8,  # 배치 크기
    per_device_eval_batch_size=8,  # 배치 크기
    num_train_epochs=10,  # 에폭 수
    weight_decay=1e-5,  # 가중치 감소 (일반적으로 1e-5 ~ 1e-2)
    logging_dir='./logs',  # 로그 디렉토리
    logging_steps=500,  # 로그 출력 주기
)

# 10. Trainer 설정
trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 학습 인자
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=eval_dataset,           # 평가 데이터셋
    data_collator=data_collator,         # 데이터 콜레이터
    compute_metrics=compute_metrics      # compute_metrics 함수 추가
)

# 11. 학습 시작
trainer.train()

# 12. 모델 평가 (validation 데이터셋으로)
results_eval = trainer.evaluate(eval_dataset)
print("Evaluation results on validation dataset:", results_eval)

# 13. 모델 최종 평가 (test 데이터셋으로)
results_test = trainer.evaluate(test_dataset)
print("Final evaluation results on test dataset:", results_test)

# 14. 모델 저장
model.save_pretrained("./bc5cdr_bart_model")
tokenizer.save_pretrained("./bc5cdr_bart_model")
