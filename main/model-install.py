import torch
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments

# 1. 데이터셋 로드 (MedQuad)
dataset = load_dataset("tner/bc5cdr")

# 2. BART 모델과 Tokenizer 로드
model_path = "./biobart"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# 3. label2id 딕셔너리 정의
label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}

id2label = {v: k for k, v in label2id.items()}

# 4. 데이터셋 전처리 함수
def preprocess_function(examples):
    # max_length 설정
    max_length = 2000  # 원하는 길이로 설정 (예: 2000)
    
    # tokens를 토큰화하고 padding을 추가하며, max_length를 2000으로 설정
    inputs = tokenizer(examples['tokens'], truncation=True, padding="max_length", max_length=max_length, is_split_into_words=True)
    
    # 이미 숫자 ID로 제공된 태그를 그대로 사용
    labels = examples['tags']
    
    # tags의 길이가 max_length보다 길면 잘림
    if len(labels) > max_length:
        labels = labels[:max_length]  # 길이가 max_length 이상이면 잘림
    # tags의 길이가 max_length보다 작으면 "O"로 패딩
    labels = labels + [label2id["O"]] * (max_length - len(labels))  # 부족한 부분은 "O"로 채움
    
    # "labels"를 텐서로 변환하여 inputs에 할당
    inputs["labels"] = labels
    
    return inputs

# 5. 데이터셋 분할 및 전처리 적용
train_dataset = dataset['train'].map(preprocess_function, batched=True)
val_dataset = dataset['validation'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 6. Trainer 설정 (평가도 진행)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    evaluation_strategy="epoch",  # 평가 주기를 에포크마다 설정
    save_strategy="epoch",  # 에포크마다 모델 저장
    weight_decay=0.01,
    logging_steps=500,
    load_best_model_at_end=True,  # 가장 좋은 모델을 저장하고 끝에서 로드
    metric_for_best_model="eval_loss",  # 가장 좋은 모델 기준으로 평가
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # validation 데이터셋 추가
)

# 7. 모델 훈련
trainer.train()

# 8. 모델 저장
model.save_pretrained("./biobart-f")
tokenizer.save_pretrained("./biobart-f")

# 9. 평가: test 데이터셋에 대해 예측
predictions, label_ids, metrics = trainer.predict(test_dataset)