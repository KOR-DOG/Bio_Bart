from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# RAG 모델과 토크나이저 로드
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

def generate_response(input_text):
    """
    사용자가 입력한 질문에 대해 RAG 모델을 사용하여 응답을 생성합니다.
    """
    # 입력을 토큰화
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 모델을 통해 응답 생성
    generated = model.generate(input_ids=inputs['input_ids'], 
                               num_beams=5,        # 생성할 답변의 품질을 높이기 위한 설정 (다양성)
                               max_length=150,     # 최대 길이 설정
                               early_stopping=True) # 생성이 끝나면 중지

    # 생성된 응답을 디코딩하여 텍스트로 변환
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return response

def chatbot():
    """
    사용자와의 대화 흐름을 처리하는 함수입니다.
    """
    print("챗봇에 오신 것을 환영합니다! 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    
    while True:
        user_input = input("You: ")  # 사용자 입력 받기
        
        # 종료 조건 (exit 또는 quit 입력 시 종료)
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # 챗봇 응답 생성
        response = generate_response(user_input)
        
        # 챗봇의 응답 출력
        print("Bot:", response)

# 챗봇 시작
chatbot()