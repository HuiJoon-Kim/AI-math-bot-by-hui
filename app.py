import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
from google.api_core.exceptions import ResourceExhausted # 🌟 토큰 에러 처리를 위한 모듈 추가

# 1. API 키 설정 
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 🌟 고도화된 전처리 함수 (CLAHE 적용)
# ==========================================
def preprocess_image(img):
    # 1. PIL 이미지를 OpenCV(numpy 배열) 포맷으로 변환
    img_array = np.array(img)
    
    # 2. 흑백(Grayscale) 변환
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 3. 실무 OCR 표준: CLAHE 적용 (강제 흑백 이진화의 부작용 방지)
    # 흐린 글씨는 보존하면서 그림자만 자연스럽게 제거해 줍니다.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    processed = clahe.apply(gray)
    
    # 4. 다시 Streamlit에 띄우기 위해 PIL 이미지로 복구
    final_img = Image.fromarray(processed)
    return final_img

# ==========================================
# 🌟 세션 상태 초기화 (대화 기억하기)
# ==========================================
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None # AI와의 채팅 세션(맥락) 유지
if "messages" not in st.session_state:
    st.session_state.messages = []       # 화면에 띄울 대화 기록 유지

st.title("👨‍🏫 김휘준 대체 AI Bot")
st.write("수학 문제 사진을 올리면, 수준에 맞춰 풀이를 제공합니다.")

# 2. 이미지 업로드 UI
uploaded_file = st.file_uploader("수학 문제 사진을 올려주세요 (jpg, png)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file)
    processed_img = preprocess_image(raw_img)
    
    with st.expander("원본 및 전처리된 이미지 확인하기"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(raw_img, caption="원본", use_container_width=True)
        with col2:
            st.image(processed_img, caption="AI 분석용 (CLAHE 보정)", use_container_width=True)

    choice = st.radio(
        "현재 나의 상태를 선택해 주세요.",
        ("1. 손도 못 대겠다. (가장 첫 핵심 힌트만 제공)",
         "2. 약간의 힌트만 알면 될 것 같다. (5단계 순차 풀이 제공)",
         "3. 다 풀었는데 다른 풀이와 비교해보고 싶다. (모범 답안 제공)")
    )

    # 3. 새로운 문제 풀이 요청 버튼
    if st.button("선생님께 새 문제 질문하기"):
        st.session_state.messages = [] # 새 문제니까 이전 대화 기록 싹 지우기
        
        with st.spinner("이미지를 분석하고 풀이를 고민 중입니다."):
            base_cot = """
            너는 꼼꼼한 수학 과외 선생님이야. 다음의 엄격한 규칙을 지켜.
            1. 제공된 이미지에 '수학 문제'가 있는지 확인해라.
            2. 수학과 무관한 사진이면 "수학 문제가 아닌 것 같아요!"라고 거절해라.
            3. 화질이 너무 나쁘면 "사진이 흐려서 잘 안 보입니다. 다시 찍어주세요!"라고 거절해라.
            4. 정상 문제라면 반드시 단계별로 논리적으로 생각하고 스스로 검증해라(Chain of Thought).
            """

            if choice.startswith('1'):
                system_instruction = base_cot + "시작을 위한 가장 첫 번째 핵심 개념이나 공식 딱 하나만 힌트로 제공해 줘."
            elif choice.startswith('2'):
                system_instruction = base_cot + "전체 풀이 과정을 5단계로 쪼개고, 각 단계 사이에 '[STEP]'을 넣어줘.이때, 문제 해설이 시작되기 전 서론은 [STEP]에 포함하지 마라"
            else:
                system_instruction = base_cot + "가장 깔끔하고 효율적인 모범 답안을 전체적으로 제공해줘."

            model = genai.GenerativeModel(
                model_name="gemini-3.5-flash",
                system_instruction=system_instruction
            )

            # 🌟 단발성 생성이 아닌 '채팅 세션' 시작!
            chat = model.start_chat(history=[])
            st.session_state.chat_session = chat
            
            try:
                # 첫 번째 메시지 (전처리된 이미지 + 초기 질문) 전송
                response = chat.send_message([processed_img, "이 문제를 조건에 맞춰 풀어줘."])
                
                # 첫 번째 답변 기록 저장
                st.session_state.messages.append({"role": "user", "content": "(사진과 함께 질문을 보냈습니다.)"})
                st.session_state.messages.append({"role": "ai", "content": response.text})
            
            except ResourceExhausted:
                st.error("앗! 선생님이 한 번에 너무 많은 질문을 처리하느라 지쳤어요. (API 할당량 초과) 딱 1분만 기다렸다가 다시 질문해 주세요!")
            except Exception as e:
                st.error(f"알 수 없는 오류가 발생했습니다: {e}")

# ==========================================
# 4. 💬 채팅 UI (대화 내역 출력 및 꼬리 질문 입력)
# ==========================================
st.divider()
st.subheader("💬 선생님과 대화하기")

# 저장된 대화 기록을 카카오톡처럼 말풍선으로 순서대로 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): # user는 사람 모양, ai는 로봇 모양 아이콘으로 출력됨
        if "[STEP]" in msg["content"]:
            steps = msg["content"].split('[STEP]')
            for i, step in enumerate(steps):
                if step.strip():
                    st.markdown(f"**[ {i+1} 단계 ]**")
                    st.markdown(step.strip())
                    st.divider()
        else:
            st.markdown(msg["content"])

# 사용자가 꼬리 질문을 입력하는 채팅 입력창
if prompt := st.chat_input("추가로 궁금한 점을 물어보세요."):
    if st.session_state.chat_session is None:
        st.warning("먼저 문제 사진을 올리고 '선생님께 새 문제 질문하기' 버튼을 눌러주세요!")
    else:
        # 내 질문을 화면에 띄우고 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # AI에게 꼬리 질문 전송 및 답변 받기 (이전 맥락을 기억함)
        with st.chat_message("ai"):
            with st.spinner("생각 중입니다."):
                try:
                    # 🌟 [핵심 수술] 토큰 폭발을 막기 위한 '메모리 다이어트' (Sliding Window)
                    current_history = st.session_state.chat_session.history
                    
                    # 대화 턴이 3번(질문-답변 세트가 3개 = history 길이 6)을 넘어가면
                    if len(current_history) > 6:
                        # 최초의 이미지와 풀이(인덱스 0, 1) + 가장 최근의 꼬리 질문과 풀이(마지막 4개)만 남기고 재조립
                        diet_history = current_history[:2] + current_history[-4:]
                        st.session_state.chat_session.history = diet_history
                    
                    # 다이어트된 상태로 새로운 꼬리 질문 전송
                    response = st.session_state.chat_session.send_message(prompt)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "ai", "content": response.text})
                    
                except ResourceExhausted:
                    st.error("앗! 선생님이 한 번에 너무 많은 질문을 처리하느라 지쳤어요. (API 할당량 초과) 딱 1분만 기다렸다가 다시 질문해 주세요!")
                except Exception as e:
                    st.error(f"알 수 없는 오류가 발생했습니다: {e}")
