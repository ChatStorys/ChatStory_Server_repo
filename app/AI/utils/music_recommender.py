"""
음악 추천 시스템 (Hugging Face 모델 버전)

Algorithm 1, 3, 4를 구현하여 감정 기반 음악 추천을 수행합니다.
"""

import numpy as np
from typing import Dict, List
from app.AI.utils.emotion_analyzer import EmotionAnalyzer
import re

class MusicRecommender:
    def __init__(self, db_manager=None, use_hf_model=True, model_name=None):
        """
        음악 추천 시스템 초기화 (Hugging Face 모델 버전)
        
        Args:
            db_manager: DatabaseManager 인스턴스 (선택사항)
            use_hf_model (bool): Hugging Face 모델 사용 여부 (기본값: True)
            model_name (str): Hugging Face 모델 이름 (기본값: "Jinuuuu/KoELECTRA_fine_tunning_emotion")
        """
        if use_hf_model:
            self.emotion_analyzer = EmotionAnalyzer(
                model_name=model_name or "Jinuuuu/KoELECTRA_fine_tunning_emotion",
                use_local=False
            )
        else:
            self.emotion_analyzer = EmotionAnalyzer(use_local=True)
            
        self.db_manager = db_manager
        
        # Algorithm 4: get_music_features_for_emotion의 weight_table (실제 KoELECTRA 레이블 기준)
        self.weight_table = {
            'angry': [0.14, 0.86, 0.95, 0.05, 0.25, 0.75, 0.20],
            'sad': [0.82, 0.18, 0.14, 0.86, 0.05, 0.95, 0.05],
            'anxious': [0.22, 0.78, 0.92, 0.08, 0.17, 0.83, 0.10],
            'heartache': [0.75, 0.25, 0.20, 0.80, 0.13, 0.88, 0.05],
            'embarrassed': [0.33, 0.67, 0.89, 0.11, 0.33, 0.67, 0.15],
            'happy': [0.50, 0.50, 0.09, 0.91, 0.95, 0.05, 0.90]
        }
        
        # 특성 이름들
        self.feature_names = ['acoustic', 'electronic', 'aggressive', 'relaxed', 'happy', 'sad', 'party']

    def _split_text_into_chunks(self, text: str, max_length: int = 300) -> List[str]:
        """
        긴 텍스트를 청크 단위로 분할
        
        Args:
            text (str): 분할할 텍스트
            max_length (int): 각 청크의 최대 문자 수 (토큰 수 고려하여 300자로 설정)
            
        Returns:
            List[str]: 분할된 텍스트 청크들
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return [""]
        
        text = text.strip()
        
        # 텍스트가 짧으면 그대로 반환
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        # 문장 단위로 먼저 분리 시도
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 현재 청크에 문장을 추가했을 때 길이 확인
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                # 현재 청크가 비어있지 않으면 저장
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 문장 자체가 너무 긴 경우 강제로 분할
                if len(sentence) > max_length:
                    # 단어 단위로 분할
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + word) <= max_length:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk.strip():
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence + ". "
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 빈 청크가 없도록 확인
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks if chunks else [text[:max_length]]

    def _analyze_long_text_music_features(self, text: str) -> List[float]:
        """
        긴 텍스트를 청크 단위로 분석하여 평균 음악 특성 벡터 반환
        
        각 청크별로:
        1. 감정 분석 수행
        2. 주요 감정에 해당하는 음악 특성 벡터 생성
        3. 모든 청크의 음악 특성 벡터를 평균내어 최종 특성 벡터 반환
        
        Args:
            text (str): 분석할 긴 텍스트
            
        Returns:
            List[float]: 평균 음악 특성 벡터 [acoustic, electronic, aggressive, relaxed, happy, sad, party]
        """
        try:
            # 텍스트를 청크로 분할
            chunks = self._split_text_into_chunks(text)
            
            if not chunks:
                return self.get_music_features_for_emotion('happy')  # 기본값
            
            print(f"텍스트를 {len(chunks)}개 청크로 분할하여 음악 특성 분석 중...")
            
            # 각 청크별 음악 특성 벡터 수집
            all_music_features = []
            chunk_emotions = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # 1. 청크별 감정 분석
                    emotion_probs = self.emotion_analyzer.analyze_emotion_with_KoELECTRA(chunk)
                    if emotion_probs:
                        # 2. 주요 감정 추출
                        dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
                        chunk_emotions.append(dominant_emotion)
                        
                        # 3. 해당 감정의 음악 특성 벡터 생성
                        music_features = self.get_music_features_for_emotion(dominant_emotion)
                        all_music_features.append(music_features)
                        
                        print(f"청크 {i+1}/{len(chunks)}: {dominant_emotion} (확률: {emotion_probs[dominant_emotion]:.3f}) -> 음악 특성 추출 완료")
                    else:
                        # 감정 분석 실패 시 기본값 사용
                        default_features = self.get_music_features_for_emotion('happy')
                        all_music_features.append(default_features)
                        chunk_emotions.append('happy')
                        print(f"청크 {i+1}/{len(chunks)}: 기본값(happy) 사용")
                        
                except Exception as e:
                    print(f"청크 {i+1} 분석 오류: {str(e)} -> 기본값 사용")
                    # 오류 시 기본값 사용
                    default_features = self.get_music_features_for_emotion('happy')
                    all_music_features.append(default_features)
                    chunk_emotions.append('happy')
                    continue
            
            if not all_music_features:
                return self.get_music_features_for_emotion('happy')  # 기본값
            
            # 4. 모든 청크의 음악 특성 벡터를 평균내어 최종 특성 벡터 생성
            num_features = len(self.feature_names)  # 7개 특성
            averaged_features = []
            
            for feature_idx in range(num_features):
                feature_values = [features[feature_idx] for features in all_music_features]
                averaged_value = sum(feature_values) / len(feature_values)
                averaged_features.append(averaged_value)
            
            print(f"전체 {len(all_music_features)}개 청크의 음악 특성 평균:")
            print(f"청크별 주요 감정: {chunk_emotions}")
            for i, (feature_name, value) in enumerate(zip(self.feature_names, averaged_features)):
                print(f"  {feature_name}: {value:.3f}")
            
            return averaged_features
            
        except Exception as e:
            print(f"긴 텍스트 음악 특성 분석 오류: {str(e)}")
            return self.get_music_features_for_emotion('happy')  # 기본값

    def get_music_features_for_emotion(self, emotion: str) -> List[float]:
        """
        Algorithm 4: 감정에 대한 음악 특성 벡터 반환
        
        Args:
            emotion (str): 감정 ('angry', 'happy', 'anxious', 'embarrassed', 'sad', 'heartache')
            
        Returns:
            List[float]: 음악 특성 벡터 [acoustic, electronic, aggressive, relaxed, happy, sad, party]
        """
        # 1. Define weight_table as in Table 1 (이미 __init__에서 정의됨)
        
        # 2. // 1. Lookup the row for the given emotion
        # 3. music_features ← weight_table[emotion]
        if emotion in self.weight_table:
            music_features = self.weight_table[emotion].copy()
        else:
            # 기본값으로 happy 사용
            music_features = self.weight_table['happy'].copy()
            
        # 4. return music_features
        return music_features

    def cosine_similarity(self, vecA: List[float], vecB: List[float]) -> float:
        """
        Algorithm 3: 두 벡터 간의 코사인 유사도 계산
        
        Args:
            vecA (List[float]): 벡터 A
            vecB (List[float]): 벡터 B
            
        Returns:
            float: 코사인 유사도 (0-1)
        """
        # NumPy 배열로 변환
        vec_a = np.array(vecA)
        vec_b = np.array(vecB)
        
        # 1. dot_prod ← dot(vecA, vecB)
        dot_prod = np.dot(vec_a, vec_b)
        
        # 2. normA ← magnitude(vecA)
        norm_a = np.linalg.norm(vec_a)
        
        # 3. normB ← magnitude(vecB)
        norm_b = np.linalg.norm(vec_b)
        
        # 4. if normA = 0 or normB = 0 then
        # 5.   return 0
        # 6. end if
        if norm_a == 0 or norm_b == 0:
            return 0
            
        # 7. similarity ← dot_prod / (normA * normB)
        similarity = dot_prod / (norm_a * norm_b)
        
        # 8. return similarity
        return float(similarity)

    def recommend_music(self, userID: str,novelContents: str, musicDB: List[Dict] = None, N: int = 5, db_manager=None) -> List[Dict]:
        """Algorithm 1: 소설 내용 기반 음악 추천""" 
        # 1) 전체를 감싸는 try/except
        try:
            # 2) 특징 추출 단계
            try:
                if len(novelContents) > 300:
                    logging.info(f"긴 텍스트 감지: {len(novelContents)}자, 청크별 분석")
                    target_features = self._analyze_long_text_music_features(novelContents)
                else:
                    emotion_probs = self.emotion_analyzer.analyze_emotion_with_KoELECTRA(novelContents)
                    top_emotion = (
                        max(emotion_probs.items(), key=lambda x: x[1])[0]
                        if emotion_probs else "happy"
                    )
                    # logging.info(f"선택된 주요 감정: {top_emotion}")
                    target_features = self.get_music_features_for_emotion(top_emotion)
            except Exception as e:
                # logging.error("특징 추출 실패:\n%s", traceback.format_exc())
                raise RuntimeError(f"특징 추출 오류: {e}")

            # 3) 음악 DB 로드 단계
            try:
                if musicDB is None:
                    manager = self.db_manager or db_manager
                    if manager is None:
                        raise ValueError("musicDB 또는 db_manager 중 하나가 필요합니다")
                    musicDB = manager.get_music_database_for_recommendation()
            except Exception as e:
                # logging.error("음악 DB 로드 실패:\n%s", traceback.format_exc())
                raise RuntimeError(f"음악 DB 로드 오류: {e}")

            # 4) 유사도 계산 및 리스트 생성
            similarity_list = []
            try:
                for entry in musicDB:
                    songName = entry.get("songName", entry.get("musicTitle", "Unknown"))
                    artist = entry.get("artist", entry.get("composer", "Unknown"))
                    song_feats = entry.get("feature_vector", [])
                    if not song_feats or len(song_feats) != 7:
                        song_feats = [0.5] * 7
                    sim = self.cosine_similarity(target_features, song_feats)
                    similarity_list.append({
                        "songName": songName,
                        "artist": artist,
                        "similarity": sim,
                    })
            except Exception as e:
                # logging.error("유사도 계산 실패:\n%s", traceback.format_exc())
                raise RuntimeError(f"유사도 계산 오류: {e}")

            # 5) 정렬 및 결과 반환
            try:
                similarity_list.sort(key=lambda x: x["similarity"], reverse=True)
                return similarity_list[:N]
            except Exception as e:
                # logging.error("정렬/결과 추출 실패:\n%s", traceback.format_exc())
                raise RuntimeError(f"결과 처리 오류: {e}")
            
        except Exception as e:
            # 최종 오류 핸들링: 빈 리스트 반환
            # logging.error("음악 추천 전체 실패: %s", e)
            raise HTTP_500_INTERNAL_SERVER_ERROR(f"결과 처리 오류: {e}")
