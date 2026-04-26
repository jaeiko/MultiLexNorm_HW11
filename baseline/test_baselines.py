# test_baselines.py

from LAI_baseline import LAIBaseline
from MFR_baseline import MFRBaseline
from ByT5_baseline import ByT5Baseline

def run_baseline_tests():
    # 1. 공통 테스트용 샘플 데이터 (소셜 미디어 노이즈 문장 가정)
    sample_sentence = ["I", "lov", "u", "bcause", "ur", "cute"]
    print("="*50)
    print(f"원본 문장 (Input): {sample_sentence}")
    print("="*50)

    # 2. LAI 베이스라인 테스트
    print("\n[1. LAI Baseline 테스트 시작]")
    lai_model = LAIBaseline()
    lai_result = lai_model.predict(sample_sentence)
    print(f"-> 결과: {lai_result}")

    # 3. MFR 베이스라인 테스트
    print("\n[2. MFR Baseline 테스트 시작]")
    # 주의: 실제 환경에서는 전체 17개국 훈련 데이터가 들어간다.
    # 테스트 구동을 위해 utils.py가 읽을 수 있는 임시 훈련 데이터 모형을 주입한다.
    mock_train_data = [
        {"raw": ["u", "r", "cute"], "norm": ["you", "are", "cute"]},
        {"raw": ["lov"], "norm": ["love"]},
        {"raw": ["bcause"], "norm": ["because"]},
        {"raw": ["ur"], "norm": ["you", "are"]} 
    ]
    mfr_model = MFRBaseline(mock_train_data)
    mfr_result = mfr_model.predict(sample_sentence)
    print(f"-> 결과: {mfr_result}")

    # 4. ByT5 (ÚFAL) 베이스라인 테스트
    print("\n[3. ByT5 Baseline 테스트 시작]")
    # 처음 실행 시 허깅페이스 서버에서 모델을 다운로드하므로 1~2분 소요될 수 있다.
    byt5_model = ByT5Baseline(model_checkpoint="google/byt5-small")
    byt5_result = byt5_model.predict(sample_sentence)
    print(f"-> 결과: {byt5_result}\n")

if __name__ == "__main__":
    run_baseline_tests()