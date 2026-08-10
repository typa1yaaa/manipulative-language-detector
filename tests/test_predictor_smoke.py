# # TOGO: классы gain_loss_exaggeration и topic_shift_misrepresentation показывают низкую precision/recall на текущей версии данных; требуется проверка разметки этих категорий

# import pytest
 
# from src.inference.predictor import ManipulationDetector
# from src.training import config


# @pytest.fixture(scope="session")
# def detector():
#     return ManipulationDetector(config.MODEL_OUTPUT_DIR)


# @pytest.mark.slow
# def test_model_loads_without_error(detector):
#     assert detector.model is not None
#     assert detector.id2label


# @pytest.mark.slow
# def test_detects_manipulation_in_clearly_manipulative_text(detector):
#     text = (
#         "Внимание! Ваш аккаунт будет заблокирован в течение 24 часов, "
#         "если вы срочно не подтвердите данные карты. Служба безопасности банка "
#         "уже зафиксировала подозрительную активность."
#     )
 
#     spans = detector.predict(text)
 
#     assert len(spans) > 0, f"Ожидалось хотя бы одно совпадение, получено: {spans}"


# @pytest.mark.slow
# def test_no_manipulation_detected_in_neutral_text(detector):
#     text = "Собрание кафедры переносится на вторник, аудитория 314, начало в 15:00."
 
#     spans = detector.predict(text)
 
#     assert spans == [], f"Ожидался пустой список, получено: {spans}"


# @pytest.mark.slow
# def test_empty_string_returns_empty_list(detector):
#     assert detector.predict("") == []
#     assert detector.predict("   ") == []

 
# @pytest.mark.slow
# def test_predict_returns_expected_span_structure(detector):
#     text = (
#         "Только сегодня! Успейте купить, пока не закончилось — "
#         "все ваши друзья уже это сделали."
#     )
 
#     spans = detector.predict(text)
 
#     for span in spans:
#         assert set(span.keys()) == {"text", "pattern_name", "confidence"}
#         assert isinstance(span["text"], str) and span["text"]
#         assert span["pattern_name"] in config.LABEL_NAMES
#         assert 0.0 <= span["confidence"] <= 1.0
