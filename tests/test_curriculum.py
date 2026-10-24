from senyas.curriculum import check_meaning, load_lessons, normalize_answer


def test_answer_normalization_handles_case_and_punctuation() -> None:
    assert normalize_answer("  You're   Welcome! ") == "youre welcome"


def test_bilingual_answer_is_accepted() -> None:
    lessons = load_lessons()
    thank_you = lessons.loc[lessons["label"].eq("THANK YOU")].iloc[0]
    assert check_meaning("Salamat!", thank_you)
    assert check_meaning("thank you", thank_you)
    assert not check_meaning("good morning", thank_you)

