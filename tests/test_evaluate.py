from training.evaluate import Report, score
from training.generate import Example


class _Stub:
    def __init__(self, mapping):
        self._mapping = mapping

    def classify(self, text):
        return self._mapping.get(text, ("out_of_scope", 0.9))


def test_perfect_classifier_scores_one():
    examples = [Example("a", "greeting", "g:0"), Example("b", "get_time", "t:0")]
    report = score(_Stub({"a": ("greeting", 0.99), "b": ("get_time", 0.99)}), examples)
    assert report.macro_f1 == 1.0


def test_confusion_matrix_records_the_mistake():
    examples = [Example("a", "query_calendar", "q:0")]
    report = score(_Stub({"a": ("add_calendar_event", 0.99)}), examples)
    assert report.confusion[("query_calendar", "add_calendar_event")] == 1.0
    assert report.worst_confusion()[:2] == ("query_calendar", "add_calendar_event")


def test_deferral_rate_counts_low_confidence_in_scope():
    examples = [Example("a", "greeting", "g:0"), Example("b", "greeting", "g:1")]
    report = score(_Stub({"a": ("greeting", 0.99), "b": ("greeting", 0.2)}),
                   examples, defer_threshold=0.6)
    assert report.deferral_rate == 0.5


def test_leak_rate_counts_confident_out_of_scope_mislabels():
    examples = [Example("a", "out_of_scope", "o:0")]
    report = score(_Stub({"a": ("greeting", 0.99)}), examples, defer_threshold=0.6)
    assert report.leak_rate == 1.0
