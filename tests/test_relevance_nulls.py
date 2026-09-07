"""Regression tests for null-safe graded relevance definitions."""

import numpy as np
import pandas as pd

from src.relevance import compute_relevance_matrix


COLUMNS = ["CI Name (aff)", "CI Subtype (aff)", "Category", "Closure Code"]


def _relevance(
    query: dict[str, object],
    candidate: dict[str, object],
    definition: str = "D",
) -> int:
    """Compute relevance for one query-candidate pair."""
    dataframe = pd.DataFrame([query, candidate], columns=COLUMNS)
    return int(
        compute_relevance_matrix(
            dataframe,
            np.array([1]),
            np.array([0]),
            definition=definition,
        ).toarray()[0, 0]
    )


def test_null_query_ci_does_not_match_null_candidate_ci() -> None:
    """Definition A: null query CI and candidate CI yield zero, not grade 3."""
    query = {
        "CI Name (aff)": np.nan,
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": np.nan,
        "CI Subtype (aff)": "OTHER",
        "Category": "OTHER",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="A") == 0


def test_null_query_subtype_does_not_match_null_candidate_subtype() -> None:
    """Definition A: null query subtype and candidate subtype yield zero, not grade 1."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": np.nan,
        "Category": "CAT_Q",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": np.nan,
        "Category": "CAT_C",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="A") == 0


def test_present_query_ci_absent_from_candidates_has_no_relevance() -> None:
    """Definition A: absent query CI with null subtype yields all zeros."""
    dataframe = pd.DataFrame(
        {
            "CI Name (aff)": ["CI_NEW", "CI_A", np.nan, np.nan],
            "CI Subtype (aff)": [np.nan, np.nan, np.nan, "SUB_X"],
            "Category": ["CAT_Q", "CAT_A", "CAT_B", "CAT_C"],
            "Closure Code": ["CODE_1", "CODE_1", "CODE_1", "CODE_1"],
        }
    )
    relevance = compute_relevance_matrix(
        dataframe,
        np.array([1, 2, 3]),
        np.array([0]),
        definition="A",
    )
    np.testing.assert_array_equal(relevance.toarray()[0], [0, 0, 0])


def test_null_closure_on_either_side_yields_zero() -> None:
    """Definition A: null closure on either side yields zero for every level."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": "SUB_Q",
        "Category": "CAT_Q",
        "Closure Code": np.nan,
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": "SUB_C",
        "Category": "CAT_C",
        "Closure Code": "C",
    }
    reverse = {**candidate, "Closure Code": np.nan}
    assert _relevance(query, candidate, definition="A") == 0
    assert _relevance({**query, "Closure Code": "C"}, reverse, definition="A") == 0


def test_same_ci_and_closure_yields_grade_three() -> None:
    """Definition A: same non-null CI and closure yield exactly grade 3."""
    query = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "OTHER",
        "Category": "OTHER",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="A") == 3


def test_same_ci_different_closure_yields_grade_two() -> None:
    """Definition A: same CI with different closure yields exactly grade 2."""
    query = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C1",
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "OTHER",
        "Category": "OTHER",
        "Closure Code": "C2",
    }
    assert _relevance(query, candidate, definition="A") == 2


def test_same_subtype_and_closure_different_ci_yields_grade_one() -> None:
    """Definition A: same subtype and closure with different CI yield grade 1."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT_Q",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT_C",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="A") == 1


def test_d_same_closure_and_ci_yields_grade_three() -> None:
    """Definition D: same closure and CI yield exactly grade 3."""
    query = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB_Q",
        "Category": "CAT_Q",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB_C",
        "Category": "CAT_C",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="D") == 3


def test_d_same_closure_and_subtype_different_ci_yields_grade_two() -> None:
    """Definition D: same closure and subtype with different CI yield grade 2."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT_Q",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT_C",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="D") == 2


def test_d_same_closure_and_category_different_ci_and_subtype_yields_grade_one() -> None:
    """Definition D: same closure and category with different CI and subtype yield grade 1."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": "SUB_Q",
        "Category": "CAT",
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": "SUB_C",
        "Category": "CAT",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="D") == 1


def test_d_same_ci_different_closure_yields_zero() -> None:
    """Definition D: same CI with different closure yields zero, not grade 2."""
    query = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C1",
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "OTHER",
        "Category": "OTHER",
        "Closure Code": "C2",
    }
    assert _relevance(query, candidate, definition="D") == 0


def test_d_null_category_on_either_side_yields_zero() -> None:
    """Definition D: null category cannot create grade 1 with matching closure."""
    query = {
        "CI Name (aff)": "CI_Q",
        "CI Subtype (aff)": "SUB_Q",
        "Category": np.nan,
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI_C",
        "CI Subtype (aff)": "SUB_C",
        "Category": np.nan,
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="D") == 0


def test_d_null_closure_on_either_side_yields_zero() -> None:
    """Definition D: null closure on either side yields zero at every level."""
    query = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": np.nan,
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C",
    }
    reverse = {**candidate, "Closure Code": np.nan}
    assert _relevance(query, candidate, definition="D") == 0
    assert _relevance({**query, "Closure Code": "C"}, reverse, definition="D") == 0


def test_d_all_null_query_attributes_yield_zero() -> None:
    """Definition D: null query CI, subtype, and category yield zero."""
    query = {
        "CI Name (aff)": np.nan,
        "CI Subtype (aff)": np.nan,
        "Category": np.nan,
        "Closure Code": "C",
    }
    candidate = {
        "CI Name (aff)": "CI",
        "CI Subtype (aff)": "SUB",
        "Category": "CAT",
        "Closure Code": "C",
    }
    assert _relevance(query, candidate, definition="D") == 0


def main() -> None:
    """Run regression tests without a test framework."""
    tests = [
        test_null_query_ci_does_not_match_null_candidate_ci,
        test_null_query_subtype_does_not_match_null_candidate_subtype,
        test_present_query_ci_absent_from_candidates_has_no_relevance,
        test_null_closure_on_either_side_yields_zero,
        test_same_ci_and_closure_yields_grade_three,
        test_same_ci_different_closure_yields_grade_two,
        test_same_subtype_and_closure_different_ci_yields_grade_one,
        test_d_same_closure_and_ci_yields_grade_three,
        test_d_same_closure_and_subtype_different_ci_yields_grade_two,
        test_d_same_closure_and_category_different_ci_and_subtype_yields_grade_one,
        test_d_same_ci_different_closure_yields_zero,
        test_d_null_category_on_either_side_yields_zero,
        test_d_null_closure_on_either_side_yields_zero,
        test_d_all_null_query_attributes_yield_zero,
    ]
    for test in tests:
        test()
    print(f"{len(tests)} relevance definition tests passed")


if __name__ == "__main__":
    main()
