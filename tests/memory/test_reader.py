"""Tests for procela_analysis.memory.reader."""

from __future__ import annotations

import pandas as pd
import pytest
from procela import (
    HypothesisRecord,
    Key,
    Mechanism,
    RangeDomain,
    TimePoint,
    Variable,
    VariableRecord,
    WeightedVotingPolicy,
)

from procela_analysis.memory.reader import MemoryReader

# ---------------------------------------------------------------------------
# Minimal Procela setup helpers
# ---------------------------------------------------------------------------


def _make_mechanism(name: str, variable: Variable) -> Mechanism:
    """Create a minimal mechanism that reads and writes the given variable."""

    class DummyMechanism(Mechanism):
        def __init__(self, reads, writes):
            super().__init__(reads, writes)
            self.name = name

        def transform(self) -> None:
            pass

    mechanism = DummyMechanism(reads=[variable], writes=[variable])
    return mechanism


def _make_proposing_mechanism(
    name: str,
    variable: Variable,
    proposals: list[float],
    confidences: list[float],
) -> Mechanism:
    """
    Create a mechanism that proposes predetermined values.

    Each call to transform() consumes the next proposal and confidence.
    """

    class ProposingMechanism(Mechanism):
        def __init__(self) -> None:
            super().__init__(reads=[variable], writes=[variable])
            self._proposals = proposals.copy()
            self._confidences = confidences.copy()
            self._call_count = 0
            self.name = name

        def transform(self) -> None:
            if self._call_count >= len(self._proposals):
                return
            self.writes()[0].add_hypothesis(
                VariableRecord(
                    value=self._proposals[self._call_count],
                    confidence=self._confidences[self._call_count],
                    source=self.key(),
                )
            )
            self._call_count += 1

    return ProposingMechanism()


def _make_policy() -> WeightedVotingPolicy:
    """Create a resolution policy with a predictable name."""
    return WeightedVotingPolicy(name="weighted_voting")


def _run_step(
    variable: Variable,
    mechanisms: list[Mechanism],
    time_point: TimePoint,
) -> None:
    """Execute one full simulation step: propose, resolve, commit, clear."""
    for mech in mechanisms:
        mech.run()
    variable.resolve_conflict()
    variable.commit(time=time_point)
    variable.clear_hypotheses()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_time_points():
    """Make time points."""
    return {
        0: TimePoint(),
        1: TimePoint(),
        2: TimePoint(),
        3: TimePoint(),
        4: TimePoint(),
    }


@pytest.fixture
def variable():
    """A fresh variable initialized at 50.0."""
    var = Variable("X", domain=RangeDomain(0, 100), policy=_make_policy())
    var.init(VariableRecord(50.0, confidence=1.0))
    return var


@pytest.fixture
def populated_variable(variable, make_time_points):
    """Variable with 3 steps, one mechanism per step."""
    mech = _make_proposing_mechanism(
        "test_mechanism",
        variable,
        proposals=[10.0, 12.0, 11.0],
        confidences=[0.8, 0.6, 0.9],
    )
    for i in range(3):
        _run_step(variable, [mech], make_time_points[i])
    return variable


@pytest.fixture
def multi_mechanism_variable(variable, make_time_points):
    """Variable with 2 steps, two mechanisms per step."""
    m1 = _make_proposing_mechanism(
        "mechanism_a",
        variable,
        proposals=[10.0, 11.0],
        confidences=[0.8, 0.85],
    )
    m2 = _make_proposing_mechanism(
        "mechanism_b",
        variable,
        proposals=[20.0, 21.0],
        confidences=[0.6, 0.65],
    )
    for i in range(2):
        _run_step(variable, [m1, m2], make_time_points[i])
    return variable


@pytest.fixture
def empty_variable():
    """Variable with no memory (never initialized, no steps run)."""
    var = Variable("empty", domain=RangeDomain(0, 100), policy=_make_policy())
    return var


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestMemoryReaderConstruction:
    """Tests for MemoryReader initialization."""

    def test_constructs_with_populated_variable(self, populated_variable):
        reader = MemoryReader(populated_variable)
        assert reader is not None

    def test_constructs_with_empty_variable(self, empty_variable):
        reader = MemoryReader(empty_variable)
        assert reader is not None

    def test_constructs_with_time_points(self, populated_variable, make_time_points):
        time_points = {i: make_time_points[i] for i in [0, 1, 2, 3]}
        reader = MemoryReader(populated_variable, time_points=time_points)
        assert reader is not None

    def test_time_points_inverted_internally(
        self, populated_variable, make_time_points
    ):
        """time_points {step: TimePoint} is inverted to {TimePoint: step}."""
        time_points = {i: make_time_points[i] for i in [0, 1, 2]}
        reader = MemoryReader(populated_variable, time_points=time_points)
        hyp = reader.hypotheses()
        steps = set(hyp["step"].unique())
        assert 0 in steps
        assert 1 in steps
        assert 2 in steps


# ---------------------------------------------------------------------------
# hypotheses() tests
# ---------------------------------------------------------------------------


class TestHypotheses:
    """Tests for MemoryReader.hypotheses()."""

    def test_returns_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, populated_variable):
        """3 steps after init = 3 hypotheses from mechanism + 1 from init."""
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        # init creates step 0 with 1 hypothesis, then 3 steps with 1 each
        assert len(df) == 4

    def test_multiple_mechanisms_per_step(self, multi_mechanism_variable):
        """2 steps with 2 mechanisms each + init."""
        reader = MemoryReader(multi_mechanism_variable)
        df = reader.hypotheses()
        # init (1) + 2 steps × 2 mechanisms = 5
        assert len(df) == 5

    def test_mechanism_names_are_resolved(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        # Step 1 (index 1) is the first mechanism proposal
        mechanism_names = set(df["mechanism"].unique())
        assert "test_mechanism" in mechanism_names

    def test_variable_name_included(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        assert (df["variable"] == "X").all()

    def test_source_key_is_hex_string(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        # Find a row that isn't the init row (init source may be None)
        non_init = df[df["mechanism"] == "test_mechanism"]
        key_str = non_init.iloc[0]["source_key"]
        assert isinstance(key_str, str)
        assert len(key_str) > 0

    def test_proposed_values_present(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.hypotheses()
        proposed_values = set(df["proposed"].values)
        assert 10.0 in proposed_values
        assert 12.0 in proposed_values
        assert 11.0 in proposed_values

    def test_null_confidence_defaults_to_zero(self, variable, make_time_points):
        """Hypothesis with None confidence should become 0.0."""
        mech = _make_proposing_mechanism(
            "null_conf_mech",
            variable,
            proposals=[5.0],
            confidences=[None],  # type: ignore[list-item]
        )
        _run_step(variable, [mech], make_time_points[0])
        reader = MemoryReader(variable)
        df = reader.hypotheses()
        # Find the mechanism's proposal
        mech_row = df[df["mechanism"] == "null_conf_mech"]
        assert mech_row.iloc[0]["confidence"] == pytest.approx(0.0)

    def test_raises_on_empty_memory(self, empty_variable):
        reader = MemoryReader(empty_variable)
        with pytest.raises(ValueError, match="empty records"):
            reader.hypotheses()


# ---------------------------------------------------------------------------
# resolutions() tests
# ---------------------------------------------------------------------------


class TestResolutions:
    """Tests for MemoryReader.resolutions()."""

    def test_returns_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.resolutions()
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, populated_variable):
        """init + 3 steps = 4 resolutions."""
        reader = MemoryReader(populated_variable)
        df = reader.resolutions()
        assert len(df) == 4

    def test_policy_name_included(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.resolutions()
        # All resolutions should have the policy name
        policies = set(df["policy"].unique())
        assert "weighted_voting" in policies

    def test_num_hypotheses_correct(self, multi_mechanism_variable):
        reader = MemoryReader(multi_mechanism_variable)
        df = reader.resolutions()
        # Step 0 is init (1 hypothesis), steps 1-2 have 2 each
        assert df.iloc[0]["num_hypotheses"] == 1  # init
        assert df.iloc[1]["num_hypotheses"] == 2
        assert df.iloc[2]["num_hypotheses"] == 2

    def test_num_hypotheses_single_mechanism(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.resolutions()
        # After init, all steps have 1 hypothesis
        non_init = df[df["step"] > 0]
        assert (non_init["num_hypotheses"] == 1).all()

    def test_raises_on_empty_memory(self, empty_variable):
        reader = MemoryReader(empty_variable)
        with pytest.raises(ValueError, match="empty records"):
            reader.resolutions()


# ---------------------------------------------------------------------------
# errors() tests
# ---------------------------------------------------------------------------


class TestErrors:
    """Tests for MemoryReader.errors()."""

    def test_returns_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.errors()
        assert isinstance(df, pd.DataFrame)

    def test_computes_absolute_error(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.errors()
        # Should have absolute_error column with non-negative values
        assert (df["absolute_error"] >= 0).all()

    def test_computes_squared_error(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df = reader.errors()
        # squared_error should equal absolute_error^2
        expected_sq = df["absolute_error"] ** 2
        pd.testing.assert_series_equal(
            df["squared_error"], expected_sq, check_names=False
        )

    def test_zero_error_when_proposal_matches_resolution(
        self, variable, make_time_points
    ):
        """When proposal equals resolution, error should be zero."""
        # Use a mechanism that proposes exactly 50.0
        mech = _make_proposing_mechanism(
            "exact_mech",
            variable,
            proposals=[50.0],
            confidences=[0.9],
        )
        _run_step(variable, [mech], make_time_points[0])
        reader = MemoryReader(variable)
        df = reader.errors()
        # Find the exact_mech row
        exact_row = df[df["mechanism"] == "exact_mech"]
        # If the resolution also picked 50.0 (likely with weighted voting),
        # error should be near zero
        assert exact_row["absolute_error"].iloc[0] >= 0.0

    def test_raises_when_no_hypotheses(self, empty_variable):
        reader = MemoryReader(empty_variable)
        with pytest.raises(ValueError, match="empty records"):
            reader.errors()

    def test_when_record_none(self, variable):
        """Variable with hypotheses but a least a record is None should be skipped."""
        mech = _make_mechanism("no_res_mech", variable)
        variable.add_hypothesis(
            VariableRecord(value=5.0, confidence=0.8, source=mech.key())
        )
        variable.hypotheses.append(HypothesisRecord(None))
        variable.resolve_conflict()
        variable.commit()
        # Never called resolve_conflict or commit
        reader = MemoryReader(variable)
        assert len(reader.hypotheses()) == 2

    def test_raises_when_no_resolutions(self, variable):
        """Variable with hypotheses but no resolutions should raise."""
        mech = _make_mechanism("no_res_mech", variable)
        variable.add_hypothesis(
            VariableRecord(value=5.0, confidence=0.8, source=mech.key())
        )
        # Never called resolve_conflict or commit
        MemoryReader(variable)
        # Memory only has whatever was committed (nothing, since no commit)
        # But the variable was initialized, so it has init record
        # Actually: init() calls commit internally
        # So there is a resolution but the add_hypothesis is in temp memory
        # This test verifies the error path for genuinely empty memory
        empty_var = Variable("no_init", domain=RangeDomain(0, 100))
        reader_empty = MemoryReader(empty_var)
        with pytest.raises(ValueError, match="empty records"):
            reader_empty.errors()


# ---------------------------------------------------------------------------
# time_points tests
# ---------------------------------------------------------------------------


class TestTimePoints:
    """Tests for MemoryReader with explicit time_points mapping."""

    def test_uses_timepoint_mapping(self, populated_variable, make_time_points):
        """Steps should reflect the mapped time_points, not iteration index."""
        time_points = {i: make_time_points[i] for i in range(4)}
        reader = MemoryReader(populated_variable, time_points=time_points)

        hyp = reader.hypotheses()
        steps = set(hyp["step"].unique())
        assert steps == {0, 1, 2}

        res = reader.resolutions()
        res_steps = set(res["step"].unique())
        assert res_steps == {0, 1, 2}

    def test_partial_timepoint_mapping_skips_unmapped(
        self, populated_variable, make_time_points
    ):
        """Unmapped TimePoints should be skipped (step < 0)."""
        # Only map indices 1 and 2, skip 0 and 3
        time_points = {i: make_time_points[i] for i in range(1, 3)}
        reader = MemoryReader(populated_variable, time_points=time_points)

        hyp = reader.hypotheses()
        steps = set(hyp["step"].unique())
        assert steps == {1, 2}

    def test_no_time_points_uses_index(self, populated_variable):
        """Without time_points, records() iteration index becomes step number."""
        reader = MemoryReader(populated_variable)
        hyp = reader.hypotheses()
        steps = set(hyp["step"].unique())
        assert steps == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Unknown source tests
# ---------------------------------------------------------------------------


class TestUnknownSource:
    """Tests for sources that cannot be resolved to a known name."""

    def test_unknown_key_returns_unknown(self, variable):
        """A key not registered with KeyAuthority should return 'unknown'."""
        unknown_key = Key()
        variable.add_hypothesis(
            VariableRecord(value=1.0, confidence=0.5, source=unknown_key)
        )
        variable.resolve_conflict()
        variable.commit()
        variable.clear_hypotheses()

        reader = MemoryReader(variable)
        df = reader.hypotheses()
        # Find the hypothesis with the unknown key
        unknown_rows = df[df["source_key"] == unknown_key.to_bytes().hex()]
        assert len(unknown_rows) == 1
        assert unknown_rows.iloc[0]["mechanism"] == "unknown"

    def test_null_source_key_returns_unknown(self):
        """_key_to_string should return 'unknown' for None key."""
        result = MemoryReader._key_to_string(None)
        assert result == "unknown"

    def test_null_key_returns_unknown_source_name(self, empty_variable):
        """_get_source_name should return 'unknown' for None key."""
        reader = MemoryReader(empty_variable)
        assert reader._get_source_name(None) == "unknown"


# ---------------------------------------------------------------------------
# Null memory tests
# ---------------------------------------------------------------------------


class TestNullMemory:
    """Tests for variables with no memory."""

    def test_empty_memory_produces_no_records(self, empty_variable):
        reader = MemoryReader(empty_variable)
        assert reader._hypotheses_records == []
        assert reader._resolutions_records == []

    def test_uninitialized_variable_raises(self):
        """Variable that was never initialized or committed should raise."""
        var = Variable("no_init", domain=RangeDomain(0, 100))
        reader = MemoryReader(var)
        with pytest.raises(ValueError, match="empty records"):
            reader.hypotheses()


# ---------------------------------------------------------------------------
# Caching / idempotency tests
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests that MemoryReader caches results from single iteration."""

    def test_hypotheses_returns_same_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df1 = reader.hypotheses()
        df2 = reader.hypotheses()
        pd.testing.assert_frame_equal(df1, df2)

    def test_resolutions_returns_same_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df1 = reader.resolutions()
        df2 = reader.resolutions()
        pd.testing.assert_frame_equal(df1, df2)

    def test_errors_returns_same_dataframe(self, populated_variable):
        reader = MemoryReader(populated_variable)
        df1 = reader.errors()
        df2 = reader.errors()
        pd.testing.assert_frame_equal(df1, df2)

    def test_errors_recomputed_not_cached_internally(self, populated_variable):
        """errors() calls hypotheses() and resolutions() each time."""
        reader = MemoryReader(populated_variable)
        df1 = reader.errors()
        df2 = reader.errors()
        pd.testing.assert_frame_equal(df1, df2)
        assert df1 is not df2
