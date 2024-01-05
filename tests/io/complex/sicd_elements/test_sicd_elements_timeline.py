#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import re

import numpy as np
import numpy.polynomial.polynomial as npp
import pytest

from sarpy.io.complex.sicd_elements import Timeline


def test_timeline(sicd, kwargs):
    def get_ipp_set(ipp, idx, **kwargs):
        return Timeline.IPPSetType(
            ipp.TStart, ipp.TEnd, ipp.IPPStart, ipp.IPPEnd, ipp.IPPPoly, idx, **kwargs
        )

    ippset1 = get_ipp_set(sicd.Timeline.IPP[0], 1, **kwargs)
    assert ippset1.TStart == sicd.Timeline.IPP[0].TStart
    assert ippset1.TEnd == sicd.Timeline.IPP[0].TEnd
    assert ippset1.IPPStart == sicd.Timeline.IPP[0].IPPStart
    assert ippset1.IPPEnd == sicd.Timeline.IPP[0].IPPEnd
    assert ippset1.IPPPoly == sicd.Timeline.IPP[0].IPPPoly
    assert ippset1._basic_validity_check()

    ippset2 = get_ipp_set(sicd.Timeline.IPP[1], 2, **kwargs)
    ipp_list = [ippset1, ippset2]

    timeline = Timeline.TimelineType(
        sicd.Timeline.CollectStart, sicd.Timeline.CollectDuration, ipp_list
    )
    assert timeline.CollectStart == sicd.Timeline.CollectStart
    assert timeline.CollectDuration == sicd.Timeline.CollectDuration

    for idx in np.arange(len(timeline.IPP)):
        assert timeline.IPP[idx].TStart == ipp_list[idx].TStart
        assert timeline.IPP[idx].TEnd == ipp_list[idx].TEnd
        assert timeline.IPP[idx].IPPStart == ipp_list[idx].IPPStart
        assert timeline.IPP[idx].IPPEnd == ipp_list[idx].IPPEnd
        assert timeline.IPP[idx].IPPPoly == ipp_list[idx].IPPPoly

    assert timeline.CollectEnd == timeline.CollectStart + np.timedelta64(
        int(timeline.CollectDuration * 1e6), "us"
    )
    assert timeline._check_ipp_consecutive()
    assert timeline._check_ipp_times()
    assert timeline._basic_validity_check()

    # Check bail out for IPP consecutive with a single IPP set
    timeline = Timeline.TimelineType(
        sicd.Timeline.CollectStart, sicd.Timeline.CollectDuration, ipp_list[0]
    )
    assert timeline._check_ipp_consecutive()


def test_valid_ippset():
    ippset = Timeline.IPPSetType(
        TStart=0,
        TEnd=1.234,
        IPPStart=8,
        IPPEnd=24,
        IPPPoly=[8, (24 - 8 + 1) / 1.234],
        index=0,
    )
    assert ippset.is_valid()


def test_ippset_unreasonable_prf(caplog):
    ippset = Timeline.IPPSetType(
        TStart=0,
        TEnd=1.234,
        IPPStart=8,
        IPPEnd=24e6,
        IPPPoly=[8, (24e6 - 8 + 1) / 1.234],
        index=0,
    )
    assert ippset.is_valid()  # valid but with a warning
    assert any(
        re.search(r"IPPSet has an unreasonable PRF", x.message) for x in caplog.records
    )


def test_ippset_decreasing_t(caplog):
    ippset = Timeline.IPPSetType(
        TStart=1.234,
        TEnd=0,
        IPPStart=8,
        IPPEnd=24,
        IPPPoly=[24 + 1, -(24 - 8 + 1) / 1.234],
        index=0,
    )
    assert not ippset.is_valid()
    assert any(re.search(r"TStart \(.+\) >= TEnd", x.message) for x in caplog.records)
    assert any(
        re.search(r"IPPSet has a negative PRF", x.message) for x in caplog.records
    )


def test_ippset_decreasing_ipp(caplog):
    ippset = Timeline.IPPSetType(
        TStart=0,
        TEnd=1.234,
        IPPStart=24,
        IPPEnd=8,
        IPPPoly=[24, (8 - 24 + 1) / 1.234],
        index=0,
    )
    assert not ippset.is_valid()
    assert any(
        re.search(r"IPPStart \(.+\) >= IPPEnd", x.message) for x in caplog.records
    )
    assert any(
        re.search(r"IPPSet has a negative PRF", x.message) for x in caplog.records
    )


def test_ippset_start_mismatch(caplog):
    ippset = Timeline.IPPSetType(
        TStart=0,
        TEnd=1.234,
        IPPStart=8,
        IPPEnd=24,
        IPPPoly=[7, (24 - 7 + 1) / 1.234],
        index=0,
    )
    assert not ippset.is_valid()
    assert any(
        re.search(r"IPPStart \(.+\) inconsistent with IPPPoly", x.message)
        for x in caplog.records
    )


def test_ippset_end_mismatch(caplog):
    ippset = Timeline.IPPSetType(
        TStart=0,
        TEnd=1.234,
        IPPStart=8,
        IPPEnd=24,
        IPPPoly=[8, (24 - 8) / 1.234],
        index=0,
    )
    assert not ippset.is_valid()
    assert any(
        re.search(r"IPPEnd \(.+\) inconsistent with IPPPoly", x.message)
        for x in caplog.records
    )


def test_timeline_no_collect_start():
    bad_timeline = Timeline.TimelineType(CollectStart=None, CollectDuration=2.4)
    assert bad_timeline.CollectEnd is None
    assert not bad_timeline.is_valid()


def test_timeline_no_collect_duration():
    bad_timeline = Timeline.TimelineType(
        CollectStart=np.datetime64("2010-03-14T15:00:00.00"), CollectDuration=None
    )
    assert bad_timeline.CollectEnd is None
    assert not bad_timeline.is_valid()


def test_timeline_no_ipp():
    timeline = Timeline.TimelineType(
        CollectStart=np.datetime64("2010-03-14T15:00:00.00"), CollectDuration=2.4
    )
    assert timeline.CollectEnd is not None
    assert timeline.is_valid()


@pytest.fixture
def timeline():
    timeline = Timeline.TimelineType(
        CollectStart=np.datetime64("2010-03-14T15:00:00.00"),
        CollectDuration=2.4,
        IPP=[
            Timeline.IPPSetType(
                TStart=0,
                TEnd=1.2,
                IPPStart=0,
                IPPEnd=1e3,
                IPPPoly=[0, (1e3 + 1) / 1.2],
                index=1,
            ),
            Timeline.IPPSetType(
                TStart=1.2,
                TEnd=2.4,
                IPPStart=1e3 + 1,
                IPPEnd=3e3,
                IPPPoly=npp.polyfit([1.2, 2.4], [1e3 + 1, 3e3 + 1], deg=1),
                index=2,
            ),
        ],
    )
    assert timeline.is_valid(recursive=True)
    return timeline


def test_timeline_swap_indices(timeline, caplog):
    for ipp in timeline.IPP:
        ipp.index = len(timeline.IPP) - ipp.index + 1
    assert not timeline.is_valid()
    assert any(
        re.search(r"The IPPSets are not in start time order", x.message)
        for x in caplog.records
    )
    assert any(
        re.search(r"The IPPSets are not in end time order", x.message)
        for x in caplog.records
    )


def test_timeline_nonconsecutive_index(timeline, caplog):
    for ipp in timeline.IPP:
        ipp.index *= 2
    assert not timeline.is_valid()
    assert any(
        re.search(r"IPPSets indices \(.+\) are not 1..size", x.message)
        for x in caplog.records
    )


@pytest.mark.parametrize("modifier,expected_behavior", ((+1, "overlap"), (-1, "a gap")))
def test_timeline_consecutive_times(timeline, caplog, modifier, expected_behavior):
    timeline.IPP[0].TEnd += modifier
    assert not timeline.is_valid()
    assert any(
        re.search(
            rf"There is {expected_behavior} between IPPSet.+ of .+ seconds", x.message
        )
        for x in caplog.records
    )


@pytest.mark.parametrize("modifier,expected_behavior", ((+1, "overlap"), (-1, "a gap")))
def test_timeline_consecutive_indices(timeline, caplog, modifier, expected_behavior):
    timeline.IPP[0].IPPEnd += modifier
    assert not timeline.is_valid()
    assert any(
        re.search(
            rf"There is {expected_behavior} between IPPSet.+ of .+ IPPs", x.message
        )
        for x in caplog.records
    )


def test_timeline_negative_tstart(timeline, caplog):
    timeline.IPP[0].TStart = -1e-3
    assert not timeline.is_valid()
    assert any(
        re.search(r"Earliest TStart is negative", x.message) for x in caplog.records
    )


def test_timeline_large_tend(timeline, caplog):
    timeline.IPP[-1].TEnd = timeline.CollectDuration * 10
    assert not timeline.is_valid()
    assert any(
        re.search(
            r"time range in IPP entries \(.+\) not in keeping with populated CollectDuration",
            x.message,
        )
        for x in caplog.records
    )
