#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
import numpy as np

from sarpy.io.complex.sicd_elements import Timeline


def test_timeline(sicd, kwargs):
    def get_ipp_set(ipp, idx, **kwargs):
        return Timeline.IPPSetType(ipp.TStart,
                                   ipp.TEnd,
                                   ipp.IPPStart,
                                   ipp.IPPEnd,
                                   ipp.IPPPoly,
                                   idx,
                                   **kwargs)

    ippset1 = get_ipp_set(sicd.Timeline.IPP[0], 1, **kwargs)
    assert ippset1.TStart == sicd.Timeline.IPP[0].TStart
    assert ippset1.TEnd == sicd.Timeline.IPP[0].TEnd
    assert ippset1.IPPStart == sicd.Timeline.IPP[0].IPPStart
    assert ippset1.IPPEnd == sicd.Timeline.IPP[0].IPPEnd
    assert ippset1.IPPPoly == sicd.Timeline.IPP[0].IPPPoly
    assert ippset1._basic_validity_check()

    ippset2 = get_ipp_set(sicd.Timeline.IPP[1], 2, **kwargs)
    ipp_list = [ippset1, ippset2]

    timeline = Timeline.TimelineType(sicd.Timeline.CollectStart,
                                     sicd.Timeline.CollectDuration,
                                     ipp_list)
    assert timeline.CollectStart == sicd.Timeline.CollectStart
    assert timeline.CollectDuration == sicd.Timeline.CollectDuration

    for idx in np.arange(len(timeline.IPP)):
        assert timeline.IPP[idx].TStart == ipp_list[idx].TStart
        assert timeline.IPP[idx].TEnd == ipp_list[idx].TEnd
        assert timeline.IPP[idx].IPPStart == ipp_list[idx].IPPStart
        assert timeline.IPP[idx].IPPEnd == ipp_list[idx].IPPEnd
        assert timeline.IPP[idx].IPPPoly == ipp_list[idx].IPPPoly

    assert timeline.CollectEnd == timeline.CollectStart + np.timedelta64(int(timeline.CollectDuration*1e6), 'us')
    assert timeline._check_ipp_consecutive()
    assert timeline._check_ipp_times()
    assert timeline._basic_validity_check()

    # Check bail out for IPP consecutive with a single IPP set
    timeline = Timeline.TimelineType(sicd.Timeline.CollectStart,
                                     sicd.Timeline.CollectDuration,
                                     ipp_list[0])
    assert timeline._check_ipp_consecutive()


def test_timeline_start_end_mismatches(sicd, kwargs, caplog):
    # Swap Tstart and TEnd along with IPPStart and IPPEnd
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(ipp0.TEnd,
                                     ipp0.TStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPStart,
                                     ipp0.IPPPoly,
                                     1,
                                     **kwargs)
    bad_ippset._basic_validity_check()
    assert 'TStart ({}) >= TEnd ({})'.format(ipp0.TEnd, ipp0.TStart) in caplog.text
    assert 'IPPStart ({}) >= IPPEnd ({})'.format(ipp0.IPPEnd, ipp0.IPPStart) in caplog.text

    # CollectEnd is None with no CollectStart
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    assert bad_timeline.CollectEnd is None


def test_timeline_negative_tstart(sicd, kwargs, caplog):
    # Negative TStart
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(-ipp0.TStart,
                                     ipp0.TEnd,
                                     ipp0.IPPStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPPoly,
                                     1,
                                     **kwargs)
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    bad_timeline._check_ipp_times()
    assert 'IPP entry 0 has negative TStart' in caplog.text


def test_timeline_bad_tend(sicd, kwargs, caplog):
    # TEnd too large
    ipp0 = sicd.Timeline.IPP[0]
    bad_ippset = Timeline.IPPSetType(ipp0.TStart,
                                     sicd.Timeline.CollectDuration+1,
                                     ipp0.IPPStart,
                                     ipp0.IPPEnd,
                                     ipp0.IPPPoly,
                                     1,
                                     **kwargs)
    bad_timeline = Timeline.TimelineType(None, sicd.Timeline.CollectDuration, bad_ippset)
    bad_timeline._check_ipp_times()
    assert 'appreciably larger than CollectDuration' in caplog.text


def test_timeline_unset_ipp(sicd):
    # Check times is True with no IPP set
    bad_timeline = Timeline.TimelineType(sicd.Timeline.CollectStart, sicd.Timeline.CollectDuration, None)
    assert bad_timeline._check_ipp_times()
