#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.phase_history.cphd1_elements import TxRcv


def test_txrcv_txrcvtype():
    """Smoke test"""
    tx_rcv_type = TxRcv.TxRcvType()
    assert tx_rcv_type.NumTxWFs == 0
    assert tx_rcv_type.NumRcvs == 0

    expected_num_tx_wf_params = 2
    tx_rcv_type.TxWFParameters = [
        TxRcv.TxWFParametersType(Identifier=str(x),
                                 PulseLength=x*1e-6,
                                 RFBandwidth=x*1e8,
                                 FreqCenter=1e10,
                                 LFMRate=0.0,
                                 Polarization="S",
                                 Power=1.23456789)
        for x in range(expected_num_tx_wf_params)
    ]
    assert tx_rcv_type.NumTxWFs == expected_num_tx_wf_params

    expected_num_rcv_params = 8
    tx_rcv_type.RcvParameters = [
        TxRcv.RcvParametersType(Identifier=str(x),
                                WindowLength=x*1e-6,
                                SampleRate=x*1e8,
                                IFFilterBW=x*1e8,
                                FreqCenter=10036761634.5518,
                                LFMRate=0.0,
                                Polarization="S",
                                PathGain=5.6789)
        for x in range(expected_num_rcv_params)
    ]
    assert tx_rcv_type.NumRcvs == expected_num_rcv_params

    tx_rcv_type.RcvParameters.clear()
    assert tx_rcv_type.NumRcvs == 0

