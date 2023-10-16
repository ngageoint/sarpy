#
# Copyright 2023 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#
from sarpy.io.phase_history.cphd1_elements import Data


def test_data_datatype(kwargs):
    """Test DataType class"""
    data_type = Data.DataType()
    assert data_type.NumCPHDChannels == 0
    assert data_type.NumSupportArrays == 0
    assert data_type.calculate_support_block_size() == 0
    assert data_type.calculate_pvp_block_size() == 0
    assert data_type.calculate_signal_block_size() == 0

    channel0 = Data.ChannelSizeType(
        Identifier="channel0",
        NumVectors=10,
        NumSamples=10,
        SignalArrayByteOffset=0,
        PVPArrayByteOffset=0,
        CompressedSignalSize=123456,
        **kwargs,
    )

    support_arr_size_type0 = Data.SupportArraySizeType(
        Identifier="support0",
        NumRows=10,
        NumCols=10,
        BytesPerElement=8,
        ArrayByteOffset=0,
        **kwargs,
    )

    assert support_arr_size_type0.calculate_size() == (
        support_arr_size_type0.BytesPerElement
        * support_arr_size_type0.NumRows
        * support_arr_size_type0.NumCols
    )

    support_arr_size_type1 = Data.SupportArraySizeType(
        Identifier="support1",
        NumRows=10,
        NumCols=10,
        BytesPerElement=8,
        ArrayByteOffset=0,
        **kwargs,
    )

    data_type = Data.DataType(
        SignalArrayFormat="CF8",
        NumBytesPVP=100,
        SignalCompressionID="NODATA",
        Channels=channel0,
        SupportArrays=[support_arr_size_type0, support_arr_size_type1],
        **kwargs,
    )

    assert data_type.NumCPHDChannels == 1
    assert data_type.NumSupportArrays == 2
    assert (
        data_type.calculate_support_block_size()
        == support_arr_size_type0.calculate_size()
        + support_arr_size_type1.calculate_size()
    )
    assert (
        data_type.calculate_pvp_block_size()
        == channel0.NumVectors * data_type.NumBytesPVP
    )
    assert (
        data_type.calculate_signal_block_size()
        == data_type.Channels[0].CompressedSignalSize
    )

    data_type.SignalCompressionID = None
    assert (
        data_type.calculate_signal_block_size()
        == channel0.NumVectors * channel0.NumSamples * 8
    )

