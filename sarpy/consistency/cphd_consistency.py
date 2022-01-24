#
# Copyright 2020-2021 Valkyrie Systems Corporation
#
# Licensed under MIT License.  See LICENSE.
#

__classification__ = "UNCLASSIFIED"
__author__ = "Nathan Bombaci, Valkyrie"


import logging
import argparse
import collections
import functools
import itertools
import numbers
import os
import re
from typing import List

import numpy as np
import numpy.polynomial.polynomial as npp

from sarpy.geometry import geocoords

import sarpy.consistency.consistency as con
import sarpy.consistency.parsers as parsers
import sarpy.io.phase_history.cphd1_elements.CPHD
import sarpy.io.phase_history.cphd1_elements.utils as cphd1_utils
from sarpy.io.phase_history.cphd_schema import get_schema_path

logger = logging.getLogger(__name__)

try:
    import pytest
except ImportError:
    pytest = None
    logger.critical(
        'Functionality for CPHD consistency testing cannot proceed WITHOUT the pytest '
        'package')

try:
    from lxml import etree
except ImportError:
    etree = None
    pytest = None
    logger.critical(
        'Functionality for CPHD consistency testing cannot proceed WITHOUT the lxml '
        'package')

try:
    import shapely.geometry as shg
    have_shapely = True
except ImportError:
    have_shapely = False

try:
    import networkx as nx
    have_networkx = True
except ImportError:
    have_networkx = False


INVALID_CHAR_REGEX = re.compile(r'\W')
DEFAULT_SCHEMA = get_schema_path(version='1.0.1')


def strip_namespace(root):
    """
    Returns a copy of the input etree with namespaces removed.

    Parameters
    ----------
    root : etree.ElementTree
        The element tree

    Returns
    -------
    etree.ElementTree
        The element tree
    """

    # strip namespace from each element
    for elem in root.iter():
        try:
            elem.tag = elem.tag.split('}')[-1]
        except (AttributeError, TypeError):
            pass
    # remove default namespace
    nsmap = root.nsmap
    nsmap.pop(None, None)
    new_root = etree.Element(root.tag, nsmap)
    new_root[:] = root[:]

    return new_root


def parse_pvp_elem(elem):
    """
    Reverse of `pvp_elem`.

    Parameters
    ----------
    elem : etree.ElementTree.Element
        Node for the specified PVP parameter.

    Returns
    -------
    Tuple
        Tuple (parameter_name, {``'offset'``:offset, ``'size'``:size, ``'dtype'``:dtype}). PVP element information.
    """

    if elem.tag == "AddedPVP":
        name = elem.find('Name').text
    else:
        name = elem.tag

    offset = int(elem.find('Offset').text)
    size = int(elem.find('Size').text)

    dtype = cphd1_utils.binary_format_string_to_dtype(elem.find('Format').text)

    return name, {"offset": offset,
                  "size": size,
                  "dtype": dtype}


def read_header(file_handle):
    """Reads a CPHD header from a file.

    Parameters
    ----------
    file_handle
    Readable File object, i.e., ``file_handle = open(filename, 'rb')``.

        Handle of the CPHD file that is to be read

    Returns
    -------
    Dict
        Dictionary containing CPHD header values.
    """

    file_handle.seek(0, 0)
    version = file_handle.readline().decode()
    assert version.startswith('CPHD/1.0')

    header = sarpy.io.phase_history.cphd1_elements.CPHD.CPHDHeader.from_file_object(file_handle)
    return {k:getattr(header, k) for k in header._fields if getattr(header, k) is not None}


def per_channel(method):
    """
    Decorator to mark check methods as being applicable to each CPHD channel

    Parameters
    ----------
    method : Callable
        Method to mark

    Returns
    -------
    Callable
        Marked input `method`
    """

    method.per_channel = True
    return method


def get_by_id(xml, path, identifier):
    """
    Find a node with a specific child Identifier node.

    Parameters
    ----------
    xml : etree.Element
        Root node of XPath expression
    path : str
        XPath expression relative to `xml`
    identifier : str
        Value of child Identifier node

    Returns
    -------
    etree.Element
        node found by path with an Identifier node with value of `identifier`
    """

    return xml.xpath('{path}/Identifier[text()="{identifier}"]/..'.format(path=path, identifier=identifier))[0]


class CphdConsistency(con.ConsistencyChecker):
    """
    Check CPHD file structure and metadata for internal consistency

    Parameters
    ----------
    cphdroot : etree.Element
        root CPHD XML node
    pvps : None|Dict[str, np.ndarray]
        numpy structured array of PVPs
    header : Dict
        CPHD header key value pairs
    filename : None|str
        Path to CPHD file (or None if not available)
    schema : str
        Path to CPHD XML Schema
    check_signal_data: bool
        Should the signal array be checked for invalid values
    """

    def __init__(self, cphdroot, pvps, header, filename, schema, check_signal_data):
        super(CphdConsistency, self).__init__()
        self.xml = strip_namespace(etree.fromstring(etree.tostring(cphdroot)))
        self.xml_with_ns = cphdroot
        self.pvps = pvps
        self.filename = filename
        self.header = header
        self.schema = schema
        self.check_signal_data = check_signal_data
        channel_ids = [x.text for x in self.xml.findall('./Data/Channel/Identifier')]

        # process decorated methods to generate per-channel tests
        # reverse the enumerated list so that we don't disturb indices on later iterations as we insert into the list
        for index, func in reversed(list(enumerate(self.funcs))):
            if getattr(func, 'per_channel', False):
                subfuncs = []
                for channel_id in channel_ids:
                    channel_node = self.xml.xpath('./Channel/Parameters/Identifier[text()="{}"]/..'.format(channel_id))[0]
                    subfunc = functools.partial(func, channel_id, channel_node)
                    subfunc.__doc__ = "{doc} for channel {chanid}".format(doc=func.__doc__, chanid=channel_id)
                    modified_channel_id = re.sub(INVALID_CHAR_REGEX, '_', channel_id)
                    subfunc.__name__ = "{name}_{chanid}".format(name=func.__name__, chanid=modified_channel_id)
                    subfuncs.append(subfunc)
                self.funcs[index:index+1] = subfuncs

    @classmethod
    def from_file(cls, filename, schema, check_signal_data):
        """
        Create a CphdConsistency object from a CPHD file.

        Parameters
        ----------
        filename : str
            Path to CPHD file
        schema : str
            Path to CPHD XML schema
        check_signal_data : bool
            Should the signal array be checked for invalid values

        Returns
        -------
        CphdConsistency
            new object
        """
        with open(filename, 'rb') as infile:
            try:
                header = None
                cphdroot = etree.parse(infile)
                pvp_block = None
            except etree.XMLSyntaxError:
                header = read_header(infile)
                infile.seek(header['XML_BLOCK_BYTE_OFFSET'], 0)
                xml_block = infile.read(header['XML_BLOCK_SIZE'])
                cphdroot = etree.fromstring(xml_block)
                infile.seek(header['PVP_BLOCK_BYTE_OFFSET'], 0)
                pvp_block = infile.read(header['PVP_BLOCK_SIZE'])

        cphdroot_no_ns = strip_namespace(etree.fromstring(etree.tostring(cphdroot)))
        fields = [parse_pvp_elem(field) for field in list(cphdroot_no_ns.find('./PVP'))]
        dtype = np.dtype({'names': [name for name, _ in fields],
                          'formats': [info['dtype'] for _, info in fields],
                          'offsets': [info['offset']*8 for _, info in fields]}).newbyteorder('B')

        if pvp_block is None:
            pvps = None
        else:
            pvps = {}
            for channel_node in cphdroot_no_ns.findall('./Data/Channel'):
                channel_id = channel_node.findtext('./Identifier')
                channel_pvps = np.frombuffer(pvp_block, dtype=dtype,
                                             count=int(channel_node.findtext('./NumVectors')),
                                             offset=int(channel_node.findtext('./PVPArrayByteOffset')))
                pvps[channel_id] = channel_pvps
        return cls(cphdroot, pvps, header, filename, schema=schema, check_signal_data=check_signal_data)

    def check_header_keys(self):
        """
        Asserts that the required keys are in the header.
        """

        with self.precondition():
            assert self.header is not None
            required_fields = set(['XML_BLOCK_SIZE', 'XML_BLOCK_BYTE_OFFSET',
                                   'PVP_BLOCK_SIZE', 'PVP_BLOCK_BYTE_OFFSET',
                                   'SIGNAL_BLOCK_SIZE', 'SIGNAL_BLOCK_BYTE_OFFSET',
                                   'CLASSIFICATION', 'RELEASE_INFO'])
            for name in required_fields:
                with self.need('Required header field: {} is in header'.format(name)):
                    assert name in self.header
            with self.precondition():
                assert 'SUPPORT_BLOCK_SIZE' in self.header
                with self.need("SUPPORT_BLOCK fields go together"):
                    assert 'SUPPORT_BLOCK_BYTE_OFFSET' in self.header
            with self.precondition():
                assert 'SUPPORT_BLOCK_BYTE_OFFSET' in self.header
                with self.need("SUPPORT_BLOCK fields go together"):
                    assert 'SUPPORT_BLOCK_SIZE' in self.header

    def check_classification_and_release_info(self):
        """
        Asserts that the Classification and ReleaseInfo fields are the same in header and the xml.
        """
        with self.precondition():
            assert self.header is not None
            with self.need("Header CLASSIFICATION matches XML Classification"):
                assert self.header['CLASSIFICATION'] == self.xml.findtext('./CollectionID/Classification') != None
            with self.need("Header RELEASE_INFO matches XML ReleaseInfo"):
                assert self.header['RELEASE_INFO'] == self.xml.findtext('./CollectionID/ReleaseInfo') != None

    def check_against_schema(self):
        """
        The XML matches the schema.
        """

        with self.precondition():
            assert self.schema is not None
            with open(self.schema, mode='rb') as schema_file:
                schema = etree.XMLSchema(etree.parse(schema_file))

            with self.need("XML passes schema"):
                assert schema.validate(self.xml_with_ns), schema.error_log

    @per_channel
    def check_channel_dwell_exist(self, channel_id, channel_node):
        """
        The referenced Dwell and COD nodes exist.
        """

        cod_id = channel_node.findtext('./DwellTimes/CODId')
        with self.need("COD node exists"):
            assert self.xml.xpath('./Dwell/CODTime/Identifier[text()="{}"]/..'.format(cod_id))[0] is not None

        dwell_id = channel_node.findtext('./DwellTimes/DwellId')
        with self.need("DwellTime node exists"):
            assert self.xml.xpath('./Dwell/DwellTime/Identifier[text()="{}"]/..'.format(dwell_id))[0] is not None

    def check_antenna(self):
        """
        Check that antenna node is consistent.
        """
        with self.precondition():
            antenna_node = self.xml.find("./Antenna")
            assert antenna_node is not None

            expected_num_acfs = int(antenna_node.findtext("./NumACFs"))
            actual_num_acfs = len(antenna_node.findall("./AntCoordFrame"))
            with self.need("The NumACFs must be equal to the number of ACF nodes."):
                assert expected_num_acfs == actual_num_acfs

            expected_num_apcs = int(antenna_node.findtext("./NumAPCs"))
            actual_num_apcs = len(antenna_node.findall("./AntPhaseCenter"))
            with self.need("The NumAPCs must be equal to the number of APC nodes."):
                assert expected_num_apcs == actual_num_apcs

            expected_num_antpats = int(antenna_node.findtext("./NumAntPats"))
            actual_num_antpats = len(antenna_node.findall("AntPattern"))
            with self.need("The NumAntPats must be equal to the number of AntPattern nodes."):
                assert expected_num_antpats == actual_num_antpats

            apc_acfids = antenna_node.findall("./AntPhaseCenter/ACFId")
            apc_acf_ids_text = {apc_acfid.text for apc_acfid in apc_acfids}
            acf_identifiers = antenna_node.findall("./AntCoordFrame/Identifier")
            acf_identifiers_text = {acf_identifier.text for acf_identifier in acf_identifiers}
            with self.need("./AntPhaseCenter/ACFId references an identifier in AntCoordFrame."):
                assert apc_acf_ids_text <= acf_identifiers_text

    @per_channel
    def check_channel_antenna_exist(self, channel_id, channel_node):
        """
        The antenna patterns and phase centers exist if declared.
        """

        with self.precondition():
            assert channel_node.find('./Antenna') is not None
            for side in 'Tx', 'Rcv':
                apc_id = channel_node.findtext('./Antenna/{}APCId'.format(side))
                with self.need("AntPhaseCenter node exists with name {} (for {})".format(apc_id, side)):
                    assert get_by_id(self.xml, './Antenna/AntPhaseCenter/', apc_id) is not None
                apat_id = channel_node.findtext('./Antenna/{}APATId'.format(side))
                with self.need("AntPattern node exists with name {} (for {})".format(apat_id, side)):
                    assert get_by_id(self.xml, './Antenna/AntPattern/', apat_id) is not None

    @per_channel
    def check_channel_txrcv_exist(self, channel_id, channel_node):
        """
        The declared TxRcv nodes exist.
        """

        with self.precondition():
            assert channel_node.find('./TxRcv') is not None
            for tx_wf_id in channel_node.findall('./TxRcv/TxWFId'):
                with self.need("TxWFParameters node exists with id {}".format(tx_wf_id.text)):
                    assert get_by_id(self.xml, './TxRcv/TxWFParameters', tx_wf_id.text) is not None
            for rcv_id in channel_node.findall('./TxRcv/RcvId'):
                with self.need("RcvParameters node exists with id {}".format(rcv_id.text)):
                    assert get_by_id(self.xml, './TxRcv/RcvParameters', rcv_id.text) is not None

    @per_channel
    def check_time_monotonic(self, channel_id, channel_node):
        """
        PVP times increase monotonically.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            for side in 'Tx', 'Rcv':
                spvp = pvp['{}Time'.format(side)]
                mask = np.isfinite(spvp)
                with self.need("{}Time is monotonic (diff > 0)".format(side)):
                    assert np.all(np.greater(np.diff(spvp[mask]), 0))

    @per_channel
    def check_rcv_after_tx(self, channel_id, channel_node):
        """
        RcvTime is after TxTime.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            tx_time = pvp['TxTime']
            rcv_time = pvp['RcvTime']
            mask = np.logical_and(np.isfinite(pvp['TxTime']), np.isfinite(pvp['RcvTime']))
            with self.need("Rcv after Tx"):
                assert np.all(np.greater(rcv_time[mask], tx_time[mask]))

    @per_channel
    def check_rcv_finite(self, channel_id, channel_node):
        """
        RcvTime and Pos are finite.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            rcv_time = pvp['RcvTime']
            rcv_pos = pvp['RcvPos']
            with self.need("RcvTime"):
                assert np.all(np.isfinite(rcv_time))
            with self.need("RcvPos"):
                assert np.all(np.isfinite(rcv_pos))

    @per_channel
    def check_channel_fxfixed(self, channel_id, channel_node):
        """
        PVP agrees with FXFixed.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            fx1_tol = con.Approx(np.nanmean(pvp['FX1']))
            fx2_tol = con.Approx(np.nanmean(pvp['FX2']))
            fx1_min_max = np.array([pvp['FX1'].min(), pvp['FX1'].max()])
            fx2_min_max = np.array([pvp['FX2'].min(), pvp['FX2'].max()])
            with self.precondition():
                assert parsers.parse_bool(channel_node.find('./FXFixed'))
                with self.need("FX1 does not change"):
                    assert fx1_min_max == fx1_tol
                with self.need("FX2 does not change"):
                    assert fx2_min_max == fx2_tol

            with self.precondition():
                assert not parsers.parse_bool(channel_node.find('./FXFixed'))
                with self.need("FX1 and/or FX2 are not exactly constant"):
                    assert not ((pvp['FX1'].min() == pvp['FX1'].max()) and (pvp['FX2'].min() == pvp['FX2'].max()))
                    with self.want("FX1 and/or FX2 is not almost constant"):
                        assert not ((fx1_min_max == fx1_tol) and (fx2_min_max == fx2_tol))

    @per_channel
    def check_channel_toafixed(self, channel_id, channel_node):
        """
        PVP agrees with TOAFixed.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            toa1_tol = con.Approx(np.nanmean(pvp['TOA1']), atol=1e-11)
            toa2_tol = con.Approx(np.nanmean(pvp['TOA2']), atol=1e-11)
            toa1_min_max = np.array([pvp['TOA1'].min(), pvp['TOA1'].max()])
            toa2_min_max = np.array([pvp['TOA2'].min(), pvp['TOA2'].max()])
            with self.precondition():
                assert parsers.parse_bool(channel_node.find('./TOAFixed'))
                with self.need("TOA1 does not change"):
                    assert toa1_min_max == toa1_tol
                with self.need("TOA2 does not change"):
                    assert toa2_min_max == toa2_tol

            with self.precondition():
                assert not parsers.parse_bool(channel_node.find('./TOAFixed'))
                with self.need("TOA1 and/or TOA2 are not exactly constant"):
                    assert not ((pvp['TOA1'].min() == pvp['TOA1'].max()) and (pvp['TOA2'].min() == pvp['TOA2'].max()))
                    with self.want("TOA1 and/or TOA2 is not almost constant"):
                        assert not ((toa1_min_max == toa1_tol) and (toa2_min_max == toa2_tol))

    @per_channel
    def check_channel_srpfixed(self, channel_id, channel_node):
        """
        PVP agrees with SRPFixed.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.precondition():
                assert parsers.parse_bool(channel_node.find('./SRPFixed'))
                with self.need("SRPPos is fixed"):
                    assert con.Approx(np.nanmean(pvp['SRPPos'], axis=0), atol=1e-3) == pvp['SRPPos']

            with self.precondition():
                assert not parsers.parse_bool(channel_node.find('./SRPFixed'))
                with self.need("SRPPos is not exactly fixed"):
                    assert not np.array_equal(np.nanmin(pvp['SRPPos'], axis=0), np.nanmax(pvp['SRPPos'], axis=0))
                    with self.want("SRPPos is not approximately fixed"):
                        assert not (con.Approx(np.nanmean(pvp['SRPPos'], axis=0), atol=1e-3) == pvp['SRPPos'])

    def check_file_fxfixed(self):
        """
        The FXFixedCPHD element matches the rest of the file.
        """

        fxc_vals = np.array([float(elem.text) for elem in self.xml.findall('./Channel/Parameters/FxC')])
        fxc_minmax = np.array([fxc_vals.min(), fxc_vals.max()])
        fxc_tol = con.Approx(fxc_vals.mean())
        fx_bw_vals = np.array([float(elem.text) for elem in self.xml.findall('./Channel/Parameters/FxBW')])
        fx_bw_minmax = np.array([fx_bw_vals.min(), fx_bw_vals.max()])
        fx_bw_tol = con.Approx(fx_bw_vals.mean())
        with self.precondition():
            assert parsers.parse_bool(self.xml.find('./Channel/FXFixedCPHD'))
            with self.need("All channels have FXFixed"):
                assert all(parsers.parse_bool(elem) for elem in self.xml.findall('./Channel/Parameters/FXFixed'))
            with self.need("All channels have same FxC"):
                assert fxc_minmax == fxc_tol
            with self.need("All channels have same FxBW"):
                assert fx_bw_minmax == fx_bw_tol

        with self.precondition():
            assert not parsers.parse_bool(self.xml.find('./Channel/FXFixedCPHD'))
            assert all(parsers.parse_bool(elem) for elem in self.xml.findall('./Channel/Parameters/FXFixed'))
            with self.need("Channels are not the same"):
                assert not (fxc_vals.min() == fxc_vals.max() and fx_bw_vals.min() == fx_bw_vals.max())

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            fx1_tol = con.Approx(np.nanmean(pvp['FX1']))
            fx2_tol = con.Approx(np.nanmean(pvp['FX2']))
            fx1_min_max = np.array([pvp['FX1'].min(), pvp['FX1'].max()])
            fx2_min_max = np.array([pvp['FX2'].min(), pvp['FX2'].max()])
            with self.precondition():
                assert parsers.parse_bool(self.xml.find('./Channel/FXFixedCPHD'))
                with self.need("FX1 does not change"):
                    assert fx1_min_max == fx1_tol
                with self.need("FX2 does not change"):
                    assert fx2_min_max == fx2_tol

            with self.precondition():
                assert not parsers.parse_bool(self.xml.find('./Channel/FXFixedCPHD'))
                with self.need("FX1 and/or FX2 are not exactly constant"):
                    assert not ((pvp['FX1'].min() == pvp['FX1'].max()) and (pvp['FX2'].min() == pvp['FX2'].max()))
                    with self.want("FX1 and/or FX2 is not almost constant"):
                        assert not ((fx1_min_max == fx1_tol) and (fx2_min_max == fx2_tol))

    def check_file_toafixed(self):
        """
        The TOAFixedCPHD element matches the rest of the file.
        """

        with self.precondition():
            assert parsers.parse_bool(self.xml.find('./Channel/TOAFixedCPHD'))
            with self.need("All channels have TOAFixed"):
                assert all(parsers.parse_bool(elem) for elem in self.xml.findall('./Channel/Parameters/TOAFixed'))

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            toa1_tol = con.Approx(np.nanmean(pvp['TOA1']), atol=1e-11)
            toa2_tol = con.Approx(np.nanmean(pvp['TOA2']), atol=1e-11)
            toa1_min_max = np.array([pvp['TOA1'].min(), pvp['TOA1'].max()])
            toa2_min_max = np.array([pvp['TOA2'].min(), pvp['TOA2'].max()])
            with self.precondition():
                assert parsers.parse_bool(self.xml.find('./Channel/TOAFixedCPHD'))
                with self.need("TOA1 does not change"):
                    assert toa1_min_max == toa1_tol
                with self.need("TOA2 does not change"):
                    assert toa2_min_max == toa2_tol

            with self.precondition():
                assert not parsers.parse_bool(self.xml.find('./Channel/TOAFixedCPHD'))
                with self.need("TOA1 and/or TOA2 is not exactly constant"):
                    assert not ((pvp['TOA1'].min() == pvp['TOA1'].max()) and (pvp['TOA2'].min() == pvp['TOA2'].max()))
                    with self.want("TOA1 and/or TOA2 is not almost constant"):
                        assert not ((toa1_min_max == toa1_tol) and (toa2_min_max == toa2_tol))

    def check_file_srpfixed(self):
        """
        The SRPFixedCPHD element matches the rest of the file.
        """

        with self.precondition():
            assert parsers.parse_bool(self.xml.find('./Channel/SRPFixedCPHD'))
            with self.need("All channels have SRPFixed"):
                assert all(parsers.parse_bool(elem) for elem in self.xml.findall('./Channel/Parameters/SRPFixed'))

        with self.precondition():
            assert self.pvps is not None
            pvp = np.concatenate(list(self.pvps.values()))
            with self.precondition():
                assert parsers.parse_bool(self.xml.find('./Channel/SRPFixedCPHD'))
                with self.need("SRPPos is fixed"):
                    assert con.Approx(np.nanmean(pvp['SRPPos'], axis=0), atol=1e-3) == pvp['SRPPos']

            with self.precondition():
                assert not parsers.parse_bool(self.xml.find('./Channel/SRPFixedCPHD'))
                with self.need("SRPPos is not exactly fixed"):
                    assert not np.array_equal(np.nanmin(pvp['SRPPos'], axis=0), np.nanmax(pvp['SRPPos'], axis=0))
                    with self.want("SRPPos is not approximately fixed"):
                        assert not (con.Approx(np.nanmean(pvp['SRPPos'], axis=0), atol=1e-3) == pvp['SRPPos'])

    @per_channel
    def check_channel_signalnormal(self, channel_id, channel_node):
        """
        PVP agrees with SignalNormal.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.precondition():
                assert channel_node.find('./SignalNormal') is not None
                with self.need('SIGNAL PVP present'):
                    assert 'SIGNAL' in pvp.dtype.names
                with self.precondition():
                    assert 'SIGNAL' in pvp.dtype.names
                    with self.need('SignalNormal matches SIGNAL PVPs'):
                        assert np.all(pvp['SIGNAL'] == 1) == parsers.parse_bool(channel_node.find('./SignalNormal'))

    @per_channel
    def check_channel_fxc(self, channel_id, channel_node):
        """
        PVP agrees with FxC.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("FxC is (max(fx2) + min(fx1)) / 2"):
                assert (con.Approx(float(channel_node.findtext('./FxC')))
                        == (np.nanmax(pvp['FX2']) + np.nanmin(pvp['FX1'])) / 2)

    @per_channel
    def check_channel_fxbw(self, channel_id, channel_node):
        """
        PVP agrees with FxBW.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("FxBW is max(fx2) - min(fx1)"):
                assert (con.Approx(float(channel_node.findtext('./FxBW')))
                        == np.nanmax(pvp['FX2']) - np.nanmin(pvp['FX1']))

    @per_channel
    def check_channel_fxbwnoise(self, channel_id, channel_node):
        """
        PVP agrees with FxBWNoise.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.precondition():
                assert channel_node.find('./FxBWNoise') is not None
                with self.need("Domain is FX when FxBWNoise is provided"):
                    assert self.xml.findtext('./Global/DomainType') == 'FX'
                with self.need("FxBWNoise is max(FXN2) - min(FXN1)"):
                    assert (con.Approx(float(channel_node.findtext('./FxBWNoise')))
                            == np.nanmax(pvp['FXN2']) - np.nanmin(pvp['FXN1']))

    @per_channel
    def check_channel_toasaved(self, channel_id, channel_node):
        """
        PVP agrees with TOASaved.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("TOASaved is max(TOA2) - min(TOA1)"):
                assert (con.Approx(float(channel_node.findtext('./TOASaved')))
                        == np.nanmax(pvp['TOA2']) - np.nanmin(pvp['TOA1']))

    @per_channel
    def check_channel_global_txtime(self, channel_id, channel_node):
        """
        PVP within global TxTime1 and TxTime2.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("TxTime is greater than TxTime1"):
                assert np.nanmin(pvp['TxTime']) >= con.Approx(float(self.xml.findtext('./Global/Timeline/TxTime1')))
            with self.need("TxTime is less than TxTime2"):
                assert np.nanmax(pvp['TxTime']) <= con.Approx(float(self.xml.findtext('./Global/Timeline/TxTime2')))

    @per_channel
    def check_channel_global_fxminmax(self, channel_id, channel_node):
        """
        PVP within global FxMin and FxMax.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("FX1 is greater than FxMin"):
                assert np.nanmin(pvp['FX1']) >= con.Approx(float(self.xml.findtext('./Global/FxBand/FxMin')))
            with self.need("FX2 is less than FxMax"):
                assert np.nanmax(pvp['FX2']) <= con.Approx(float(self.xml.findtext('./Global/FxBand/FxMax')))

    @per_channel
    def check_channel_global_toaswath(self, channel_id, channel_node):
        """
        PVP within global TOASwath.
        """

        with self.precondition():
            assert self.pvps is not None
            assert channel_id in self.pvps
            pvp = self.pvps[channel_id]
            with self.need("TOA1 is greater than TOAMin"):
                assert np.nanmin(pvp['TOA1']) >= con.Approx(float(self.xml.findtext('./Global/TOASwath/TOAMin')))
            with self.need("TOA2 is less than TOAMax"):
                assert np.nanmax(pvp['TOA2']) <= con.Approx(float(self.xml.findtext('./Global/TOASwath/TOAMax')))

    @per_channel
    def check_channel_imagearea_polygon(self, channel_id, channel_node):
        """
        Image area polygon is simple and consistent with X1Y1 and X2Y2.
        """

        polygon_node = channel_node.find('./ImageArea/Polygon')
        with self.precondition():
            assert polygon_node is not None
            with self.precondition():
                assert have_shapely
                polygon = self.get_polygon(polygon_node, check=True)
                x1y1 = parsers.parse_xy(channel_node.find('./ImageArea/X1Y1'))
                x2y2 = parsers.parse_xy(channel_node.find('./ImageArea/X2Y2'))
                with self.need("Polygon works with X1Y1"):
                    assert polygon.min(axis=0) == con.Approx(x1y1, atol=1e-3)
                with self.need("Polygon works with X2Y2"):
                    assert polygon.max(axis=0) == con.Approx(x2y2, atol=1e-3)
                with self.need("Polygon is simple"):
                    assert shg.Polygon(polygon).is_simple

    def check_global_imagearea_polygon(self):
        """
        Scene Image area polygon is simple and consistent with X1Y1 and X2Y2.
        """

        scene_coords_node = self.xml.find('./SceneCoordinates')
        polygon_node = scene_coords_node.find('./ImageArea/Polygon')
        with self.precondition():
            assert polygon_node is not None
            with self.precondition():
                assert have_shapely
                polygon = self.get_polygon(polygon_node, check=True)
                x1y1 = parsers.parse_xy(scene_coords_node.find('./ImageArea/X1Y1'))
                x2y2 = parsers.parse_xy(scene_coords_node.find('./ImageArea/X2Y2'))
                with self.need("Polygon works with X1Y1"):
                    assert polygon.min(axis=0) == con.Approx(x1y1, atol=1e-3)
                with self.need("Polygon works with X2Y2"):
                    assert polygon.max(axis=0) == con.Approx(x2y2, atol=1e-3)
                with self.need("Polygon is simple"):
                    assert shg.Polygon(polygon).is_simple

    def get_polygon(self, polygon_node, check=False, reverse=False, parser=parsers.parse_xy):
        vertex_nodes = sorted(list(polygon_node), key=lambda x: int(x.attrib['index']))
        polygon = np.asarray([parser(vertex) for vertex in vertex_nodes])
        if check:
            with self.need("Polygon indices are all present"):
                assert [int(x.attrib['index']) for x in vertex_nodes] == list(range(1, len(vertex_nodes) + 1))
            if 'size' in polygon_node.attrib:
                size = int(polygon_node.attrib['size'])
                with self.need("Polygon size attribute matches the number of vertices"):
                    assert size == len(vertex_nodes)
            shg_polygon = shg.Polygon(polygon)
            with self.need("Polygon is simple"):
                assert shg_polygon.is_simple
            with self.need("Polygon is clockwise"):
                assert not shg_polygon.exterior.is_ccw
        return polygon

    def check_geoinfo_polygons(self):
        """
        GeoInfo polygons are simple polygons in clockwise order.
        """

        geo_polygons = self.xml.findall('.//GeoInfo/Polygon')
        with self.precondition():
            assert geo_polygons
            with self.precondition():
                assert have_shapely
                for geo_polygon in geo_polygons:
                    with self.need(etree.ElementTree(self.xml).getpath(geo_polygon)):
                        self.get_polygon(geo_polygon, check=True, reverse=True, parser=parsers.parse_ll)

    def check_image_area_corner_points(self):
        """
        The corner points represent a simple quadrilateral in clockwise order.
        """

        with self.precondition():
            assert have_shapely
            iacp_node = self.xml.find('./SceneCoordinates/ImageAreaCornerPoints')
            iacp = self.get_polygon(iacp_node, check=True, reverse=True, parser=parsers.parse_ll)
            with self.need("4 corner points"):
                assert len(iacp) == 4

    def check_extended_imagearea_polygon(self):
        """
        Scene extended area polygon is simple and consistent with X1Y1 and X2Y2.
        """

        scene_coords_node = self.xml.find('./SceneCoordinates')
        extended_area_node = scene_coords_node.find('./ExtendedArea')
        with self.precondition():
            assert extended_area_node is not None
            extended_area_polygon_node = extended_area_node.find('./Polygon')
            with self.precondition():
                assert extended_area_polygon_node is not None
                with self.precondition():
                    assert have_shapely
                    extended_area_polygon = self.get_polygon(extended_area_polygon_node, check=True)
                    extended_x1y1 = parsers.parse_xy(extended_area_node.find('./X1Y1'))
                    extended_x2y2 = parsers.parse_xy(extended_area_node.find('./X2Y2'))

                    with self.need("Polygon works with X1Y1"):
                        assert extended_area_polygon.min(axis=0) == con.Approx(extended_x1y1, atol=1e-3)
                    with self.need("Polygon works with X2Y2"):
                        assert extended_area_polygon.max(axis=0) == con.Approx(extended_x2y2, atol=1e-3)
                    polygon_node = scene_coords_node.find('./ImageArea/Polygon')
                    with self.precondition():
                        assert polygon_node is not None
                        polygon = self.get_polygon(polygon_node)
                        shg_extended = shg.Polygon(extended_area_polygon)
                        shg_polygon = shg.Polygon(polygon)
                        with self.need("Extended area polygon covers image area polygon"):
                            assert shg.Polygon(shg_extended).covers(shg_polygon)

    @per_channel
    def check_channel_imagearea_x1y1(self, channel_id, channel_node):
        """
        Image area X1Y1 and X2Y2 work with global X1Y1 and X2Y2.
        """

        with self.precondition():
            assert channel_node.find('./ImageArea') is not None
            x1y1 = parsers.parse_xy(channel_node.find('./ImageArea/X1Y1'))
            x2y2 = parsers.parse_xy(channel_node.find('./ImageArea/X2Y2'))
            global_x1y1 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ImageArea/X1Y1'))
            global_x2y2 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ImageArea/X2Y2'))
            with self.need("X1Y1"):
                assert x1y1 >= con.Approx(global_x1y1)
            with self.need("X2Y2"):
                assert x2y2 <= con.Approx(global_x2y2)

    def check_extended_imagearea_x1y1_x2y2(self):
        """
        Extended image area contains the image area.
        """

        with self.precondition():
            assert self.xml.find('./SceneCoordinates/ExtendedArea') is not None
            extended_x1y1 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ExtendedArea/X1Y1'))
            extended_x2y2 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ExtendedArea/X2Y2'))
            global_x1y1 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ImageArea/X1Y1'))
            global_x2y2 = parsers.parse_xy(self.xml.find('./SceneCoordinates/ImageArea/X2Y2'))
            with self.need("Extended X1Y1 less than image area X1Y1"):
                assert extended_x1y1 <= con.Approx(global_x1y1)
            with self.need("Extended X2Y2 geater than image area X2Y2"):
                assert extended_x2y2 >= con.Approx(global_x2y2)

    @per_channel
    def check_channel_signal_data(self, channel_id, channel_node):
        """
        Sample data is all finite.
        """

        with self.precondition():
            assert self.header is not None
            format_string = self.xml.findtext('./Data/SignalArrayFormat')
            signal_dtype = cphd1_utils.binary_format_string_to_dtype(format_string)

            channel_data_node = get_by_id(self.xml, './Data/Channel', channel_id)
            signal_offset = int(channel_data_node.findtext('./SignalArrayByteOffset'))
            num_vectors = int(channel_data_node.findtext('./NumVectors'))
            num_samples = int(channel_data_node.findtext('./NumSamples'))
            signal_end = signal_offset + num_vectors * num_samples * signal_dtype.itemsize
            signal_file_offset = self.header['SIGNAL_BLOCK_BYTE_OFFSET'] + signal_offset
            with self.need("Channel signal fits in signal block"):
                assert self.header['SIGNAL_BLOCK_SIZE'] >= signal_end
            with self.precondition():
                assert self.check_signal_data
                assert self.filename is not None
                assert format_string == 'CF8'
                with self.need("All signal samples are finite and not NaN"):
                    assert np.all(np.isfinite(np.memmap(self.filename, signal_dtype.newbyteorder('B'), mode='r',
                                                        offset=signal_file_offset,
                                                        shape=(num_vectors, num_samples),
                                                        order='C')))

    def check_image_grid_exists(self):
        """
        Verify that the ImageGrid is defined
        """
        with self.precondition():
            with self.want("It is recommended to populate SceneCoordinates.ImageGrid for processing purposes"):
                assert self.xml.find('./SceneCoordinates/ImageGrid') is not None

    def check_pad_header_xml(self):
        """
        The pad between the header and XML is 0.
        """

        with self.precondition():
            assert self.header is not None
            with self.want("XML appears early in the file"):
                assert self.header["XML_BLOCK_BYTE_OFFSET"] < 2**28
            assert self.filename is not None
            with open(self.filename, 'rb') as fp:
                before_xml = fp.read(self.header['XML_BLOCK_BYTE_OFFSET'])
                first_form_feed = before_xml.find('\f\n'.encode('utf-8'))
                with self.need("header section terminator exists before XML"):
                    assert b'\f\n' in before_xml
                with self.want("Pad is 0"):
                    assert np.all(np.frombuffer(before_xml[first_form_feed+2:], dtype=np.uint8) == 0)

    def check_pad_after_xml(self):
        """
        The pad after XML is 0.
        """

        with self.precondition():
            assert self.header is not None
            assert self.filename is not None
            xml_end = self.header['XML_BLOCK_BYTE_OFFSET'] + self.header['XML_BLOCK_SIZE']
            if 'SUPPORT_BLOCK_BYTE_OFFSET' in self.header:
                num_bytes_after_xml = self.header['SUPPORT_BLOCK_BYTE_OFFSET'] - xml_end
                next_block = 'Support'
            else:
                num_bytes_after_xml = self.header['PVP_BLOCK_BYTE_OFFSET'] - xml_end
                next_block = 'PVP'
            with self.need("{} comes after XML".format(next_block)):
                assert num_bytes_after_xml - 2 >= 0
            bytes_after_xml = np.memmap(self.filename, np.uint8, mode='r', offset=xml_end, shape=num_bytes_after_xml)
            with self.need("Section terminator exists"):
                assert bytes_after_xml[:2].tobytes() == b'\f\n'
            with self.want("Pad is 0"):
                assert np.all(np.frombuffer(bytes_after_xml[2:], dtype=np.uint8) == 0)

    def check_pad_after_support(self):
        """
        The pad after support arrays is 0.
        """

        with self.precondition():
            assert self.header is not None
            assert self.filename is not None
            assert 'SUPPORT_BLOCK_BYTE_OFFSET' in self.header
            support_end = self.header['SUPPORT_BLOCK_BYTE_OFFSET'] + self.header['SUPPORT_BLOCK_SIZE']
            num_bytes_after_support = self.header['PVP_BLOCK_BYTE_OFFSET'] - support_end
            with self.need("PVP comes after Support"):
                assert num_bytes_after_support >= 0
            bytes_after_support = np.memmap(self.filename, np.uint8, mode='r', offset=support_end,
                                            shape=num_bytes_after_support)
            with self.want("Pad is 0"):
                assert np.all(np.frombuffer(bytes_after_support, dtype=np.uint8) == 0)

    def check_pad_after_pvp(self):
        """
        The pad after PVPs is 0.
        """

        with self.precondition():
            assert self.header is not None
            assert self.filename is not None
            pvp_end = self.header['PVP_BLOCK_BYTE_OFFSET'] + self.header['PVP_BLOCK_SIZE']
            num_bytes_after_pvp = self.header['SIGNAL_BLOCK_BYTE_OFFSET'] - pvp_end
            with self.need("Signal comes after PVP"):
                assert num_bytes_after_pvp >= 0
            bytes_after_pvp = np.memmap(self.filename, np.uint8, mode='r', offset=pvp_end,
                                        shape=num_bytes_after_pvp)
            with self.want("Pad is 0"):
                assert np.all(np.frombuffer(bytes_after_pvp, dtype=np.uint8) == 0)

    def check_signal_at_end_of_file(self):
        with self.precondition():
            assert self.header is not None
            assert self.filename is not None
            with self.need("Signal is at the end of the file"):
                file_size = os.stat(self.filename).st_size
                assert file_size == self.header['SIGNAL_BLOCK_BYTE_OFFSET'] + self.header['SIGNAL_BLOCK_SIZE']

    def check_scene_plane_axis_vectors(self):
        """
        Scene plane axis vectors are orthonormal.
        """

        planar_node = self.xml.find('./SceneCoordinates/ReferenceSurface/Planar')
        with self.precondition():
            assert planar_node is not None
            uiax = parsers.parse_xyz(planar_node.find('./uIAX'))
            uiay = parsers.parse_xyz(planar_node.find('./uIAY'))
            with self.need("uIAX is unit"):
                assert np.linalg.norm(uiax) == con.Approx(1)
            with self.need("uIAY is unit"):
                assert np.linalg.norm(uiay) == con.Approx(1)
            with self.need("uIAX and uIAY are orthogonal (dot is zero)"):
                assert np.dot(uiax, uiay) == con.Approx(0, atol=1e-6)

    def check_global_txtime_limits(self):
        """
        The Global TxTime1 and TxTime2 match the PVPs.
        """

        with self.precondition():
            assert self.pvps is not None
            txtime1_chan = min(np.nanmin(x['TxTime']) for x in self.pvps.values())
            txtime2_chan = max(np.nanmax(x['TxTime']) for x in self.pvps.values())
            with self.need("Timeline TxTime1 matches PVP"):
                assert txtime1_chan == con.Approx(float(self.xml.findtext('./Global/Timeline/TxTime1')))
            with self.need("Timeline TxTime2 matches PVP"):
                assert txtime2_chan == con.Approx(float(self.xml.findtext('./Global/Timeline/TxTime2')))

    def check_global_fx_band(self):
        """
        The Global FXBand matches the PVPs.
        """

        with self.precondition():
            assert self.pvps is not None
            fx1min_chan = min(np.nanmin(x['FX1']) for x in self.pvps.values())
            fx2max_chan = max(np.nanmax(x['FX2']) for x in self.pvps.values())
            with self.need("FxMin matches PVP"):
                assert fx1min_chan == con.Approx(float(self.xml.findtext('./Global/FxBand/FxMin')))
            with self.need("FxMax match PVP"):
                assert fx2max_chan == con.Approx(float(self.xml.findtext('./Global/FxBand/FxMax')))

    def check_global_toaswath(self):
        """
        The Global TOASwath matches the PVPs.
        """

        with self.precondition():
            assert self.pvps is not None
            toa1min_chan = min(np.nanmin(x['TOA1']) for x in self.pvps.values())
            toa2max_chan = max(np.nanmax(x['TOA2']) for x in self.pvps.values())
            with self.need("TOAMin matches PVP"):
                assert toa1min_chan == pytest.approx(float(self.xml.findtext('./Global/TOASwath/TOAMin')))
            with self.need("TOAMax matches PVP"):
                assert toa2max_chan == pytest.approx(float(self.xml.findtext('./Global/TOASwath/TOAMax')))

    def _check_ids_in_channel_for_optional_branch(self, branch_name):
        with self.precondition():
            assert self.xml.find('./{}'.format(branch_name)) is not None
            with self.want("{} present in /Channel/Parameters".format(branch_name)):
                assert self.xml.find('./Channel/Parameters/{}'.format(branch_name)) is not None

    def check_antenna_ids_in_channel(self):
        """
        If the Antenna branch exists, then Antenna is also present in /Channel/Parameters
        """

        self._check_ids_in_channel_for_optional_branch('Antenna')

    def check_txrcv_ids_in_channel(self):
        """
        If the TxRcv branch exists, then TxRcv is also present in /Channel/Parameters
        """

        self._check_ids_in_channel_for_optional_branch('TxRcv')

    def _check_refgeom_parameters(self, xml_node, expected_parameters):
        for xml_path, expected_value in expected_parameters.items():
            if isinstance(expected_value, np.ndarray) and expected_value.size == 3:
                parser = parsers.parse_xyz
            else:
                parser = parsers.parse_text

            approx_args = {}
            if 'Angle' in xml_path:
                approx_args['atol'] = 1
            elif xml_path.endswith('Time'):
                approx_args['atol'] = 1e-6
            else:
                approx_args['atol'] = 1e-6

            actual_value = parser(xml_node.find('./{}'.format(xml_path)))
            if isinstance(expected_value, numbers.Number):
                actual_value = con.Approx(actual_value, **approx_args)

            with self.need('{} matches defined PVP/calculation'.format(xml_path)):
                if isinstance(expected_value, np.ndarray) and expected_value.dtype.name == 'float64':
                    assert np.all(np.cast['float32'](expected_value) == np.cast['float32'](actual_value))
                else:
                    assert np.all(expected_value == actual_value)

    def check_refgeom_root(self):
        """
        The ReferenceGeometry branch root parameters match the PVPs/defined calculations
        """

        with self.precondition():
            assert self.pvps is not None
            refgeom = calc_refgeom_parameters(self.xml, self.pvps).refgeom
            self._check_refgeom_parameters(self.xml.find('./ReferenceGeometry'), refgeom)

    def check_refgeom_monostatic(self):
        """
        The ReferenceGeometry branch Monostatic parameters are present and match the PVPs/defined calculations
        """

        with self.precondition():
            assert self.xml.findtext('./CollectionID/CollectType') == 'MONOSTATIC'
            refgeom_mono = self.xml.find('./ReferenceGeometry/Monostatic')
            with self.need("ReferenceGeometry type matches CollectType"):
                assert refgeom_mono is not None

            assert self.pvps is not None
            monostat = calc_refgeom_parameters(self.xml, self.pvps).monostat
            self._check_refgeom_parameters(refgeom_mono, monostat)

    def check_refgeom_bistatic(self):
        """
        The ReferenceGeometry branch Bistatic parameters are present and match the PVPs/defined calculations
        """

        with self.precondition():
            assert self.xml.findtext('./CollectionID/CollectType') == 'BISTATIC'
            refgeom_bistat = self.xml.find('./ReferenceGeometry/Bistatic')
            with self.need("ReferenceGeometry type matches CollectType"):
                assert refgeom_bistat is not None

            assert self.pvps is not None
            bistat = calc_refgeom_parameters(self.xml, self.pvps).bistat
            self._check_refgeom_parameters(refgeom_bistat, bistat)

    def check_unconnected_ids(self):
        """
        Check that all identifiers are connected back to the Data branch.
        """

        with self.precondition():
            assert have_networkx
            id_graph = make_id_graph(self.xml)
            data_subgraph = id_graph.subgraph(nx.shortest_path(id_graph, 'Data'))
            no_data_subgraph = id_graph.copy()
            no_data_subgraph.remove_nodes_from(data_subgraph)
            unconnected_ids = [] if no_data_subgraph is None else [x for x in no_data_subgraph.nodes if '<' in x]
            with self.want("All IDs connect to Data branch"):
                assert not unconnected_ids


def calc_refgeom_parameters(xml, pvps):
    """
    Calculate expected reference geometry parameters given CPHD XML and PVPs (CPHD1.0.1, Sec 6.5)
    """

    def unit(vec):
        return vec / np.linalg.norm(vec)

    # 6.5.1 - Reference Vector Parameters
    ref_id = xml.findtext('./Channel/RefChId')
    ref_chan_parameters = get_by_id(xml, './Channel/Parameters/', ref_id)
    v_ch_ref = int(ref_chan_parameters.findtext('RefVectorIndex'))

    ref_vector = pvps[ref_id][v_ch_ref]
    txc = ref_vector['TxTime']
    xmt = ref_vector['TxPos']
    vxmt = ref_vector['TxVel']
    trc_srp = ref_vector['RcvTime']
    rcv = ref_vector['RcvPos']
    vrcv = ref_vector['RcvVel']
    srp = ref_vector['SRPPos']

    ref_dwelltimes = get_by_id(xml, './Channel/Parameters/', ref_id).find('./DwellTimes')
    ref_cod_id = ref_dwelltimes.findtext('CODId')
    ref_dwell_id = ref_dwelltimes.findtext('DwellId')
    xy2cod = parsers.parse_poly2d(get_by_id(xml, './Dwell/CODTime', ref_cod_id).find('./CODTimePoly'))
    xy2dwell = parsers.parse_poly2d(get_by_id(xml, './Dwell/DwellTime', ref_dwell_id).find('./DwellTimePoly'))

    # (1) See also Section 6.2
    srp_llh = geocoords.ecf_to_geodetic(srp, 'latlong')
    srp_lat, srp_lon = np.deg2rad(srp_llh[:2])

    ref_surface = xml.find('./SceneCoordinates/ReferenceSurface/Planar')
    if ref_surface is None:  # TODO: Add HAE
        raise NotImplementedError("Non-Planar reference surfaces (e.g. HAE) are currently not supported.")

    iax = parsers.parse_xyz(ref_surface.find('./uIAX'))
    iay = parsers.parse_xyz(ref_surface.find('./uIAY'))
    iarp = parsers.parse_xyz(xml.find('./SceneCoordinates/IARP/ECF'))
    srp_iac = np.dot([iax, iay, unit(np.cross(iax, iay))], srp - iarp)

    # (2)
    srp_dec = np.linalg.norm(srp)
    uec_srp = srp / srp_dec

    # (3)
    ueast = np.array((-np.sin(srp_lon),
                      np.cos(srp_lon),
                      0))
    unor = np.array((-np.sin(srp_lat) * np.cos(srp_lon),
                     -np.sin(srp_lat) * np.sin(srp_lon),
                     np.cos(srp_lat)))
    uup = np.array((np.cos(srp_lat) * np.cos(srp_lon),
                    np.cos(srp_lat) * np.sin(srp_lon),
                    np.sin(srp_lat)))

    # (4)
    r_xmt_srp = np.linalg.norm(xmt - srp)
    r_rcv_srp = np.linalg.norm(rcv - srp)

    # (5)
    t_ref = txc + r_xmt_srp / (r_xmt_srp + r_rcv_srp) * (trc_srp - txc)

    # (6)
    t_cod_srp = npp.polyval2d(*srp_iac[:2], c=xy2cod)
    t_dwell_srp = npp.polyval2d(*srp_iac[:2], c=xy2dwell)

    # (7)
    refgeom = {'SRP/ECF': srp,
               'SRP/IAC': srp_iac,
               'ReferenceTime': t_ref,
               'SRPCODTime': t_cod_srp,
               'SRPDwellTime': t_dwell_srp}

    def calc_apc_parameters(position, velocity):
        """Calculate APC parameters given a position and velocity.

        Use arp/varp variable substitution for similarity with CPHD v1.0.1 Section 6.5.2

        """
        # (1)
        arp = position
        varp = velocity

        # (2)
        r_arp_srp = np.linalg.norm(arp - srp)
        uarp = (arp - srp) / r_arp_srp
        rdot_arp_srp = np.dot(uarp, varp)

        # (3)
        arp_dec = np.linalg.norm(arp)
        uec_arp = arp / arp_dec

        # (4)
        ea_arp = np.arccos(np.dot(uec_arp, uec_srp))
        rg_arp_srp = srp_dec * ea_arp

        # (5)
        varp_m = np.linalg.norm(varp)
        uvarp = varp / varp_m
        left = np.cross(uec_arp, uvarp)

        # (6)
        look = +1 if np.dot(left, uarp) < 0 else -1
        side_of_track = 'L' if look == +1 else 'R'

        # (7)
        dca = np.arccos(-rdot_arp_srp / varp_m)

        # (8)
        ugpz = uup
        gpy = np.cross(uup, uarp)
        ugpy = unit(gpy)
        ugpx = np.cross(ugpy, ugpz)

        # (9)
        graz = np.arccos(np.dot(uarp, ugpx))
        # incidence angle in (15)

        # (10)
        gpx_n = np.dot(ugpx, unor)
        gpx_e = np.dot(ugpx, ueast)
        azim = np.arctan2(gpx_e, gpx_n)

        # (11)
        uspn = unit(look * np.cross(uarp, uvarp))

        # (12)
        twst = -np.arcsin(np.dot(uspn, ugpy))

        # (13)
        slope = np.arccos(np.dot(ugpz, uspn))

        # (14)
        lodir_n = np.dot(-uspn, unor)
        lodir_e = np.dot(-uspn, ueast)
        lo_ang = np.arctan2(lodir_e, lodir_n)

        # (15)
        return {'ARPPos': arp,
                'ARPVel': varp,
                'SideOfTrack': side_of_track,
                'SlantRange': r_arp_srp,
                'GroundRange': rg_arp_srp,
                'DopplerConeAngle': np.rad2deg(dca),
                'GrazeAngle': np.rad2deg(graz),
                'IncidenceAngle': 90 - np.rad2deg(graz),
                'AzimuthAngle': np.rad2deg(azim) % 360,
                'TwistAngle': np.rad2deg(twst),
                'SlopeAngle': np.rad2deg(slope),
                'LayoverAngle': np.rad2deg(lo_ang) % 360}

    def calc_apc_parameters_bi(platform, time, position, velocity):
        apc_params = calc_apc_parameters(position, velocity)
        apc_params['Time'] = time
        apc_params['Pos'] = apc_params.pop('ARPPos')
        apc_params['Vel'] = apc_params.pop('ARPVel')
        del apc_params['TwistAngle']
        del apc_params['SlopeAngle']
        del apc_params['LayoverAngle']

        # Conditions unique to bistatic (6.5.3 18-19)
        if np.linalg.norm(velocity) == 0:
            apc_params['DopplerConeAngle'] = 90
            apc_params['SideOfTrack'] = 'L'
        if apc_params['GroundRange'] == 0:
            apc_params['GrazeAngle'] = 90
            apc_params['IncidenceAngle'] = 0
            apc_params['AzimuthAngle'] = 0

        return {'{platform}Platform/{k}'.format(platform=platform, k=k): v for k, v in apc_params.items()}

    def calc_refgeom_mono():
        return calc_apc_parameters((xmt + rcv) / 2, (vxmt + vrcv) / 2)

    def calc_refgeom_bi():
        # 6.5.3 Reference Geometry: Collect Type = BISTATIC
        # (1)
        uxmt = (xmt - srp) / r_xmt_srp
        rdot_xmt_srp = np.dot(uxmt, vxmt)
        uxmtdot = (vxmt - np.dot(rdot_xmt_srp, uxmt)) / r_xmt_srp

        # (2)
        urcv = (rcv - srp) / r_rcv_srp
        rdot_rcv_srp = np.dot(urcv, vrcv)
        urcvdot = (vrcv - np.dot(rdot_rcv_srp, urcv)) / r_rcv_srp

        # (3)
        bp = (uxmt + urcv) / 2
        bpdot = (uxmtdot + urcvdot) / 2

        # (4)
        bp_mag = np.linalg.norm(bp)
        bistat_ang = 2 * np.arccos(bp_mag)

        # (5)
        bistat_ang_rate = 0.0 if bp_mag in (0, 1) else -(4 * np.dot(bp, bpdot) / np.sin(bistat_ang))

        # (6)
        ugpz = uup
        bp_gpz = np.dot(bp, ugpz)
        bp_gp = bp - np.dot(bp_gpz, ugpz)
        bp_gpx = np.linalg.norm(bp_gp)

        # (7)
        ubgpx = bp_gp / bp_gpx
        ubgpy = np.cross(ugpz, ubgpx)

        # (8)
        bistat_graz = np.arctan(bp_gpz / bp_gpx)

        # (9)
        bgpx_n = np.dot(ubgpx, unor)
        bgpx_e = np.dot(ubgpx, ueast)
        bistat_azim = np.arctan2(bgpx_e, bgpx_n)

        # (10)
        bpdot_bgpy = np.dot(bpdot, ubgpy)
        bistat_azim_rate = -(bpdot_bgpy / bp_gpx)

        # (11)
        bistat_sgn = +1 if bpdot_bgpy > 0 else -1

        # (12)
        ubp = bp / bp_mag
        bpdotp = np.dot(bpdot, ubp) * ubp
        bpdotn = bpdot - bpdotp

        # (13)
        bipn = bistat_sgn * np.cross(bp, bpdotn)
        ubipn = unit(bipn)

        # (14)
        bistat_twst = -np.arcsin(np.dot(ubipn, ubgpy))

        # (15)
        bistat_slope = np.arccos(np.dot(ugpz, ubipn))

        # (16)
        b_lodir_n = np.dot(-ubipn, unor)
        b_lodir_e = np.dot(-ubipn, ueast)
        bistat_lo_ang = np.arctan2(b_lodir_e, b_lodir_n)

        # Caveat in (6)
        if bp_gpx == 0:
            bistat_azim = 0
            bistat_azim_rate = 0
            bistat_graz = 0
            bistat_twst = 0
            bistat_slope = 0
            bistat_lo_ang = 0

        # Caveat in (10)
        if bpdot_bgpy == 0:
            bistat_twst = 0
            bistat_slope = 0
            bistat_lo_ang = 0

        refgeom_bi = {
            # (17)
            'AzimuthAngle': np.rad2deg(bistat_azim) % 360,
            'AzimuthAngleRate': np.rad2deg(bistat_azim_rate),
            'BistaticAngle': np.rad2deg(bistat_ang),
            'BistaticAngleRate': np.rad2deg(bistat_ang_rate),
            'GrazeAngle': np.rad2deg(bistat_graz),
            'TwistAngle': np.rad2deg(bistat_twst),
            'SlopeAngle': np.rad2deg(bistat_slope),
            'LayoverAngle': np.rad2deg(bistat_lo_ang) % 360
        }
        # (18)
        refgeom_bi.update(calc_apc_parameters_bi('Tx', txc, xmt, vxmt))
        # (19)
        refgeom_bi.update(calc_apc_parameters_bi('Rcv', trc_srp, rcv, vrcv))
        return refgeom_bi

    mono = calc_refgeom_mono()
    bistat = calc_refgeom_bi()

    return collections.namedtuple('refgeom_params', 'refgeom monostat bistat')(refgeom, mono, bistat)


def make_id_graph(xml):
    """
    Make an undirected graph with CPHD identifiers as nodes and edges from correspondence and hierarchy.

    Nodes are named as {xml_path}<{id}, e.g. /Data/Channel/Identifier<Ch1
    There is a single "Data" node formed from the Data branch root that signifies data that can be read from the file

    Args
    ----
    xml: `lxml.etree.ElementTree.Element`
        Root CPHD XML node

    Returns
    -------
    id_graph: `networkx.Graph`
        Undirected graph

            * nodes: Data node, CPHD identifiers
            * edges: Parent identifiers to child identifiers; corresponding identifiers across XML branches

    """

    id_graph = nx.Graph()

    def add_id_nodes_from_path(xml_path):
        id_graph.add_nodes_from(["{}<{}".format(xml_path, n.text) for n in xml.findall('.' + xml_path)])

    def add_id_nodes_from_path_with_connected_root(xml_path):
        root_node = xml_path.split('/')[1]
        id_graph.add_edges_from(zip(itertools.repeat(root_node),
                                    ["{}<{}".format(xml_path, n.text) for n in xml.findall('.' + xml_path)]))

    def get_id_from_node_name(node_name):
        return node_name.split('<')[-1]

    def connect_matching_id_nodes(path_a, path_b):
        all_nodes = list(id_graph.nodes)
        all_a = {get_id_from_node_name(x): x for x in all_nodes if x.split('<')[0] == path_a}
        all_b = {get_id_from_node_name(x): x for x in all_nodes if x.split('<')[0] == path_b}

        for k in set(all_a).intersection(all_b):
            id_graph.add_edge(all_a[k], all_b[k])

    def add_and_connect_id_nodes(path_a, path_b):
        add_id_nodes_from_path(path_a)
        add_id_nodes_from_path(path_b)
        connect_matching_id_nodes(path_a, path_b)

    def add_and_connect_children(parent_path, parent_id_name, children_paths):
        for parent in xml.findall('.' + parent_path):
            parent_id = parent.findtext(parent_id_name)
            for child_path in children_paths:
                for child in parent.findall('.' + child_path):
                    id_graph.add_edge('{}/{}<{}'.format(parent_path, parent_id_name, parent_id),
                                      '{}/{}<{}'.format(parent_path, child_path, child.text))

    add_id_nodes_from_path_with_connected_root('/Data/Channel/Identifier')
    add_id_nodes_from_path_with_connected_root('/Data/SupportArray/Identifier')

    channel_children = ['DwellTimes/CODId', 'DwellTimes/DwellId']
    channel_children += ['Antenna/'+ident for ident in ('TxAPCId', 'TxAPATId', 'RcvAPCId', 'RcvAPATId')]
    channel_children += ['TxRcv/TxWFId', 'TxRcv/RcvId']
    add_and_connect_children('/Channel/Parameters', 'Identifier', channel_children)

    connect_matching_id_nodes('/Data/Channel/Identifier', '/Channel/Parameters/Identifier')

    add_and_connect_id_nodes('/Data/SupportArray/Identifier', '/SupportArray/IAZArray/Identifier')
    add_and_connect_id_nodes('/Data/SupportArray/Identifier', '/SupportArray/AntGainPhase/Identifier')
    add_and_connect_id_nodes('/Data/SupportArray/Identifier', '/SupportArray/AddedSupportArray/Identifier')

    add_and_connect_id_nodes('/Channel/Parameters/DwellTimes/CODId', '/Dwell/CODTime/Identifier')
    add_and_connect_id_nodes('/Channel/Parameters/DwellTimes/DwellId', '/Dwell/DwellTime/Identifier')

    add_and_connect_id_nodes('/Antenna/AntCoordFrame/Identifier', '/Antenna/AntPhaseCenter/ACFId')
    add_and_connect_children('/Antenna/AntPattern', 'Identifier',
                             ('GainPhaseArray/ArrayId', 'GainPhaseArray/ElementId'))
    add_and_connect_children('/Antenna/AntPhaseCenter', 'Identifier', ('ACFId',))

    add_and_connect_id_nodes('/Channel/Parameters/Antenna/TxAPCId', '/Antenna/AntPhaseCenter/Identifier')
    add_and_connect_id_nodes('/Channel/Parameters/Antenna/TxAPATId', '/Antenna/AntPattern/Identifier')
    add_and_connect_id_nodes('/Channel/Parameters/Antenna/RcvAPCId', '/Antenna/AntPhaseCenter/Identifier')
    add_and_connect_id_nodes('/Channel/Parameters/Antenna/RcvAPATId', '/Antenna/AntPattern/Identifier')

    connect_matching_id_nodes('/SupportArray/AntGainPhase/Identifier', '/Antenna/AntPattern/GainPhaseArray/ArrayId')
    connect_matching_id_nodes('/SupportArray/AntGainPhase/Identifier', '/Antenna/AntPattern/GainPhaseArray/ElementId')

    add_and_connect_id_nodes('/Channel/Parameters/TxRcv/TxWFId', '/TxRcv/TxWFParameters/Identifier')
    add_and_connect_id_nodes('/Channel/Parameters/TxRcv/RcvId', '/TxRcv/RcvParameters/Identifier')

    return id_graph


def main(args=None):
    """
    CphdConsistency CLI tool.  Prints results to stdout.

    Parameters
    ----------
    args: None|List[str]
        List of CLI argument strings.  If None use sys.argv
    """
    parser = argparse.ArgumentParser(description="Analyze a CPHD and display inconsistencies")
    parser.add_argument('cphd_or_xml')
    parser.add_argument('-v', '--verbose', default=0,
                        action='count', help="Increase verbosity (can be specified more than once >4 doesn't help)")
    parser.add_argument('--schema', help="Use a supplied schema file", default=DEFAULT_SCHEMA)
    parser.add_argument('--noschema', action='store_const', const=None, dest='schema', help="Disable schema checks")
    parser.add_argument('--signal-data', action='store_true', help="Check the signal data for NaN and +/- Inf")
    config = parser.parse_args(args)

    # Some questionable abuse of the pytest internals
    import ast
    import _pytest.assertion.rewrite
    base, ext = os.path.splitext(__file__)  # python2 can return the '*.pyc' file
    with open(base + '.py', 'r') as fd:
        source = fd.read()
    tree = ast.parse(source)
    try:
        _pytest.assertion.rewrite.rewrite_asserts(tree)
    except TypeError as e:
        _pytest.assertion.rewrite.rewrite_asserts(tree, source)

    co = compile(tree, __file__, 'exec', dont_inherit=True)
    ns = {}
    exec(co, ns)

    cphd_con = ns['CphdConsistency'].from_file(config.cphd_or_xml, config.schema, config.signal_data)
    cphd_con.check()
    failures = cphd_con.failures()
    cphd_con.print_result(fail_detail=config.verbose >= 1,
                          include_passed_asserts=config.verbose >= 2,
                          include_passed_checks=config.verbose >= 3,
                          skip_detail=config.verbose >= 4)

    return bool(failures)


if __name__ == "__main__":     # pragma: no cover
    import sys
    sys.exit(int(main()))
