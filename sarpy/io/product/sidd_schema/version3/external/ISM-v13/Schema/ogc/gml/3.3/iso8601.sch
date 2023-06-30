<?xml version="1.0" encoding="UTF-8"?>
<sch:schema xmlns:sch="http://purl.oclc.org/dsdl/schematron">
    <sch:ns prefix="fn" uri="http://www.w3.org/2005/xpath-functions"/>
    <sch:ns prefix="gml" uri="http://www.opengis.net/gml/3.2"/>
    <sch:ns prefix="gmlxbt" uri="http://www.opengis.net/gml/3.3/xbt"/>
    <sch:pattern>
    	<!-- Replace {nodesOfTypeTimePositionUnion} with an Xpath expression to all nodes with a content model of gmlxbt:TimePositionUnit -->
        <sch:rule context="{nodesOfTypeTimePositionUnion}">
        	<!-- These tests are only possible with Xpath 2.0 support, so this is what we use here -->
            <sch:assert test="not(fn:matches(.,'^-?0{4}')">
                Year 0000 is not a valid year.
            </sch:assert>
            <sch:assert test="not(fn:matches(.,'^-?[0-9]{4}-0{3}'))">
                Calendar day 000 is not a valid value, the first day of the year is 001.
            </sch:assert>
            <!-- more to be added -->
        </sch:rule>
    </sch:pattern>
</sch:schema>