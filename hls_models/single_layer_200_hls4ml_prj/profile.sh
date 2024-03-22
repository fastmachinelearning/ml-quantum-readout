#!/bin/bash

if [ $# == 2 ]; then

    XML_HLS_REPORT_FILE=$1
    XML_SYN_REPORT_FILE=$2

    # HLS
    HLS_TARGET_CLOCK_PERIOD=`xmllint --xpath '/profile/UserAssignments/TargetClockPeriod/text()' $XML_HLS_REPORT_FILE`
    #echo "HLS: target clock period: $HLS_TARGET_CLOCK_PERIOD"

    HLS_ESTIMATED_CLOCK_PERIOD=`xmllint --xpath '/profile/PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod/text()' $XML_HLS_REPORT_FILE`
    #echo "HLS: estimated clock period: $HLS_ESTIMATED_CLOCK_PERIOD"

    HLS_ESTIMATED_BRAM=`xmllint --xpath '/profile/AreaEstimates/Resources/BRAM_18K/text()' $XML_HLS_REPORT_FILE`
    HLS_AVAILABLE_BRAM=`xmllint --xpath '/profile/AreaEstimates/AvailableResources/BRAM_18K/text()' $XML_HLS_REPORT_FILE`
    HLS_ESTIMATED_BRAM_PER=`echo "scale=4; (100*$HLS_ESTIMATED_BRAM)/$HLS_AVAILABLE_BRAM" | bc -l`
    #echo "HLS: estimated BRAMs: $HLS_ESTIMATED_BRAM ($HLS_ESTIMATED_BRAM_PER%)"
    #echo "HLS: available BRAMs: $HLS_AVAILABLE_BRAM"

    HLS_ESTIMATED_DSP=`xmllint --xpath '/profile/AreaEstimates/Resources/DSP48E/text()' $XML_HLS_REPORT_FILE`
    HLS_AVAILABLE_DSP=`xmllint --xpath '/profile/AreaEstimates/AvailableResources/DSP48E/text()' $XML_HLS_REPORT_FILE`
    HLS_ESTIMATED_DSP_PER=`echo "scale=4; (100*$HLS_ESTIMATED_DSP)/$HLS_AVAILABLE_DSP" | bc -l`
    #echo "HLS: estimated DSPs: $HLS_ESTIMATED_DSP ($ESTIMATED_DSP_PER%)"
    #echo "HLS: available DSPs: $HLS_AVAILABLE_DSP"

    HLS_ESTIMATED_FF=`xmllint --xpath '/profile/AreaEstimates/Resources/FF/text()' $XML_HLS_REPORT_FILE`
    HLS_AVAILABLE_FF=`xmllint --xpath '/profile/AreaEstimates/AvailableResources/FF/text()' $XML_HLS_REPORT_FILE`
    HLS_ESTIMATED_FF_PER=`echo "scale=4; (100*$HLS_ESTIMATED_FF)/$HLS_AVAILABLE_FF" | bc -l`
    #echo "HLS: estimated FFs: $HLS_ESTIMATED_FF ($HLS_ESTIMATED_FF_PER%)"
    #echo "HLS: available FFs: $HLS_SAVAILABLE_FF"

    HLS_ESTIMATED_LUT=`xmllint --xpath '/profile/AreaEstimates/Resources/LUT/text()' $XML_HLS_REPORT_FILE`
    HLS_AVAILABLE_LUT=`xmllint --xpath '/profile/AreaEstimates/AvailableResources/LUT/text()' $XML_HLS_REPORT_FILE`
    HLS_ESTIMATED_LUT_PER=`echo "scale=4; (100*$HLS_ESTIMATED_LUT)/$HLS_AVAILABLE_LUT" | bc -l`
    #echo "HLS: estimated LUTs: $HLS_ESTIMATED_LUT ($HLS_ESTIMATED_LUT_PER%)"
    #echo "HLS: available LUTs: $HLS_AVAILABLE_LUT"

    # SYN
    SYN_TARGET_CLOCK_PERIOD=`xmllint --xpath '/profile/TimingReport/TargetClockPeriod/text()' $XML_SYN_REPORT_FILE`
    #echo "SYN: target clock period: $SYN_TARGET_CLOCK_PERIOD"

    SYN_ESTIMATED_CLOCK_PERIOD=`xmllint --xpath '/profile/TimingReport/AchievedClockPeriod/text()' $XML_SYN_REPORT_FILE`
    #echo "SYN: estimated clock period: $SYN_ESTIMATED_CLOCK_PERIOD"

    SYN_ESTIMATED_BRAM=`xmllint --xpath '/profile/AreaReport/Resources/BRAM/text()' $XML_SYN_REPORT_FILE`
    SYN_AVAILABLE_BRAM=`xmllint --xpath '/profile/AreaReport/AvailableResources/BRAM/text()' $XML_SYN_REPORT_FILE`
    SYN_ESTIMATED_BRAM_PER=`echo "scale=4; (100*$SYN_ESTIMATED_BRAM)/$SYN_AVAILABLE_BRAM" | bc -l`
    #echo "SYN: estimated BRAMs: $SYN_ESTIMATED_BRAM ($SYN_ESTIMATED_BRAM_PER%)"
    #echo "SYN: available BRAMs: $SYN_AVAILABLE_BRAM"

    SYN_ESTIMATED_DSP=`xmllint --xpath '/profile/AreaReport/Resources/DSP/text()' $XML_SYN_REPORT_FILE`
    SYN_AVAILABLE_DSP=`xmllint --xpath '/profile/AreaReport/AvailableResources/DSP/text()' $XML_SYN_REPORT_FILE`
    SYN_ESTIMATED_DSP_PER=`echo "scale=4; (100*$SYN_ESTIMATED_DSP)/$SYN_AVAILABLE_DSP" | bc -l`
    #echo "SYN: estimated DSPs: $SYN_ESTIMATED_DSP ($SYN_ESTIMATED_DSP_PER%)"
    #echo "SYN: available DSPs: $SYN_AVAILABLE_DSP"

    SYN_ESTIMATED_FF=`xmllint --xpath '/profile/AreaReport/Resources/FF/text()' $XML_SYN_REPORT_FILE`
    SYN_AVAILABLE_FF=`xmllint --xpath '/profile/AreaReport/AvailableResources/FF/text()' $XML_SYN_REPORT_FILE`
    SYN_ESTIMATED_FF_PER=`echo "scale=4; (100*$SYN_ESTIMATED_FF)/$SYN_AVAILABLE_FF" | bc -l`
    #echo "SYN: estimated FFs: $SYN_ESTIMATED_FF ($SYN_ESTIMATED_FF_PER%)"
    #echo "SYN: available FFs: $SYN_AVAILABLE_FF"

    SYN_ESTIMATED_LUT=`xmllint --xpath '/profile/AreaReport/Resources/LUT/text()' $XML_SYN_REPORT_FILE`
    SYN_AVAILABLE_LUT=`xmllint --xpath '/profile/AreaReport/AvailableResources/LUT/text()' $XML_SYN_REPORT_FILE`
    SYN_ESTIMATED_LUT_PER=`echo "scale=4; (100*$SYN_ESTIMATED_LUT)/$SYN_AVAILABLE_LUT" | bc -l`
    #echo "SYN: estimated LUTs: $SYN_ESTIMATED_LUT ($SYN_ESTIMATED_LUT_PER%)"
    #echo "SYN: available LUTs: $SYN_AVAILABLE_LUT"

    # Header
    printf "%-5s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | \n" \
        "" \
        "Target CLK" "Est. CLK" \
        "BRAM" "BRAM (%)" "Total BRAM" \
        "DSP" "DSP (%)" "Total DSP" \
        "FF" "FF (%)" "Total FF" \
        "LUT" "LUT (%)" "Total LUT" \

    # Values
    printf "%-5s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | \n" \
        "HLS" \
        "$HLS_TARGET_CLOCK_PERIOD" "$HLS_ESTIMATED_CLOCK_PERIOD" \
        "$HLS_ESTIMATED_BRAM" "$HLS_ESTIMATED_BRAM_PER" "$HLS_AVAILABLE_BRAM" \
        "$HLS_ESTIMATED_DSP" "$HLS_ESTIMATED_DSP_PER" "$HLS_AVAILABLE_DSP" \
        "$HLS_ESTIMATED_FF" "$HLS_ESTIMATED_FF_PER" "$HLS_AVAILABLE_FF" \
        "$HLS_ESTIMATED_LUT" "$HLS_ESTIMATED_LUT_PER" "$HLS_AVAILABLE_LUT"
    printf "%-5s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | \n" \
        "SYN" \
        "$SYN_TARGET_CLOCK_PERIOD" "$SYN_ESTIMATED_CLOCK_PERIOD" \
        "$SYN_ESTIMATED_BRAM" "$SYN_ESTIMATED_BRAM_PER" "$SYN_AVAILABLE_BRAM" \
        "$SYN_ESTIMATED_DSP" "$SYN_ESTIMATED_DSP_PER" "$SYN_AVAILABLE_DSP" \
        "$SYN_ESTIMATED_FF" "$SYN_ESTIMATED_FF_PER" "$SYN_AVAILABLE_FF" \
        "$SYN_ESTIMATED_LUT" "$SYN_ESTIMATED_LUT_PER" "$SYN_AVAILABLE_LUT"


else
    echo "ERROR: you must pass HLS and SYN report files"
fi


