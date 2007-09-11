package require csv

#----------------------------------------------------------------------------------------------------
#--- label is the term in the sourceTermSet; targetTermSet contains the translation 
#----------------------------------------------------------------------------------------------------
proc QueryAtlasVocabularyInit { term sourceTermSet targetTermSet } {

    set ::QA(controlledVocabulary) "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/controlledVocabulary.csv"
    set ::QA(braininfoSynonyms) "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/NN2002-2-synonyms.csv"
    set ::QA(braininfoURI) "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/NN2002-3-BrainInfoURI.csv"
}


#----------------------------------------------------------------------------------------------------
#--- checks to see that the naming system is recognized
#----------------------------------------------------------------------------------------------------
proc QueryAtlasValidSystemCheck { termSet } {

    #--- what systems are valid?
    set candidates { FreeSurfer
        freesurfer
        BIRN_String
        birn_string
        BIRN_ID
        birn_id
        BIRN_URI
        birn_uri
        NN_ID
        NN_id
        NN_String
        NN_string
        UMLS_CID
        umls_cid }

    foreach c $candidates {
        if { $termSet == $c } {
            return 1
        } 
    }

    return 0
    

}


#----------------------------------------------------------------------------------------------------
#--- returns 1 if entry is mapped to its parent structure
#----------------------------------------------------------------------------------------------------
proc QueryAtlasMappedToBIRNLexParentCheck {  } {

}


#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasParseNeuroNamesSynonyms { } {
    #--- start fresh
    if { [ info exists ::QA(Synonyms) ] } {
        unset -nocomplain ::QA(Synonyms)
    }
    set ::QA(Synonyms) ""

    if { [catch "package require csv"] } {
        puts "Can't parse controlled vocabulary without package csv"
        return 
    }

    #--- The controlled vocabulary must be a CSV file
    set synonyms "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/NN2002-2-synonyms.csv"

    #--- get number of columns each entry should have
    set fp [ open $synonyms r ]
    gets $fp line
    set sline [ ::csv::split $line ]
    set numCols [ llength $sline ]
    close $fp
    
    set fp [ open $synonyms r ]
    #--- parse the file into a list of lists called $::QA(Synonyms)
    while { ! [eof $fp ] } {
        gets $fp line
        set sline [ ::csv::split $line ]
        set len [ llength $sline ]
        #--- if the line is the wrong length, blow it off
        if { $len == $numCols } {
            lappend ::QA(Synonyms) $sline
        } 
    }
    close $fp

}


#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasParseBrainInfoURIs { } {
    #--- start fresh
    if { [ info exists ::QA(BrainInfoURIs) ] } {
        unset -nocomplain ::QA(BrainInfoURIs)
    }
    set ::QA(BrainInfoURIs) ""

    if { [catch "package require csv"] } {
        puts "Can't parse controlled vocabulary without package csv"
        return 
    }

    #--- The controlled vocabulary must be a CSV file
    set uris "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/NN2002-3-BrainInfoURI.csv"

    #--- get number of columns each entry should have
    set fp [ open $uris r ]
    gets $fp line
    set sline [ ::csv::split $line ]
    set numCols [ llength $sline ]
    close $fp
    
    set fp [ open $uris r ]
    #--- parse the file into a list of lists called $::QA(Synonyms)
    while { ! [eof $fp ] } {
        gets $fp line
        set sline [ ::csv::split $line ]
        set len [ llength $sline ]
        #--- if the line is the wrong length, blow it off
        if { $len == $numCols } {
            lappend ::QA(BrainInfoURIs) $sline
        } 
    }
    close $fp

}

#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasParseControlledVocabulary { } {

    
    #--- start fresh
    if { [ info exists ::QA(CV) ] } {
        unset -nocomplain ::QA(CV)
    }
    set ::QA(CV) ""

    if { [catch "package require csv"] } {
        puts "Can't parse controlled vocabulary without package csv"
        return 
    }

    #--- The controlled vocabulary must be a CSV file
    set controlledVocabulary "$::env(SLICER_HOME)/../Slicer3/Modules/QueryAtlas/Resources/controlledVocabulary_new.csv"

    #--- get number of columns each entry should have
    set fp [ open $controlledVocabulary r ]
    gets $fp line
    set sline [ ::csv::split $line ]
    set numCols [ llength $sline ]
    close $fp
    
    set fp [ open $controlledVocabulary r ]
    #--- parse the file into a list of lists called $::QA(CV)
    while { ! [eof $fp ] } {
        gets $fp line
        set sline [ ::csv::split $line ]
        set len [ llength $sline ]
        #--- if the line is the wrong length, blow it off
        if { $len == $numCols } {
            lappend ::QA(CV) $sline
        } 
    }
    close $fp
}

#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasGetBrainInfoURI { term } {
    #--- if there's no controlled vocabulary parsed out, return.
    if { ![ info exists ::QA(BrainInfoURIs) ] } {
        return ""
    }
    if { $::QA(BrainInfoURIs) == "" } {
        return ""
    }

    set NNID [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
    puts "NNID = $NNID"

    #--- FIND the columns in the controlled vocabulary
    #--- that map to target and source TermSets
    set targetColName "BrainInfo URL"
    set targetColNum -1
    set line [ lindex $::QA(BrainInfoURIs) 0 ]
    set len [ llength $line ]
    for { set i 0 } { $i < $len } { incr i } {
        set col [ lindex $line $i ]
       if { $col == $targetColName } {
            set targetColNum $i
        }
    }
    if {$targetColNum > 0 } {
        #--- now march thru ::QA(BrainInfoURIs) down the source Column to find term
        set numRows [ llength $::QA(BrainInfoURIs) ]
        for { set i 0 } { $i < $numRows } { incr i } {
            set row [ lindex $::QA(BrainInfoURIs) $i ]
            set getT [ lindex $row 0 ]
            if { $getT == $NNID } {
                set targetT [ lindex $row $targetColNum ]
                return $targetT
            }
        }
    }

    #--- well, nothing in the table.
    return ""

}


#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasGetSynonyms { term } {

    #--- if there's no controlled vocabulary parsed out, return.
    if { ![ info exists ::QA(Synonyms) ] } {
        return ""
    }
    if { $::QA(Synonyms) == "" } {
        return ""
    }
    
    set NNterm [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]
    puts "NNterm = $NNterm"

    #--- FIND the columns in the controlled vocabulary
    #--- that map to target and source TermSets
    set targetColName "Hierarchy Lookup"
    set target1ColNum -1
    set line [ lindex $::QA(Synonyms) 0 ]
    set len [ llength $line ]
    for { set i 0 } { $i < $len } { incr i } {
        set col [ lindex $line $i ]
       if { $col == $targetColName } {
            set target1ColNum $i
        }
    }
    set targetColName "Species"
    set target2ColNum -1
    set line [ lindex $::QA(Synonyms) 0 ]
    set len [ llength $line ]
    for { set i 0 } { $i < $len } { incr i } {
        set col [ lindex $line $i ]
       if { $col == $targetColName } {
            set target2ColNum $i
        }
    }
    
    set targetColName "Synonym"
    set target3ColNum -1
    set line [ lindex $::QA(Synonyms) 0 ]
    set len [ llength $line ]
    for { set i 0 } { $i < $len } { incr i } {
        set col [ lindex $line $i ]
       if { $col == $targetColName } {
            set target3ColNum $i
        }
    }

    set synonyms ""
    if { ($target1ColNum > 0) && ($target2ColNum > 0) && ($target3ColNum > 0) } {
        #--- now march thru ::QA(Synonyms) down the
        #--- Hierarchy Lookup column, checking the species column
        #--- and find all synonyms
        set numRows [ llength $::QA(Synonyms) ]
        for { set i 0 } { $i < $numRows } { incr i } {
            set row [ lindex $::QA(Synonyms) $i ]
            #--- get the Heirarchy look up term in this row
            set termT [ lindex $row $target1ColNum ]
            #--- get the species entry in this row
            set speciesT [lindex $row $target2ColNum ]
            #--- if the Hierarchy Lookup term matches the NNterm
            #--- and the term applies to humans, add the synonym
            if { $termT == $NNterm && $speciesT == "human" } {
                set syn [ lindex $row $target3ColNum ]
                puts "$syn"
                lappend targetTerms $syn
            }
        }
    }

    #--- return the list of lists, each is a synonym
    return $synonyms


}


#----------------------------------------------------------------------------------------------------
#---
#----------------------------------------------------------------------------------------------------
proc QueryAtlasMapTerm { term sourceTermSet targetTermSet } {

    #--- if there's no controlled vocabulary parsed out, return.
    if { ![ info exists ::QA(CV) ] } {
        puts "Can't map term: no controlled vocabulary exists."
        return ""
    }
    if { $::QA(CV) == "" } {
        puts "Can't map term: no controlled vocabulary exists."
        return ""
    }

    #--- make sure sets of terms are valid
    set check [ QueryAtlasValidSystemCheck $sourceTermSet ]
    if { $check == 0 } {
        puts "QueryAtlasVocabularyMapper: $sourceTermSet is not a recognized set of terms"
        return ""
    }
    set check [ QueryAtlasValidSystemCheck $targetTermSet ]
    if { $check == 0 } {
        puts "QueryAtlasVocabularyMapper: $targetTermSet is not a recognized set of terms"
        return ""
    }

    #-- if we're just grabbing the freesurfer term, condition a little
    #-- to get rid of ugly underscores, dashes, etc.
    if { $sourceTermSet == "FreeSurfer" && $targetTermSet == "FreeSurfer" } {
        regsub -all -- "-" $term " " term
        regsub -all "ctx" $term "Cortex" term
        regsub -all "rh" $term "Right" term
        regsub -all "lh" $term "Left" term
        return $term
    }

    #--- FIND the columns in the controlled vocabulary
    #--- that map to target and source TermSets
    set sourceCol -1
    set targetCol -1
    set line [ lindex $::QA(CV) 0 ]
    set len [ llength $line ]
    for { set i 0 } { $i < $len } { incr i } {
        set col [ lindex $line $i ]
        if { $col == $sourceTermSet } {
            set sourceCol $i
        } elseif { $col == $targetTermSet } {
            set targetCol $i
        }
    }
    if { ($sourceCol > 0) && ($targetCol > 0) } {
        #--- now march thru ::QA(CV) down the source Column to find term
        set numRows [ llength $::QA(CV) ]
        for { set i 0 } { $i < $numRows } { incr i } {
            set row [ lindex $::QA(CV) $i ]
            set sourceT [ lindex $row $sourceCol ]
            if { $sourceT == $term } {
                set targetT [ lindex $row $targetCol ]
                return $targetT
            }
        }
    }

    #--- well, nothing in the table.
    return ""
}

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
proc QueryAtlasUpdateSynonymsMenu { term } {

        QueryAtlasGetSynonyms $term
        set m [ [ $::slicer3::QueryAtlasGUI GetSynonymsMenuButton ] GetMenu ]

        #--- clear menu
        $m DeleteAllItems

        #--- build menu
        set len [ llength $::QA(Synonyms) ]
        for { set i 0 } { $i < $len } { incr i } {
        #--- add new menuitems
            set item [ lindex $::QA(Synonyms) $i ]
           $m AddRadioButton $item
        }
        $m AddSeparator
        $m AddCommand "close"
}


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
proc QueryAtlasPopulateOntologyInformation { term infoType } {

    if { $infoType == "local" } {
        #--- BIRN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $term
        
        #--- BIRN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_ID" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] SetValue $val
        }
        #--- NN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] SetValue $val
        }
        #--- NN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] SetValue $val
        }
        #--- UMLS
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "UMLS_CID" ]                
        set curval  [ [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] SetValue $val
        }
    } elseif { $infoType == "BIRNLex" } {
        #---
        #--- local String
        set val [ QueryAtlasMapTerm $term "BIRN_String" "FreeSurfer" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $val
        
        #--- BIRN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_ID" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] SetValue $val
        }
        #--- NN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] SetValue $val
        }
        #--- NN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] SetValue $val
        }
        #--- UMLS
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "UMLS_CID" ]                
        set curval  [ [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] SetValue $val
        }


    } elseif { $infoType == "BIRNID" } {
        #--- local String
        set val [ QueryAtlasMapTerm $term "BIRN_String" "FreeSurfer" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $val        
        
        #--- BIRN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] SetValue $val
        }
        #--- NN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] SetValue $val
        }
        #--- NN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] SetValue $val
        }
        #--- UMLS
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "UMLS_CID" ]                
        set curval  [ [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] SetValue $val
        }
        #--- update synonyms

    } elseif { $infoType == "NN" } {
        #--- local String
        set val [ QueryAtlasMapTerm $term "BIRN_String" "FreeSurfer" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $val        

        #--- BIRN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] SetValue $val
        }
        #--- BIRN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_ID" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] SetValue $val
        }
        #--- NN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] SetValue $val
        }
        #--- UMLS
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "UMLS_CID" ]                
        set curval  [ [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] SetValue $val
        }
        #--- update synonyms

    } elseif { $infoType == "NNID" } {
        #--- local String
        set val [ QueryAtlasMapTerm $term "BIRN_String" "FreeSurfer" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $val
        
        #--- BIRN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] SetValue $val
        }
        #--- BIRN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_ID" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] SetValue $val
        }
        #--- NN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] SetValue $val
        }
        #--- UMLS
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "UMLS_CID" ]                
        set curval  [ [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetUMLSCIDEntry] SetValue $val
        }
        #--- update synonyms

    } elseif { $infoType == "UMLS" } {
        #--- local String
        set val [ QueryAtlasMapTerm $term "BIRN_String" "FreeSurfer" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetLocalSearchTermEntry] SetValue $val
        }
        #--- update synonyms
        QueryAtlasUpdateSynonymsMenu $val

        #--- BIRN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexEntry] SetValue $val
        }
        #--- BIRN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "BIRN_ID" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetBIRNLexIDEntry] SetValue $val
        }
        #--- NN String
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_String" ]        
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesEntry] SetValue $val
        }
        #--- NN ID
        set val [ QueryAtlasMapTerm $term "FreeSurfer" "NN_ID" ]
        set curval  [ [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] GetValue ]
        if { $val != $curval } {
            [$::slicer3::QueryAtlasGUI GetNeuroNamesIDEntry] SetValue $val
        }
        #--- update synonyms
    }
}




#----------------------------------------------------------------------------------------------------
#--- returns synonyms from NeuroNames resources
#----------------------------------------------------------------------------------------------------
proc QueryAtlasGetNeuroNamesURI { term sourceTermSet } {

}


#----------------------------------------------------------------------------------------------------
#--- returns synonyms from NeuroNames resources
#----------------------------------------------------------------------------------------------------
proc QueryAtlasGetNeuroNamesSynonyms { term sourceTermSet } {

}

