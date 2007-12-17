package require Itcl

#########################################################
#
if {0} { ;# comment

  This is function is executed by TumorGrowth  

# TODO : 

}
#
#########################################################

#
# namespace procs
#

namespace eval TumorGrowthTcl {

    proc Print { BLUB } {
      set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
      set NODE [$GUI  GetNode]
      set VOLUME_NODE [[$NODE GetScene] GetNodeByID [$NODE GetScan1_Ref]]
      puts "======================= $BLUB"
      puts "[[$VOLUME_NODE GetImageData] Print]"
      puts "=======================iiii"

    }

    proc HistogramNormalization_GUI { } {
      puts "HistogramNormalization_GUI Start"
      set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
      set NODE [$GUI  GetNode]
      set SCENE [$NODE GetScene]

      set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_Ref]]
      set SCAN1_SEGMENT_NODE [$SCENE GetNodeByID [$NODE GetScan1_SegmentRef]]
      puts "++++fsdffsdfsd"
      set SCAN2_NODE [$SCENE GetNodeByID [[$NODE GetScan2_GlobalRef] GetID]]
      puts "----fsdffsdfsd"
      if { $SCAN1_NODE == "" || $SCAN1_SEGMENT_NODE == "" || $SCAN2_NODE == "" } { 
      puts "Error: Not all nodes of the pipeline are defined  $SCAN1_NODE - $SCAN1_SEGMENT_NODE - $SCAN2_NODE" 
      return
      }
    
      catch {OUTPUT Delete}
      vtkImageData OUTPUT
      HistogramNormalization_FCT [$SCAN1_NODE GetImageData] [$SCAN1_SEGMENT_NODE GetImageData] [$SCAN2_NODE GetImageData] OUTPUT
          
          # Attach to current tree
      set VolumeOutputNode [vtkMRMLScalarVolumeNode New]
      $VolumeOutputNode SetName "TG_scan2_reg_norm" 
          $VolumeOutputNode  SetAndObserveImageData OUTPUT
          [$NODE GetScene] AddNode $VolumeOutputNode
      
      $NODE SetScan2_NormedRef $VolumeOutputNode
      
      # Delete 
      $VolumeOutputNode Delete
      OUTPUT Delete

    }

    proc HistogramNormalization_FCT {SCAN1 SCAN1_SEGMENT SCAN2 OUTPUT} {
      puts "Match intensities of Scan2 to Scan1" 
      # Just use pixels that are clearly inside the tumor => generate label map of inside tumor 
      # Kilian -we deviate here from slicer2 there SCAN1_SEGMENT =  [TumorGrowth(Scan1,PreSegment) GetOutput]
      
      catch {TUMOR_DIST Delete}
      vtkImageKilianDistanceTransform TUMOR_DIST 
        TUMOR_DIST   SetInput $SCAN1_SEGMENT
        TUMOR_DIST   SetAlgorithmToSaito
        TUMOR_DIST   SignedDistanceMapOn
        TUMOR_DIST   SetObjectValue  10 
        TUMOR_DIST   SetZeroBoundaryInside
        TUMOR_DIST   DistanceTransform
        TUMOR_DIST   SetMaximumDistance 100 
        TUMOR_DIST   ConsiderAnisotropyOff
      TUMOR_DIST   Update
      
      vtkImageAccumulate HistTemp
          HistTemp SetInput [TUMOR_DIST  GetOutput]
      HistTemp Update
          
      set Max [lindex [HistTemp GetMax] 0]
      HistTemp Delete
      
      catch {TUMOR_INSIDE Delete}
      vtkImageThreshold TUMOR_INSIDE
        TUMOR_INSIDE SetOutputScalarType [$SCAN1_SEGMENT GetScalarType] 
        TUMOR_INSIDE SetInput [TUMOR_DIST GetOutput]
        TUMOR_INSIDE ThresholdByUpper [expr $Max*0.5]
        TUMOR_INSIDE SetInValue 1
        TUMOR_INSIDE SetOutValue 0
      TUMOR_INSIDE Update
      
      # Calculate the mean for scan 1 and Scan 2 (we leave out the factor of voxels bc it does not matter latter
      catch { HighIntensityRegion Delete }
      foreach ID "1 2" {
         vtkImageMathematics HighIntensityRegion
        HighIntensityRegion SetInput1 [TUMOR_INSIDE GetOutput] 
        if {$ID > 1} { HighIntensityRegion SetInput2 $SCAN2
        } else { HighIntensityRegion SetInput2 $SCAN1 }
            HighIntensityRegion SetOperationToMultiply
         HighIntensityRegion Update 
         
         vtkImageSumOverVoxels SUM
              SUM SetInput [HighIntensityRegion GetOutput] 
         SUM Update
      
         set TumorGrowth(Scan${ID},ROI_SUM_INTENS) [SUM GetVoxelSum ]
         SUM Delete
         HighIntensityRegion Delete
      } 
          TUMOR_DIST Delete
          TUMOR_INSIDE Delete
      
          # Multiply scan2 with the factor that normalizes both mean  
    if {$TumorGrowth(Scan2,ROI_SUM_INTENS) == 0 } { 
        set NormFactor 0.0 
    } else {
        set NormFactor [expr  double($TumorGrowth(Scan1,ROI_SUM_INTENS)) / double($TumorGrowth(Scan2,ROI_SUM_INTENS))]  
    }
      puts "Intensity Normalization Factor:  $NormFactor"
      
    catch {TumorGrowth(Scan2,ROISuperSampleNormalized) Delete}
      vtkImageMathematics TumorGrowth(Scan2,ROISuperSampleNormalized)
              TumorGrowth(Scan2,ROISuperSampleNormalized) SetInput1  $SCAN2 
              TumorGrowth(Scan2,ROISuperSampleNormalized) SetOperationToMultiplyByK 
           TumorGrowth(Scan2,ROISuperSampleNormalized) SetConstantK $NormFactor
      TumorGrowth(Scan2,ROISuperSampleNormalized) Update
      
      $OUTPUT DeepCopy [TumorGrowth(Scan2,ROISuperSampleNormalized) GetOutput]
      TumorGrowth(Scan2,ROISuperSampleNormalized) Delete
    }

    proc Scan2ToScan1Registration_GUI { TYPE } {
        puts "=============================================="
    puts "TumorGrowthScan2ToScan1Registration $TYPE Start" 

        # -------------------------------------
        # Define Interfrace Parameters 
        # -------------------------------------
    set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
    set NODE [$GUI  GetNode]
    if {$NODE == ""} {return $NODE}

    set SCENE [$NODE GetScene]
    set LOGIC [$GUI GetLogic]

        # -------------------------------------
        # Initialize Registration 
        # -------------------------------------
    set MaxNum 50
    if { "$TYPE" == "Global" } { 
        # Kilian: How do you check for zero ! 
        set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_Ref]]
        set SCAN2_NODE [$SCENE GetNodeByID [$NODE GetScan2_Ref]]
    } else {
        set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_SuperSampleRef]]
        set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan2_SuperSampleRef]]
    }
    if {$SCAN1_NODE == "" || $SCAN2_NODE == ""} { return }
    set VOL1 [$SCAN1_NODE GetImageData]
    set VOL2 [$SCAN2_NODE GetImageData]
    
       ::TumorGrowthReg::DeleteTransformAG [$LOGIC Get${TYPE}Transform]
        set TRANSFORM [$LOGIC Create${TYPE}Transform]
         
    set OUTPUT_VOL [vtkImageData New]

        # -------------------------------------
        # Register 
        # -------------------------------------
    if { 0 } {
        # Set it automatcally later 
        set ScanOrder IS
        if {[::TumorGrowthReg::RegistrationAG $VOL1 $ScanOrder $VOL2 $ScanOrder 1 0 0 $MaxNum mono 3 $TRANSFORM ] == 0 }  {
        puts "Error:  TumorGrowthScan2ToScan1Registration: $TYPE  could not perform registration"
        return
        }
   
        ::TumorGrowthReg::TumorGrowthAGResample $VOL2 $VOL1 $TRANSFORM $OUTPUT_VOL  
         
        # ::TumorGrowthReg::WriteTransformationAG $TRANSFORM [$NODE GetWorkingDir] 
        ::TumorGrowthReg::WriteTransformationAG $TRANSFORM ~/temp
        catch { exec mv [$NODE GetWorkingDir]/LinearRegistration.txt [$NODE GetWorkingDir]/${TYPE}LinearRegistration.txt }
 
    } else {
        puts "Debugging - jump over registration"
        $OUTPUT_VOL  DeepCopy $VOL1
    }
  
    # -------------------------------------
        # Transfere output 
        # ------------------------------------- 
    set OUTPUT_NODE [$SCENE GetNodeByID [$NODE GetScan2_${TYPE}Ref]]
    if {$OUTPUT_NODE != "" } {  [$GUI GetMRMLScene] RemoveNode $OUTPUT_NODE }

    set OUTPUT_NODE [vtkMRMLScalarVolumeNode New]
       
    $OUTPUT_NODE SetAndObserveImageData $OUTPUT_VOL 
     $OUTPUT_NODE SetName "TG_scan2_global" 
    $SCENE AddNode $OUTPUT_NODE
    
    $NODE SetScan2_${TYPE}Ref $OUTPUT_NODE
        # puts "[$NODE GetScan2_GlobalRef] -- [$SCENE GetNodeByID [[$NODE GetScan2_GlobalRef] GetID]] --  [$SCENE GetNodeByID [$OUTPUT_NODE GetID]]"
    # -------------------------------------
        # Clean up 
        # -------------------------------------
    $OUTPUT_VOL Delete
    # Kilian: Why can I not delete the node here ? 
    # $OUTPUT_NODE Delete

        # puts "[$NODE GetScan2_GlobalRef] -- [$SCENE GetNodeByID [[$NODE GetScan2_GlobalRef] GetID]] --  [$SCENE GetNodeByID [$OUTPUT_NODE GetID]]"
    puts "TumorGrowthScan2ToScan1Registration $TYPE End" 
        puts "=============================================="
    }


    proc IntensityThresholding_GUI { SCAN_ID } {
    # -------------------------------------
        # Define Interfrace Parameters 
        # -------------------------------------
    set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
    set NODE [$GUI  GetNode]
    if {$NODE == ""} {return $NODE}

    set SCENE [$NODE GetScene]
    set LOGIC [$GUI GetLogic]

    # -------------------------------------
        # Initialize Thresholding
        # -------------------------------------
    if { $SCAN_ID == 1} { 
        set SCAN_NODE [$SCENE GetNodeByID [$NODE GetScan1_SuperSampleRef]]
    } else {
        set SCAN_NODE [$SCENE GetNodeByID [$NODE GetScan2_NormedRef]]
    }
    if {$SCAN_NODE == ""} { return }

    set INPUT_VOL [$SCAN_NODE GetImageData]         
    set OUTPUT_VOL [vtkImageData New]

    # -------------------------------------
        # Run Thresholding and return results
        # -------------------------------------

    IntensityThresholding_Fct $INPUT_VOL [$NODE GetSegmentThreshold] $OUTPUT_VOL 

    set VolumeOutputNode [vtkMRMLScalarVolumeNode New]
    $VolumeOutputNode SetName "TG_scan${SCAN_ID}_Thr" 
    $VolumeOutputNode  SetAndObserveImageData $OUTPUT_VOL
    $SCENE AddNode $VolumeOutputNode
    $NODE  SetScan2_NormedRef $VolumeOutputNode
       
    $VolumeOutputNode Delete
    $OUTPUT_VOL Delete
    }

   # $TumorGrowth(Scan1,Threshold)  
    proc IntensityThresholding_Fct { INPUT THRESH OUTPUT} {
      # Eveyrthing outside below threhold is set to threshold

      catch {ROIThreshold Delete}
      vtkImageThreshold ROIThreshold
        ROIThreshold ThresholdByUpper $THRESH
        ROIThreshold SetInput  $INPUT
        ROIThreshold ReplaceInOff  
        ROIThreshold SetOutValue $THRESH
        ROIThreshold SetOutputScalarTypeToShort
      ROIThreshold Update
      $OUTPUT DeepCopy [ROIThreshold GetOutput]
      ROIThreshold Delete


    }

    proc AnalysisIntensity_GUI { } {
    // set SCAN1 [TumorGrowth(Scan1,ROIThreshold) GetOutput]
    // set SCAN2 [TumorGrowth(Scan2,ROIThreshold) GetOutput]
    // set SEGMENT [TumorGrowth(Scan1,Segment) GetOutput]
     $SCAN1 $SEGMENT $SCAN2 2
    }

    proc AnalysisIntensity_Fct { Scan1Data Scan1Segment Scan2Data } {
    puts "Start Tumor Growth Analysis" 
    # Subtract consecutive scans from each other
    catch {TumorGrowth(FinalSubtract)  Delete } 
      vtkImageMathematics TumorGrowth(FinalSubtract)
      TumorGrowth(FinalSubtract) SetInput1 $Scan2Data 
      TumorGrowth(FinalSubtract) SetInput2 $Scan1Data 
      TumorGrowth(FinalSubtract) SetOperationToSubtract  
    TumorGrowth(FinalSubtract) Update
    # do a little bit of smoothing 

    vtkImageMedian3D TumorGrowth(FinalSubtractSmooth)
       TumorGrowth(FinalSubtractSmooth) SetInput [TumorGrowth(FinalSubtract) GetOutput]
       TumorGrowth(FinalSubtractSmooth) SetKernelSize 3 3 3
       TumorGrowth(FinalSubtractSmooth) ReleaseDataFlagOff
    TumorGrowth(FinalSubtractSmooth) Update

    set TumorGrowth(FinalAnalysisThreshold) [TumorGrowthAnalysis_ComputeThreshold [TumorGrowth(FinalSubtractSmooth) GetOutput] $Scan1Segment ]

    # Change label 
    catch {TumorGrowth(FinalROI) Delete } 
    vtkImageThreshold TumorGrowth(FinalROI) 
      TumorGrowth(FinalROI)  SetInput $Scan1Segment 
      TumorGrowth(FinalROI)  SetInValue 1
      TumorGrowth(FinalROI)  SetOutValue 0
      TumorGrowth(FinalROI)  ThresholdByLower 0 
       TumorGrowth(FinalROI)  SetOutputScalarTypeToShort
    TumorGrowth(FinalROI) Update

    catch {TumorGrowth(FinalMultiply)  Delete } 
    vtkImageMathematics TumorGrowth(FinalMultiply)
       TumorGrowth(FinalMultiply) SetInput1 [TumorGrowth(FinalROI)       GetOutput] 
       TumorGrowth(FinalMultiply) SetInput2 [TumorGrowth(FinalSubtractSmooth)  GetOutput] 
       TumorGrowth(FinalMultiply) SetOperationToMultiply  
    TumorGrowth(FinalMultiply) Update

    catch {TumorGrowth(FinalAnalysis) Delete } 
    vtkImageThreshold TumorGrowth(FinalAnalysis) 
      TumorGrowth(FinalAnalysis) SetInput [TumorGrowth(FinalMultiply) GetOutput] 
      TumorGrowth(FinalAnalysis) ReplaceInOff
      TumorGrowth(FinalAnalysis) SetOutValue 0
      TumorGrowth(FinalAnalysis) ThresholdByUpper  $TumorGrowth(FinalAnalysisThreshold)
       TumorGrowth(FinalAnalysis) SetOutputScalarTypeToShort
    TumorGrowth(FinalAnalysis) Update

    catch {TumorGrowth(FinalROINegativeBin)  Delete } 
    vtkImageThreshold TumorGrowth(FinalROINegativeBin) 
      TumorGrowth(FinalROINegativeBin) SetInput [TumorGrowth(FinalMultiply) GetOutput] 
      TumorGrowth(FinalROINegativeBin) SetInValue -1
      TumorGrowth(FinalROINegativeBin) SetOutValue 0
      TumorGrowth(FinalROINegativeBin) ThresholdByLower  -$TumorGrowth(FinalAnalysisThreshold)
       TumorGrowth(FinalROINegativeBin) SetOutputScalarTypeToShort
    TumorGrowth(FinalROINegativeBin) Update

    # Initializing tumor growth prediction
    vtkImageThreshold TumorGrowth(FinalROIBin) 
      TumorGrowth(FinalROIBin)  SetInput [TumorGrowth(FinalAnalysis) GetOutput] 
      TumorGrowth(FinalROIBin)  SetInValue 1
      TumorGrowth(FinalROIBin)  SetOutValue 0
      TumorGrowth(FinalROIBin)  ThresholdByUpper 1 
       TumorGrowth(FinalROIBin)  SetOutputScalarTypeToShort
    TumorGrowth(FinalROIBin) Update

    vtkImageMathematics TumorGrowth(FinalROIBinReal) 
      TumorGrowth(FinalROIBinReal)  SetInput 0 [TumorGrowth(FinalROIBin) GetOutput] 
      TumorGrowth(FinalROIBinReal)  SetInput 1 [TumorGrowth(FinalROINegativeBin) GetOutput] 
      TumorGrowth(FinalROIBinReal)  SetOperationToAdd 
    TumorGrowth(FinalROIBinReal) Update

    # Remove small islands - slowes it down a lot 
    #vtkImageIslandFilter TumorGrowth(FinalROIBinRealIsl) 
    #  TumorGrowth(FinalROIBinRealIsl)  SetIslandMinSize 2 
    #  TumorGrowth(FinalROIBinRealIsl) SetInput [TumorGrowth(FinalROIBinReal) GetOutput]
    #  TumorGrowth(FinalROIBinRealIsl) SetNeighborhoodDim3D 
    #  TumorGrowth(FinalROIBinRealIsl) SetIslandOutputLabel 0
    #  TumorGrowth(FinalROIBinRealIsl) SetPrintInformation 0
    # TumorGrowth(FinalROIBinRealIsl) Update

   
    vtkImageSumOverVoxels TumorGrowth(FinalROITotal) 
    TumorGrowth(FinalROITotal)  SetInput [TumorGrowth(FinalROIBinReal) GetOutput]
    return $TumorGrowth(FinalAnalysisThreshold)
}

proc TumorGrowthMeassuringGrowth_Fct { } {
    global TumorGrowth

    # Not yet that far in the step structure    
    if { [catch {TumorGrowth(FinalAnalysis) ThresholdByUpper  $TumorGrowth(FinalAnalysisThreshold)}] } { return ""}
    TumorGrowth(FinalAnalysis) Update

    TumorGrowth(FinalROINegativeBin) ThresholdByLower  -$TumorGrowth(FinalAnalysisThreshold)
    TumorGrowth(FinalROINegativeBin) Update
   
    TumorGrowth(FinalROITotal)  Update
    
    set SUM [TumorGrowth(FinalROITotal) GetVoxelSum ]
    # if {$SUM < 0} { set SUM 0}
    
    return "[format %.3f [expr $SUM*$TumorGrowth(ROI,SuperSampleVoxelVolume)]] mm^3 ([expr int($SUM*$TumorGrowth(ROI,RatioNewOldSpacing))] Voxels)"
}


}
 
