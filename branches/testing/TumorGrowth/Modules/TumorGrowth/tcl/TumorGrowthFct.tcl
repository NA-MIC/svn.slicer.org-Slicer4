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

    proc HistogramNormalization_GUI { } {
      puts "HistogramNormalization_GUI Start"
      # -------------------------------------
      # Define Interface Parameters 
      # -------------------------------------
      set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
      set NODE [$GUI  GetNode]
      if {$NODE == ""} {return }

      set SCENE [$NODE GetScene]
      set LOGIC [$GUI GetLogic]

      # -------------------------------------
      # Initialize Thresholding
      # -------------------------------------
      set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_SuperSampleRef]]
      set SCAN1_SEGMENT_NODE [$SCENE GetNodeByID [$NODE GetScan1_SegmentRef]]
      set SCAN2_NODE [$SCENE GetNodeByID [$NODE GetScan2_SuperSampleRef]]
      if { $SCAN1_NODE == "" || $SCAN1_SEGMENT_NODE == "" || $SCAN2_NODE == "" } { 
         puts "Error: Not all nodes of the pipeline are defined  $SCAN1_NODE - $SCAN1_SEGMENT_NODE - $SCAN2_NODE" 
         return
      }
    
      set OUTPUT [ vtkImageData New]
      HistogramNormalization_FCT [$SCAN1_NODE GetImageData] [$SCAN1_SEGMENT_NODE GetImageData] [$SCAN2_NODE GetImageData] $OUTPUT
          
      # -------------------------------------
      # Transfere output 
      # -------------------------------------
      set OUTPUT_NODE [$SCENE GetNodeByID [$NODE GetScan2_NormedRef]]
      if {$OUTPUT_NODE != "" } { $SCENE RemoveNode $OUTPUT_NODE }

      set OUTPUT_NODE [$LOGIC CreateVolumeNode  $SCAN1_NODE "TG_scan2_reg_norm" ]
      $OUTPUT_NODE SetAndObserveImageData $OUTPUT
      $NODE SetScan2_NormedRef [$OUTPUT_NODE GetID]

      # -------------------------------------
      # Clean up  
      # -------------------------------------
      $OUTPUT Delete

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
        set SCAN2_NODE [$SCENE GetNodeByID [$NODE GetScan2_SuperSampleRef]]
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
    if { 1 } {
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
        puts "Debugging - jump over registration $VOL1"
        $OUTPUT_VOL  DeepCopy $VOL1
    }
  
        # -------------------------------------
        # Transfere output 
        # -------------------------------------
        puts "========================= "
        set OUTPUT_NODE [$SCENE GetNodeByID [$NODE GetScan2_${TYPE}Ref]]
        if {$OUTPUT_NODE != "" } {  [$GUI GetMRMLScene] RemoveNode $OUTPUT_NODE }

        set OUTPUT_NODE [$LOGIC CreateVolumeNode  $SCAN1_NODE  "TG_scan2_${TYPE}" ]
        $OUTPUT_NODE SetAndObserveImageData $OUTPUT_VOL 
    
        $NODE SetScan2_${TYPE}Ref [$OUTPUT_NODE GetID]

        # -------------------------------------
        # Clean up 
        # -------------------------------------
        $OUTPUT_VOL Delete

        puts "TumorGrowthScan2ToScan1Registration $TYPE End"
        puts "[$NODE GetScene]    [$GUI GetMRMLScene] "   
        puts "=============================================="
    }


    proc IntensityThresholding_GUI { SCAN_ID } {
        # -------------------------------------
        # Define Interface Parameters 
        # -------------------------------------
        set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
        set NODE [$GUI  GetNode]
        if {$NODE == ""} {return }

        set SCENE [$NODE GetScene]
        set LOGIC [$GUI GetLogic]

        # -------------------------------------
        # Initialize Thresholding
        # -------------------------------------
        set OUTPUT_NODE [$SCENE GetNodeByID [$NODE GetScan${SCAN_ID}_ThreshRef]]
        if {$OUTPUT_NODE != "" } { $SCENE RemoveNode $OUTPUT_NODE }

        if { $SCAN_ID == 1} { 
          set SCAN_NODE [$SCENE GetNodeByID [$NODE GetScan1_SuperSampleRef]]
      if {$SCAN_NODE == ""} { 
          puts "ERROR: IntensityThresholding_GUI: No Scan1_SuperSampleRef defined !"
          return 0
      }
        } else {
          set SCAN_NODE [$SCENE GetNodeByID [$NODE GetScan2_NormedRef]]
      if {$SCAN_NODE == ""} { 
          puts "ERROR: IntensityThresholding_GUI: No Scan2_NormedRef defined !"
          return 0
      }
        }

        set INPUT_VOL [$SCAN_NODE GetImageData]         
        set OUTPUT_VOL [vtkImageData New]

        # -------------------------------------
        # Run Thresholding and return results
        # -------------------------------------
    puts "Threshold [$NODE GetSegmentThreshold]" 
        IntensityThresholding_Fct $INPUT_VOL [$NODE GetSegmentThreshold] $OUTPUT_VOL 

    set OUTPUT_NODE [$LOGIC CreateVolumeNode  $SCAN_NODE "TG_scan${SCAN_ID}_Thr" ]
        $OUTPUT_NODE SetAndObserveImageData $OUTPUT_VOL
        $NODE SetScan${SCAN_ID}_ThreshRef [$OUTPUT_NODE GetID]

        $OUTPUT_VOL Delete
        return  1
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
    puts "=============================================="
        puts "AnalysisIntensity Start" 

        # -------------------------------------
        # Define Interfrace Parameters 
        # -------------------------------------
        set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
        set NODE [$GUI  GetNode]
        if {$NODE == ""} {return 0}

        set SCENE [$NODE GetScene]
        set LOGIC [$GUI GetLogic]

        # -------------------------------------
        # Initialize Analysis
        # -------------------------------------
    set OUTPUT_NODE [$SCENE GetNodeByID [$NODE GetAnalysis_Ref]]
        if {$OUTPUT_NODE != "" } { $SCENE RemoveNode $OUTPUT_NODE }

    set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_ThreshRef]]
    set SEGM_NODE  [$SCENE GetNodeByID [$NODE GetScan1_SegmentRef]]
    set SCAN2_NODE [$SCENE GetNodeByID [$NODE GetScan2_ThreshRef]]

        if {$SCAN1_NODE == "" || $SEGM_NODE == "" || $SCAN2_NODE == "" } { 
        puts "ERROR:AnalysisIntensity_GUI: Incomplete Input" 
        return 0
    }
    set AnalysisFinal          [$LOGIC CreateAnalysis_Final]
    set AnalysisROINegativeBin [$LOGIC CreateAnalysis_ROINegativeBin]
    set AnalysisROIPositiveBin [$LOGIC CreateAnalysis_ROIPositiveBin]
    set AnalysisROIBinReal     [$LOGIC CreateAnalysis_ROIBinReal]
    set AnalysisROITotal       [$LOGIC CreateAnalysis_ROITotal]
    set AnalysisSensitivity    [$NODE  GetAnalysis_Sensitivity]

        # -------------------------------------
        # Run Analysis and Save output
        # -------------------------------------

        set result "[AnalysisIntensity_Fct [$SCAN1_NODE GetImageData] [$SEGM_NODE GetImageData] [$SCAN1_NODE GetImageData] $AnalysisSensitivity \
                              $AnalysisFinal $AnalysisROINegativeBin $AnalysisROIPositiveBin $AnalysisROIBinReal $AnalysisROITotal ]"

    $NODE  SetAnalysis_Sensitivity $AnalysisSensitivity
    $LOGIC SetAnalysis_Mean [lindex $result 0]
    $LOGIC SetAnalysis_Variance [lindex $result 1]
    $LOGIC SetAnalysis_Threshold [lindex $result 2]

        set VOLUMES_GUI  [$::slicer3::Application GetModuleGUIByName "Volumes"]
    set VOLUMES_LOGIC [$VOLUMES_GUI GetLogic]

    set OUTPUT_NODE [$VOLUMES_LOGIC CreateLabelVolume $SCENE $SEGM_NODE "TG_Analysis"]
        $OUTPUT_NODE SetAndObserveImageData [$AnalysisROIBinReal GetOutput] 
        $NODE SetAnalysis_Ref [$OUTPUT_NODE GetID]
    return 1
    }

    proc AnalysisIntensity_Fct { Scan1Data Scan1Segment Scan2Data AnalysisSensitivity AnalysisFinal AnalysisROINegativeBin  AnalysisROIPositiveBin AnalysisROIBinReal AnalysisROITotal } {
       
       # Subtract consecutive scans from each other
       catch {TumorGrowth(FinalSubtract)  Delete } 
         vtkImageMathematics TumorGrowth(FinalSubtract)
         TumorGrowth(FinalSubtract) SetInput1 $Scan2Data 
         TumorGrowth(FinalSubtract) SetInput2 $Scan1Data 
         TumorGrowth(FinalSubtract) SetOperationToSubtract  
       TumorGrowth(FinalSubtract) Update

       # do a little bit of smoothing 
       catch {TumorGrowth(FinalSubtractSmooth) Delete}
       vtkImageMedian3D TumorGrowth(FinalSubtractSmooth)
        TumorGrowth(FinalSubtractSmooth) SetInput [TumorGrowth(FinalSubtract) GetOutput]
        TumorGrowth(FinalSubtractSmooth) SetKernelSize 3 3 3
        TumorGrowth(FinalSubtractSmooth) ReleaseDataFlagOff
       TumorGrowth(FinalSubtractSmooth) Update

       set result [Analysis_ComputeThreshold [TumorGrowth(FinalSubtractSmooth) GetOutput] $Scan1Segment $AnalysisSensitivity]
       set FinalThreshold [lindex $result 2]

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

        # puts "AnalysisFinal $AnalysisFinal "
         $AnalysisFinal SetInput [TumorGrowth(FinalMultiply) GetOutput] 
         $AnalysisFinal ReplaceInOff
         $AnalysisFinal SetOutValue 0
         $AnalysisFinal ThresholdByUpper  $FinalThreshold
         $AnalysisFinal SetOutputScalarTypeToShort
       $AnalysisFinal Update

       # vtkImageThreshold TumorGrowth(FinalROINegativeBin) 

         $AnalysisROINegativeBin SetInput [TumorGrowth(FinalMultiply) GetOutput] 
         $AnalysisROINegativeBin SetInValue -1
         $AnalysisROINegativeBin SetOutValue 0
         $AnalysisROINegativeBin ThresholdByLower  -$FinalThreshold
         $AnalysisROINegativeBin SetOutputScalarTypeToShort
       $AnalysisROINegativeBin Update

       # Initializing tumor growth prediction
       # catch { TumorGrowth(FinalROIBin) Delete}
         $AnalysisROIPositiveBin  SetInput [$AnalysisFinal GetOutput] 
         $AnalysisROIPositiveBin  SetInValue 1
         $AnalysisROIPositiveBin  SetOutValue 0
         $AnalysisROIPositiveBin  ThresholdByUpper  $FinalThreshold
         $AnalysisROIPositiveBin  SetOutputScalarTypeToShort
       $AnalysisROIPositiveBin Update

       # vtkImageMathematics TumorGrowth(FinalROIBinReal) 
         $AnalysisROIBinReal  SetInput 0 [$AnalysisROIPositiveBin GetOutput] 
         $AnalysisROIBinReal  SetInput 1 [$AnalysisROINegativeBin GetOutput] 
         $AnalysisROIBinReal  SetOperationToAdd 
       $AnalysisROIBinReal Update

       # vtkImageSumOverVoxels TumorGrowth(FinalROITotal) 
         $AnalysisROITotal  SetInput [$AnalysisROIBinReal GetOutput]

    return "$result"
  }   

  proc Analysis_UpdateThreshold_GUI { } {
        # -------------------------------------
        # Define Interface Parameters 
        # -------------------------------------
        set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
        set NODE [$GUI  GetNode]
        if {$NODE == ""} {return $NODE}

        set LOGIC [$GUI GetLogic]

        # -------------------------------------
        # Initialize 
        # -------------------------------------
    set AnalysisMean           [$LOGIC GetAnalysis_Mean ]
    set AnalysisVariance       [$LOGIC GetAnalysis_Variance ]
    set AnalysisSensitivity    [$NODE  GetAnalysis_Sensitivity]

        # -------------------------------------
        # Compute and return results 
        # -------------------------------------
    set ThresholdValue [Analysis_InverseStandardCumulativeDistribution $AnalysisSensitivity  $AnalysisMean $AnalysisVariance]      
        if { $ThresholdValue < 0.0 } { set ThresholdValue 0.0 }
    $LOGIC SetAnalysis_Threshold $ThresholdValue
    } 


    # Gaussian functions 
    # The inverse of the erf (or Error Function http://en.wikipedia.org/wiki/Error_function ) 
    # This is an approximation of the error function to the 20th order  - see http://functions.wolfram.com/GammaBetaErf/InverseErf/06/01/0001/
    
    # InverseErf[z] == (Sqrt[Pi]/2) (z + (Pi z^3)/12 + (7 Pi^2 z^5)/480 + (127 Pi^3 z^7)/40320 + (4369 Pi^4 z^9)/5806080 + (34807 Pi^5 z^11)/182476800 + (20036983 Pi^6 z^13)/398529331200 + (2280356863 Pi^7 z^15)/167382319104000 + (49020204823 Pi^8 z^17)/ 13007997370368000 + (65967241200001 Pi^9 z^19)/62282291409321984000) + O[z^20]
    
    proc  Analysis_InverseErf { z } {
    
       # Values are computed via matlab
       # sqrt(pi)/2
       set tcl_precision_old  $::tcl_precision
       set ::tcl_precision 17
       set Norm  0.88622692545276
    
       # 1
       set c(1) 1.0
    
       # pi/12
       set c(3)  0.26179938779915
    
       # 7*pi^2 /480
       set c(5)  0.14393173084922
    
       # 127* pi^3 /40320 
       set c(7)  0.09766361950392
    
       # 4369/5806080 * pi^4 
       set c(9) 0.07329907936638
    
       # 34807 /182476800 * pi^5 
       set c(11) 0.05837250087858
    
       # 20036983 /398529331200 *pi^6
       set c(13) 0.04833606317018
    
       # 2280356863 /167382319104000 * pi^7
       set c(15) 0.04114739494052
    
       # 49020204823/ 13007997370368000 * pi^8 
       set c(17) 0.03575721309236
    
       # 65967241200001/62282291409321984000 * pi^9 
       set c(19) 0.0315727633198
    
       set result 0.0 
       set sqr_z [expr $z*$z]
    
       for {set i 1} {$i < 20 } {incr i 2} {
           set result [expr $result + $c($i)*$z]
           set z [expr $z*$sqr_z]
       }  
    
       set result  [expr $result*$Norm]
       set ::tcl_precision $tcl_precision_old  
       return $result
    
    }

  # The result is n so that prob = N(x <= n ; \mu ,\sigma^2) 
  proc Analysis_InverseStandardCumulativeDistribution { prob mu sigma } {
    if {($prob < 0) ||  $prob > 1} {return [expr sqrt(-1)]}

    set InvErf [Analysis_InverseErf [expr 2*$prob -1 ]]
    return [expr $mu + $sigma *sqrt(2)* $InvErf]
  }

  # Compute threshold based on Gaussian noise in segmented region 
  proc Analysis_ComputeThreshold {Scan1SubScan2 Scan1Segment AnalysisSensitivity} {
    # compute Gaussian pdf for noise
    vtkImageMathematics compThrNoise 
       compThrNoise  SetInput1 $Scan1SubScan2
       compThrNoise  SetOperationToAbsoluteValue
    compThrNoise  Update

    # Make sure that Segmentation is binarized 
    vtkImageThreshold compThrROI 
     compThrROI SetInput $Scan1Segment 
     compThrROI SetInValue 1
     compThrROI SetOutValue 0
     compThrROI  ThresholdByUpper 1 
      compThrROI  SetOutputScalarTypeToShort
    compThrROI  Update

    # -----------------------------------------------------
    # Compute Mean     
    vtkImageMathematics compThrROINoise 
       compThrROINoise  SetInput1 [compThrROI  GetOutput]
       compThrROINoise  SetInput2 [compThrNoise  GetOutput]
       compThrROINoise  SetOperationToMultiply 
    compThrROINoise  Update

    # Compute Nominator 
    vtkImageSumOverVoxels compThrSum
       compThrSum SetInput [compThrROINoise GetOutput]
    compThrSum Update
    set IntensityDiffTotal   [compThrSum GetVoxelSum]

    # Compute Denominator 
    compThrSum SetInput [compThrROI GetOutput]
    compThrSum Update
    set SizeOfROI [compThrSum GetVoxelSum]

    if { $SizeOfROI } {
    set MeanNoise [expr  double($IntensityDiffTotal) / double($SizeOfROI)]
    } else {
    set MeanNoise 0
    }

    # -----------------------------------------------------
    # Compute Variance

    # Subtract mean
    vtkImageMathematics compThrROINoiseSubMean 
       compThrROINoiseSubMean  SetInput1 [compThrROINoise  GetOutput]
       compThrROINoiseSubMean  SetOperationToAddConstant 
       compThrROINoiseSubMean  SetConstantC -$MeanNoise
    compThrROINoiseSubMean  Update

    # Only consider region of interest
    vtkImageMathematics compThrVarianceInput 
       compThrVarianceInput   SetInput1 [compThrROI  GetOutput]
       compThrVarianceInput   SetInput2 [compThrROINoiseSubMean  GetOutput]
       compThrVarianceInput  SetOperationToMultiply 
    compThrVarianceInput  Update
 
    # Now square the input 
    vtkImageMathematics compThrVarianceInputSqr 
       compThrVarianceInputSqr   SetInput1 [compThrVarianceInput  GetOutput]
       compThrVarianceInputSqr  SetOperationToSquare 
    compThrVarianceInputSqr  Update
 
    # Define Variance 
    compThrSum SetInput [compThrVarianceInputSqr GetOutput]
    compThrSum Update
    set Nominator [compThrSum GetVoxelSum]

    set Variance [expr  double($Nominator) / (double($SizeOfROI) - 1.0)]
    set SqrtVariance [expr sqrt($Variance)]
    # ----------------------------------------
    # Clean Up
    compThrVarianceInputSqr Delete
    compThrVarianceInput Delete
    compThrROINoiseSubMean Delete 
    compThrSum      Delete
    compThrROINoise Delete
    compThrROI      Delete
    compThrNoise    Delete

    # ----------------------------------------
    # Compute Threshold
    # the threshold value that excludes 
    
    set ThresholdValue [Analysis_InverseStandardCumulativeDistribution $AnalysisSensitivity  $MeanNoise $SqrtVariance]

    if { $ThresholdValue < 0.0 } { set ThresholdValue 0.0 }

    puts "ComputeThreshold -- Mean: $MeanNoise Variance: $Variance Threshold: $ThresholdValue"
    
    return "$MeanNoise $SqrtVariance $ThresholdValue"
  }
}
 
