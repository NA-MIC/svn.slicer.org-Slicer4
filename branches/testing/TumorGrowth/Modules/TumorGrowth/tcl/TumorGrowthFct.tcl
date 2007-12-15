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
    set GUI  [$::slicer3::Application GetModuleGUIByName "TumorGrowth"]
    set NODE [$GUI  GetNode]
    set SCENE [$NODE GetScene]
    # Kilian: How do you check for zero ! 
    set SCAN1_NODE [$SCENE GetNodeByID [$NODE GetScan1_Ref]]
    set SCAN1_SEGMENT_NODE [$SCENE GetNodeByID [$NODE GetScan1_SegmentRef]]
    set SCAN2_NODE [$SCENE GetNodeByID [$NODE GetScan2_Ref]]
    
    vtkImageData OUTPUT
    HistogramNormalization_FCT [$SCAN1_NODE GetImageData] [$SCAN1_SEGMENT_NODE GetImageData] [$SCAN2_NODE GetImageData] OUTPUT
        
        # Attach to current tree
    set VolumeOutputNode [vtkMRMLScalarVolumeNode New]
    $VolumeOutputNode SetName "TG_scan2_reg_norm" 
        $VolumeOutputNode  SetAndObserveImageData OUTPUT
        [$NODE GetScene] AddNode $VolumeOutputNode
    
    $NODE SetScan2_greg_normRef $VolumeOutputNode

        # Delete 
        $VolumeOutputNode Delete
    OUTPUT Delete

    }

    proc HistogramNormalization_FCT {SCAN1 SCAN1_SEGMENT SCAN2 OUTPUT} {
    puts "Match intensities of Scan2 to Scan1 with Type: $TumorGrowth(Step5,HistogramType)" 
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
    foreach ID "1 2" {
       vtkImageMathematics HighIntensityRegion
          HighIntensityRegion SetInput1 [TUMOR_INSIDE GetOutput] 
          HighIntensityRegion SetInput2 ${SCAN$ID}
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
    set NormFactor [expr  double($TumorGrowth(Scan1,ROI_SUM_INTENS)) / double($TumorGrowth(Scan2,ROI_SUM_INTENS))]  
    puts "Intensity Normalization Factor:  $NormFactor"

    TumorGrowth(Scan2,ROISuperSampleNormalized)
    vtkImageMathematics TumorGrowth(Scan2,ROISuperSampleNormalized)
            TumorGrowth(Scan2,ROISuperSampleNormalized) SetInput1  $SCAN2 
            TumorGrowth(Scan2,ROISuperSampleNormalized) SetOperationToMultiplyByK 
         TumorGrowth(Scan2,ROISuperSampleNormalized) SetConstantK $NormFactor
    TumorGrowth(Scan2,ROISuperSampleNormalized) Update

    $OUTPUT DeepCopy [TumorGrowth(Scan2,ROISuperSampleNormalized) GetOutput]
    TumorGrowth(Scan2,ROISuperSampleNormalized) Delete
    }

    proc Scan2ToScan1Registration_GUI { TYPE } {puts "Has to be implemented" }
    proc IntensityThresholding_GUI { SCAN_ID } {puts "Has to be implemented" }
 }
 
