#include <iostream>
#include <vector>
#include <string>

#include "vtkTumorGrowthLogic.h"
#include "vtkMRMLTumorGrowthNode.h"

#include "TumorGrowthCommandLineCLP.h"
#include "vtkNRRDReader.h"
#include "vtkNRRDWriter.h" 
#include "vtkSlicerApplication.h"
#include "vtkKWTkUtilities.h"
#include <vtksys/stl/string>
#include "vtkImageIslandFilter.h"
#include "vtkMRMLVolumeNode.h"
#include "vtkMatrix4x4.h"
#include "vtkITKImageWriter.h"
#include "vtkGeneralTransform.h"
#include "vtkImageResample.h"
#include "vtkImageConstantPad.h"

#include <vtksys/SystemTools.hxx>

// Go to Slicer3-build/lib/Slicer3/Plugins/TumorGrowthCommandLine
// ./TumorGrowthCommandLine --sensitivity 0.5 --threshold 100,277 --roi_min 73,135,92 --roi_max 95,165,105 --intensity_analysis --deformable_analysis --scan1 /data/local/BrainScienceFoundation/Demo/07-INRIA/data/SILVA/2006-spgr.nhdr --scan2 /data/local/BrainScienceFoundation/Demo/07-INRIA/data/SILVA/2007-spgr-scan1.nhdr
//
//
// This is necessary to load in TumorGrowth package in TCL interp.
extern "C" int Slicerbasegui_Init(Tcl_Interp *interp);
extern "C" int Tumorgrowth_Init(Tcl_Interp *interp);
extern "C" int Vtkteem_Init(Tcl_Interp *interp);
extern "C" int Vtkitk_Init(Tcl_Interp *interp);

#define tgVtkCreateMacro(name,type) \
  name  = type::New(); \
  name##Tcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp, name)); 

#define tgVtkDefineMacro(name,type) \
  type *name; \
  std::string name##Tcl;\
  tgVtkCreateMacro(name,type); 

#define tgSetDataMacro(name,matrix)               \
 virtual int Set##name(const char *fileName) { \
   if (strcmp(fileName,"None")) { \
    tgVtkCreateMacro(this->name,vtkImageData); \
    this->matrix = vtkMatrix4x4::New(); \
    return tgReadVolume(fileName,this->name,this->matrix);    \
   } \
   this->name = NULL; \
   this->matrix = NULL; \
   return 0; \
 }

int tgReadVolume(const char *fileName, vtkImageData *outputData,  vtkMatrix4x4 *outputMatrix) {
  // Currently only works with nrrd files bc I do not know how to converd itk::ImageFileReader vtkImageData
  vtkNRRDReader *reader = vtkNRRDReader::New();
  reader->SetFileName(fileName);
  reader->Update();

  outputData->DeepCopy(reader->GetOutput());
  outputMatrix->DeepCopy(reader->GetRasToIjkMatrix());

  if (reader->GetReadStatus()) {
    cout << "ERROR: tgReadVolume: could not read " << fileName << endl;
    return 1;
  }
  reader->Delete();
  return 0;
}



class tgCMDLineStructure {
  public:

  tgCMDLineStructure(Tcl_Interp *init) {
    this->Scan1Data = NULL;
    this->Scan1DataTcl = "";
    this->Scan1Matrix = NULL;
    this->Scan2Data = NULL;
    this->Scan2DataTcl = "";
    this->Scan2Matrix = NULL;

    this->interp = init;
  } 

  ~tgCMDLineStructure() {
    if (this->Scan1Data) {
      Scan1Data->Delete();
      this->Scan1Data = NULL;
    }

    if (this->Scan1Matrix) {
      Scan1Matrix->Delete();
      this->Scan1Matrix = NULL;
    }
    
    if (this->Scan2Data) {
      Scan2Data->Delete();
      this->Scan2Data = NULL;
    }

    if (this->Scan2Matrix) {
      Scan2Matrix->Delete();
      this->Scan2Matrix = NULL;
    }


    this->interp = NULL;
  } 

  tgSetDataMacro(Scan1Data,Scan1Matrix);
  tgSetDataMacro(Scan2Data,Scan2Matrix);



  void SetWorkingDir(vtkKWApplication *app, const char* fileNameScan1) {
    this->WorkingDir = vtksys::SystemTools::GetFilenamePath(vtksys::SystemTools::CollapseFullPath(fileNameScan1)) + "-TGcmd";
    cout << "Setting working directory to " << this->WorkingDir << endl;
    char CMD[1024];
    sprintf(CMD,"file isdirectory %s",this->GetWorkingDir()); 
    if (!atoi(app->Script(CMD))) { 
      sprintf(CMD,"file mkdir %s",this->GetWorkingDir()); 
      app->Script(CMD); 
    } 
  }

  const char *GetWorkingDir () {return this->WorkingDir.c_str();}

  vtkImageData *Scan1Data; 
  std::string Scan1DataTcl;
  vtkMatrix4x4 *Scan1Matrix; 

  vtkImageData *Scan2Data; 
  std::string Scan2DataTcl;
  vtkMatrix4x4 *Scan2Matrix; 

  std::string WorkingDir;

  private:
  // int tgReadVolume(const char *fileName, vtkImageData *outputData,  vtkMatrix4x4 *outputMatrix); 

  Tcl_Interp *interp;

};

void tgWriteVolume(const char *fileName, vtkMatrix4x4 *export_matrix, vtkImageData *data) {
  vtkITKImageWriter *iwriter = vtkITKImageWriter::New(); 
  iwriter->SetInput(data); 
  iwriter->SetFileName(fileName);
  iwriter->SetRasToIJKMatrix(export_matrix);
  iwriter->SetUseCompression(1);
  iwriter->Write();
  iwriter->Delete();

 // Did not save spacing 
 //vtkNRRDWriter *iwriter = vtkNRRDWriter::New();
 // currently ignores spacing
 // iwriter->SetInput(Output);
 // iwriter->SetFileName(fileName);
 // iwriter->Write();
 // iwriter->Delete();
}


int tgSetSLICER_HOME(char** argv)  
{ 
  vtksys_stl::string slicerHome;
  if ( !vtksys::SystemTools::GetEnv("Slicer3_HOME", slicerHome) )
  {
    std::string programPath;
    std::string errorMessage;
    if ( !vtksys::SystemTools::FindProgramPath(argv[0], programPath, errorMessage) ) return 1;

    std::string homeEnv = "Slicer3_HOME=";
    homeEnv += vtksys::SystemTools::GetFilenamePath(programPath.c_str()) + "/../../../";
   
    // cout << "Set environment: " << homeEnv.c_str() << endl;
    vtkKWApplication::PutEnv(const_cast <char *> (homeEnv.c_str()));
  }
  return 0;
}

int tgRegisterAG(vtkKWApplication *app, Tcl_Interp *interp, vtkImageData* Target, std::string TargetTcl, vtkImageData* Source, std::string WorkingDir, std::string Output) {

  vtkImageResample *SourceRes = vtkImageResample::New();
      SourceRes->SetDimensionality(3);
      SourceRes->SetInterpolationModeToLinear();
      SourceRes->SetInput(Source); 
      SourceRes->SetBackgroundLevel(0);
      SourceRes->SetAxisOutputSpacing(0,Target->GetSpacing()[0]);
      SourceRes->SetAxisOutputSpacing(1,Target->GetSpacing()[1]);
      SourceRes->SetAxisOutputSpacing(2,Target->GetSpacing()[2]);
      SourceRes->SetOutputOrigin(Target->GetOrigin());
      SourceRes->ReleaseDataFlagOff();
   SourceRes->Update();

   vtkImageConstantPad *SourceResPad = vtkImageConstantPad::New();
      SourceResPad->SetInput(SourceRes->GetOutput());
      SourceResPad->SetOutputWholeExtent(Target->GetWholeExtent());
      SourceResPad->SetConstant(0);
   SourceResPad->Update();

   char *SourceResPadOutputTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,SourceResPad->GetOutput()));


  vtkGeneralTransform* Transform = vtkGeneralTransform::New();
  std::string TransformTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,Transform));

  std::string CMD = "::TumorGrowthReg::RegistrationAG " +  TargetTcl + " IS " + SourceResPadOutputTcl + " IS 1 0 0 50 mono 3 " + TransformTcl;
   
  if (!app->Script(CMD.c_str())) {
    cout << "Error:  Could not perform Registration";
    Transform->Delete();
    SourceResPad->Delete();
    SourceRes->Delete(); 
    return 1; 
  } 
   
  CMD = "::TumorGrowthReg::ResampleAG_GUI " + std::string(SourceResPadOutputTcl) + " " + TargetTcl + " " +  TransformTcl + " " + Output;
  app->Script(CMD.c_str());
   
  CMD = "::TumorGrowthReg::WriteTransformationAG "  +  TransformTcl + " " + WorkingDir;
  app->Script(CMD.c_str());

  Transform->Delete();
  SourceResPad->Delete();
  SourceRes->Delete(); 

  return 0; 
}

void  _Print(vtkImageData *DATA,::ostream& os)  {
  vtkIndent indent;
  // DATA->PrintSelf(os,indent.GetNextIndent());
  os << indent <<  "Origin " << DATA->GetOrigin()[0] << " "  <<  DATA->GetOrigin()[1] << " "  <<  DATA->GetOrigin()[2] << endl;
  os << indent <<  "Extent " << DATA->GetExtent()[0] << " "  <<  DATA->GetExtent()[1] << " "  <<  DATA->GetExtent()[2] << " " 
     << DATA->GetExtent()[3] << " "  <<  DATA->GetExtent()[4] << " "  <<  DATA->GetExtent()[5] << endl; 
  os << indent <<  "Spacing " << DATA->GetSpacing()[0] << " "  <<  DATA->GetSpacing()[1] << " "  <<  DATA->GetSpacing()[2] << endl;
}


int main(int argc, char** argv)
{
  //
  // parse arguments using the CLP system; this creates variables.
  PARSE_ARGS;

  try { 

    // -------------------------------------
    // Initialize TCL  Script
    // -------------------------------------
    Tcl_Interp *interp = vtkKWApplication::InitializeTcl(argc, argv, &cout);
    if (!interp)
    {
      cout << "Error: InitializeTcl failed" << endl;
      return EXIT_FAILURE; 
    }
  
    // This is necessary to load in TumorGrowth package in TCL interp.
    Tumorgrowth_Init(interp);
    Vtkteem_Init(interp);
    Vtkitk_Init(interp);

    // SLICER_HOME
    if (tgSetSLICER_HOME(argv)) {
      cout << "Error: Cannot find executable" << endl;
      return EXIT_FAILURE; 
    }

    // When I include the following line I get the leak message 
    // vtkSlicerApplication *app   = vtkSlicerApplication::GetInstance();
    vtkKWApplication *app   = vtkKWApplication::New();

    vtkTumorGrowthLogic  *logic = vtkTumorGrowthLogic::New();
    logic->SetModuleName("TumorGrowth");
    std::string logicTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,logic));
    logic->SourceAnalyzeTclScripts(app);
 
    // -------------------------------------
    // Load Parameters for pipeline 
    // -------------------------------------
    tgCMDLineStructure tg(interp);   
    // Create Working Directory 
    tg.SetWorkingDir(app,tgScan1.c_str()); 

    tgVtkDefineMacro(Scan2LocalNormalized,vtkImageData);
    tgVtkDefineMacro(Scan2Local,vtkImageData); 
    tgVtkDefineMacro(Scan1PreSegment,vtkImageThreshold); 
    tgVtkDefineMacro(Scan1Segment,vtkImageIslandFilter); 
    tgVtkDefineMacro(Scan1SegmentOutput,vtkImageData);
    tgVtkDefineMacro(Scan1SuperSample,vtkImageData); 
    tgVtkDefineMacro(Scan2SuperSample,vtkImageData); 
    tgVtkDefineMacro(Scan2Global,vtkImageData); 

    if (tg.SetScan1Data(tgScan1.c_str())) return EXIT_FAILURE;
    if (tg.SetScan2Data(tgScan2.c_str())) return EXIT_FAILURE;
    
    if (!tg.Scan1Data || !tg.Scan2Data ) {
      cout << "ERROR: --scan1 and --scan2 have to be defined" << endl;
      return EXIT_FAILURE; 
    }
       
    if ((tgROIMin.size() != 3) || (tgROIMax.size() != 3) ) {
      cout << "ERROR: --ROIMin or --ROIMax are not corretly defined!" << endl;
      return EXIT_FAILURE; 
    }

    std::string Scan1SuperSampleFileName = tg.WorkingDir + "/TG_scan1_SuperSampled.nhdr";
    std::string Scan2LocalNormalizedFileName = tg.WorkingDir + "/TG_scan2_norm.nhdr";
    std::string Scan1SegmentFileName = tg.WorkingDir + "/TG_scan1_Segment.nhdr";
   
  

    // Necessary for creating matrix with correct origin
    // 
    double *Spacing;
    double SuperSampleSpacing; 
    double SuperSampleVol;     
    double Scan1Vol;     
    double SuperSampleRatio;
    vtkMatrix4x4 *supersampleMatrix = vtkMatrix4x4::New(); 
    int ROIMin[3] = {tgROIMin[0], tgROIMin[1],  tgROIMin[2]};
    int ROIMax[3] = {tgROIMax[0], tgROIMax[1],  tgROIMax[2]};

    {
         Spacing =  tg.Scan1Data->GetSpacing();
             
         SuperSampleSpacing = logic->DefineSuperSampleSize(Spacing, ROIMin, ROIMax);
         SuperSampleVol     = SuperSampleSpacing*SuperSampleSpacing*SuperSampleSpacing;
         Scan1Vol           = (Spacing[0]*Spacing[1]*Spacing[2]);
         SuperSampleRatio   = SuperSampleVol/Scan1Vol;


         int *EXTENT = Scan1SuperSample->GetExtent();
         int dims[3] = {EXTENT[1] - EXTENT[0] + 1, EXTENT[3] - EXTENT[2] + 1,EXTENT[5] - EXTENT[4] + 1};
   
         double newIJKOrigin[4] = {ROIMin[0],ROIMin[1],ROIMin[2], 1.0 };
         double newRASOrigin[4];
         char ScanOrder[100];
   
         vtkMatrix4x4 *Scan1MatrixIJKToRAS = vtkMatrix4x4::New();
           Scan1MatrixIJKToRAS->DeepCopy(tg.Scan1Matrix);
           Scan1MatrixIJKToRAS->Invert();
           Scan1MatrixIJKToRAS->MultiplyPoint(newIJKOrigin,newRASOrigin);
           strcpy(ScanOrder, vtkMRMLVolumeNode::ComputeScanOrderFromIJKToRAS(Scan1MatrixIJKToRAS));
           
         Scan1MatrixIJKToRAS->Delete();
    
     double SuperSampleSpacingArray[3] = {SuperSampleSpacing,SuperSampleSpacing,SuperSampleSpacing};
         vtkMRMLVolumeNode::ComputeIJKToRASFromScanOrder(ScanOrder,SuperSampleSpacingArray,dims,1,supersampleMatrix);
         // vtkMRMLVolumeNode::ComputeIJKToRASFromScanOrder(ScanOrder,Scan1SuperSample->GetSpacing(),dims,1,supersampleMatrix);
         supersampleMatrix->SetElement(0,3,newRASOrigin[0]);
         supersampleMatrix->SetElement(1,3,newRASOrigin[1]);
         supersampleMatrix->SetElement(2,3,newRASOrigin[2]);
         supersampleMatrix->Invert();
    }

    // -------------------------------------
    // Run pipeline 
    // -------------------------------------
       


    if (tgDebugFlag) {
      // right now only set up for one AnalysisIntensity newtype
      vtkMatrix4x4* matrix = vtkMatrix4x4::New(); 
   
      // this is simply for debugging right now 
      std::string Dir = vtksys::SystemTools::GetFilenamePath(vtksys::SystemTools::CollapseFullPath(tgScan1.c_str())) + "-TG";
      std::string Scan1SuperSampleFileName     = Dir + "/scan1_ROI_SuperSample.nhdr"; 
      tgReadVolume(Scan1SuperSampleFileName.c_str(), Scan1SuperSample, matrix);

      std::string Scan1SegmentFileName         = Dir + "/scan1_ROI_SuperSample-Segment.nhdr";
      tgReadVolume(Scan1SegmentFileName.c_str(), Scan1SegmentOutput, matrix);

      std::string Scan2LocalFileName = Dir + "/scan2_RegGlobal_ROI_SuperSample_NORM_RegLocal.nhdr"; 
      tgReadVolume(Scan2LocalFileName.c_str(), Scan2Local, matrix);

      matrix->Delete();

    } else {
 
       // 
       // -----------GLOBAL REGISTRATION --------
       // 
       if (1) {
         // -------------------------------------
         cout << "=== Global Rigid Registration ===" << endl;
    
         if (tgRegisterAG(app, interp,  tg.Scan1Data, tg.Scan1DataTcl,  tg.Scan2Data , tg.GetWorkingDir(), Scan2GlobalTcl)) return EXIT_FAILURE;
   
         std::string CMD = "catch { exec mv " + tg.WorkingDir + "/LinearRegistration.txt " + tg.WorkingDir + "/GlobalLinearRegistration.txt }";
         app->Script(CMD.c_str());
   
         CMD = "catch { ::TumorGrowthReg::DeleteTransformAG }";
         app->Script(CMD.c_str());
   
         CMD = tg.WorkingDir + "/TG_scan2_Global.nhdr";
         tgWriteVolume(CMD.c_str(),tg.Scan1Matrix,Scan2Global);    
          
       } else {
         cout << "Debugging - jump over global registration" << endl;
         Scan2Global->DeepCopy(tg.Scan1Data);
       }
    
       // 
       // --------------- ROI --------------------
       // 

       if (1) {
         // -------------------------------------
         // Resample Scan 1
   
         cout << "=== Define ROI for each scan ===" << endl;
    
         if (logic->CreateSuperSampleFct(tg.Scan1Data,ROIMin, ROIMax, SuperSampleSpacing,Scan1SuperSample)) {
          cout << "ERROR: Could not super sample scan1 " << endl;
          return EXIT_FAILURE; 
         }
   
         //
         // Finally save results 
         // 
         tgWriteVolume(Scan1SuperSampleFileName.c_str(),supersampleMatrix,Scan1SuperSample);
   
         // -------------------------------------
         // Resample Scan2 
   
         if (logic->CreateSuperSampleFct(Scan2Global,ROIMin, ROIMax, SuperSampleSpacing,Scan2SuperSample)) {
          cout << "ERROR: Could not super sample scan1 " << endl;
          return EXIT_FAILURE; 
         }
          std::string NAME = tg.WorkingDir + "/TG_scan2_Global_SuperSampled.nhdr";
          tgWriteVolume(NAME.c_str(),supersampleMatrix, Scan2SuperSample);    
       } else {
          cout << "Debugging - jump over super sampling" << endl;      
       }
    
       // 
       // ------------- SEGMENTATION --------------------
       //
   
   
       if (1) {
         // -------------------------------------
         cout << "=== Segment Scan1 ===" << endl;
         int range[2] = {tgThreshold[0],tgThreshold[1]};
         vtkTumorGrowthLogic::DefinePreSegment(Scan1SuperSample,range,Scan1PreSegment);
         vtkTumorGrowthLogic::DefineSegment(Scan1PreSegment->GetOutput(),Scan1Segment);
   
         tgWriteVolume(Scan1SegmentFileName.c_str(),supersampleMatrix,Scan1Segment->GetOutput());
     Scan1SegmentOutput->DeepCopy(Scan1Segment->GetOutput());
       }

        // 
       // ------------- LOCAL REGISTRATION  --------------------
       //

       if (1) {
         // -------------------------------------
         cout << "=== Local Rigid Registration ===" << endl;
        
         if (tgRegisterAG( app, interp, Scan1SuperSample, Scan1SuperSampleTcl, Scan2SuperSample, tg.GetWorkingDir(), Scan2LocalTcl)) return EXIT_FAILURE;
         std::string CMD = "catch { exec mv " + tg.WorkingDir + "/LinearRegistration.txt " + tg.WorkingDir + "/LocalLinearRegistration.txt }";
         app->Script(CMD.c_str());
   
         CMD = "catch { ::TumorGrowthReg::DeleteTransformAG }";
         app->Script(CMD.c_str());

         std::string Scan2LocalFileName = tg.WorkingDir + "/TG_scan2_Local.nhdr"; 
         tgWriteVolume(Scan2LocalFileName.c_str(),supersampleMatrix,Scan2Local);
   
       } else {
        cout << "Debugging - jump over local registration" << endl;
        Scan2Local->DeepCopy(Scan1SuperSample);
       }
   
    } // End od Debug else .... 


    // 
    // ------------- NORMALIZE  --------------------
    //
    
    if (1) {
          // -------------------------------------
          cout << "=== Normalize Scan2 ===" << endl;
          std::string CMD = "::TumorGrowthTcl::HistogramNormalization_FCT " + Scan1SuperSampleTcl + " " + Scan1SegmentOutputTcl + " " 
                                                                            + Scan2LocalTcl + " " + Scan2LocalNormalizedTcl;
          app->Script(CMD.c_str()); 
          tgWriteVolume(Scan2LocalNormalizedFileName.c_str(),supersampleMatrix, Scan2LocalNormalized);
    }
   
    //
    // ------------- INTENSITY THRESHOLDING  --------------------
    // you do this to account that sometimes the area outside the tumor is more dark in one scan than the other



    // 
    // ------------- ANALYZE TYPE: INTENSITY  --------------------
    //
    tgVtkDefineMacro(Scan1Intensity,vtkImageData); 
    tgVtkDefineMacro(Scan2Intensity,vtkImageData); 
    double Analysis_Intensity_Growth = -1;


    if (tgIntensityAnalysisFlag) { 
      cout << "=== Intensity Based Analysis ===" << endl;

      std::string CMD = "return $TumorGrowthTcl::newIntensityAnalysis";
      int newIntensityAnalysis = atoi(app->Script(CMD.c_str()));

      char parameters[100];
      sprintf(parameters, " %f %d %d" ,tgSensitivity,tgThreshold[0],tgThreshold[1]);

      if (newIntensityAnalysis) { 
      // New Form without intensity thresholding
      CMD = "::TumorGrowthTcl::Analysis_Intensity_CMD " + logicTcl + " " + Scan1SuperSampleTcl + " " + Scan1SegmentOutputTcl + " " +  Scan2LocalNormalizedTcl + " " + parameters ;

      } else {
        if (1) {
         char ThreshString[1024];
         sprintf(ThreshString," %i %i ", tgThreshold[0], tgThreshold[1]);
         CMD = "::TumorGrowthTcl::IntensityThresholding_Fct " + Scan1SuperSampleTcl + " " + Scan1SuperSampleTcl + ThreshString + Scan1IntensityTcl;
         app->Script(CMD.c_str());
   
         std::string Scan1IntensityFileName = tg.WorkingDir + "/TG_scan1_Thr.nhdr";
         tgWriteVolume(Scan1IntensityFileName.c_str(),supersampleMatrix, Scan1Intensity);
   
         CMD = "::TumorGrowthTcl::IntensityThresholding_Fct " + Scan2LocalNormalizedTcl + " " + Scan1SuperSampleTcl + ThreshString + Scan2IntensityTcl;
         app->Script(CMD.c_str());
   
         std::string Scan2IntensityFileName = tg.WorkingDir + "/TG_scan2_Thr.nhdr";
         tgWriteVolume(Scan2IntensityFileName.c_str(),supersampleMatrix,Scan2Intensity);

      CMD = "::TumorGrowthTcl::Analysis_Intensity_CMD " + logicTcl + " " + Scan1IntensityTcl + " " + Scan1SegmentOutputTcl + " " + Scan2IntensityTcl + parameters;
    }


      }

      // ------------- ANALYSIS  --------------------
      app->Script(CMD.c_str());

      cout << "=========================" << endl;    
      Analysis_Intensity_Growth  = logic->MeassureGrowth(tgThreshold[0], tgThreshold[1]);
      CMD = tg.WorkingDir + "/TG_Analysis_Intensity.nhdr";
      tgWriteVolume(CMD.c_str(),supersampleMatrix,logic->GetAnalysis_Intensity_ROIBinReal());
      cout << "Analysis Intensity Growth: " <<  Analysis_Intensity_Growth  << " Super sample " << SuperSampleVol << endl;
      printf("Intensity Metric: %.3f mm^3 (%d Voxels)\n",  Analysis_Intensity_Growth *SuperSampleVol,int( Analysis_Intensity_Growth *SuperSampleRatio));

    } 

    // 
    // ------------- ANALYZE TYPE: DEFORMABLE  --------------------
    //
    double Analysis_SEGM_Growth = -1; 
    double Analysis_JACO_Growth = -1; 

    if (tgDeformableAnalysisFlag ) { 

      std::string SCAN1_TO_SCAN2_SEGM_NAME           = tg.WorkingDir + "/TG_Deformable_Scan1SegmentationAlignedToScan2.nhdr";
      std::string SCAN1_TO_SCAN2_DEFORM_NAME         = tg.WorkingDir + "/TG_Deformable_Deformation.mha";
      std::string SCAN1_TO_SCAN2_DEFORM_INVERSE_NAME = tg.WorkingDir + "/TG_Deformable_Deformation_Inverse.mha";
      std::string SCAN1_TO_SCAN2_RESAMPLED_NAME      = tg.WorkingDir + "/TG_Deformable_Scan1AlignedToScan2.nhdr";
      std::string ANALYSIS_SEGM_FILE                 = tg.WorkingDir + "/Analysis_Deformable_Sementation_Result.txt";    
      std::string ANALYSIS_JACOBIAN_FILE             = tg.WorkingDir + "/Analysis_Deformable_Jaccobian_Result.txt";  

      std::string CMD =  "::TumorGrowthTcl::Analysis_Deformable_Fct " + Scan1SuperSampleFileName + " " + Scan1SegmentFileName + " " + Scan2LocalNormalizedFileName + " "
                                                                  + SCAN1_TO_SCAN2_SEGM_NAME + " " + SCAN1_TO_SCAN2_DEFORM_NAME + " " 
                                                                      + SCAN1_TO_SCAN2_DEFORM_INVERSE_NAME + " " + SCAN1_TO_SCAN2_RESAMPLED_NAME + " "  
                                                                  + ANALYSIS_SEGM_FILE + " " + ANALYSIS_JACOBIAN_FILE;
      cout << CMD.c_str() << endl;
      cout << "=======" << endl;
   
      app->Script(CMD.c_str());

      CMD =  "lindex [::TumorGrowthTcl::ReadASCIIFile " + ANALYSIS_SEGM_FILE +"] 0";
      Analysis_SEGM_Growth = atof(app->Script(CMD.c_str()));
      cout << "Segmentation Result " << Analysis_SEGM_Growth <<endl;; 

      CMD =  "lindex [::TumorGrowthTcl::ReadASCIIFile " + ANALYSIS_JACOBIAN_FILE +"] 0";
      Analysis_JACO_Growth = atof(app->Script(CMD.c_str()));
      cout << "Jacobian Result: " << Analysis_JACO_Growth << endl;
    } 

    // 
    // ------------- Print Out Results --------------------
    // 
    if (1) {
      std::string fileName = tg.WorkingDir + "/AnalysisOutcome.log";
      std::ofstream outFile(fileName.c_str());
      if (outFile.fail()) {
         cout << "Error: Cannot write to file " << fileName.c_str() << endl;
     return EXIT_FAILURE;
      }
 
      outFile  << "This file was generated by vtkMrmTumorGrowthNode " << "\n";
      outFile  << "Date:      " << app->Script("exec date") << "\n";
      outFile  << "Scan1_Ref: " << tgScan1.c_str()     << "\n";
      outFile  << "Scan2_Ref: " << tgScan2.c_str()     << "\n";
      outFile  << "ROI:" << endl;
      outFile  << "  Min: " << tgROIMin[0] << " " << tgROIMin[1] << " " << tgROIMin[2] << "\n";
      outFile  << "  Max: " << tgROIMax[0] << " " << tgROIMax[1] << " " << tgROIMax[2] << "\n";
      outFile  << "Threshold: [" << tgThreshold[0] <<", " << tgThreshold[1] << "]\n";
      if (tgIntensityAnalysisFlag) {
        outFile  << "Analysis based on Intensity Pattern" << "\n";
        outFile  << "  Sensitivity:      " << tgSensitivity << "\n";
        outFile  << "  Intensity Metric: " << floor(Analysis_Intensity_Growth*SuperSampleVol*1000)/1000.0 << "mm^3 (" 
         << int(Analysis_Intensity_Growth*SuperSampleRatio) << " Voxels)" << "\n";
      }
      if (tgDeformableAnalysisFlag) { 
        outFile  << "Analysis based on Deformable Map" << "\n";
        outFile  << "  Segmentation Metric: "<<  floor(Analysis_SEGM_Growth*1000)/1000.0 << "mm^3 (" 
                 << int(Analysis_SEGM_Growth/Scan1Vol) << " Voxels)\n";
        outFile  << "  Jacobian Metric:     "<<  floor(Analysis_JACO_Growth*1000)/1000.0 << "mm^3 (" << int(Analysis_JACO_Growth/Scan1Vol ) << " Voxels)\n";
      }
      outFile  << "\n" << endl;
      std::string CMD;
      for (int i = 0 ; i < argc ; i ++) {
    CMD.append(argv[i]);
    CMD.append(" ");
      }
      outFile  << "PWD:      " <<  getenv("PWD") << "\n";
      outFile  << "CMD:      " << CMD.c_str() << "\n";


      outFile.close();
    }

    // 
    // ------------- CLEAN UP --------------------
    // 

    // Delete all instances
    if (Scan1Intensity)  Scan1Intensity->Delete();
    if (Scan2Intensity)  Scan2Intensity->Delete();
    if (Scan2Local)       Scan2Local->Delete();
    if (Scan2LocalNormalized)  Scan2LocalNormalized->Delete();
    if (Scan1PreSegment)  Scan1PreSegment->Delete();
    if (Scan1Segment)     Scan1Segment->Delete();
    if (Scan1SuperSample) Scan1SuperSample->Delete();
    if (Scan2SuperSample) Scan2SuperSample->Delete();
    if (Scan2Global)      Scan2Global->Delete();
    if (Scan1SegmentOutput) Scan1SegmentOutput->Delete();

    supersampleMatrix->Delete();
    logic->Delete();
    app->Delete();
  } 
  catch (...) 
    { 
    cout << "default exception"; 
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;  
}
