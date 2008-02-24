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

// ./TumorGrowthCommandLine --sensitivity 0.5 --threshold 100,277 --roi_min 73,135,92 --roi_max 95,165,105 --intensity_analysis --deformable_analysis --scan1 /data/local/BrainScienceFoundation/Demo/07-INRIA/data/SILVA/2006-spgr.nhdr --scan2 /data/local/BrainScienceFoundation/Demo/07-INRIA/data/SILVA/2007-spgr-scan1.nhdr
//
//
// This is necessary to load in TumorGrowth package in TCL interp.
extern "C" int Slicerbasegui_Init(Tcl_Interp *interp);
extern "C" int Tumorgrowth_Init(Tcl_Interp *interp);

#define tgVtkCreateMacro(name,type) \
  name  = type::New(); \
  name##Tcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp, name)); 

#define tgVtkDefineMacro(name,type) \
  type *name; \
  std::string name##Tcl;\
  tgVtkCreateMacro(name,type); 

#define tgSetDataMacro(name) \
 virtual int Set##name(const char *fileName) { \
   if (strcmp(fileName,"None")) { \
    tgVtkCreateMacro(this->name,vtkImageData); \
        return this->tgReadVolume(fileName,this->name); \
   } \
   this->name = NULL; \
   return 0; \
 }

class tgCMDLineStructure {
  public:

  tgCMDLineStructure(Tcl_Interp *init) {
    this->Scan1Data = NULL;
    this->Scan1DataTcl = "";
    this->Scan2Data = NULL;
    this->Scan2DataTcl = "";
    this->interp = init;
  } 

  ~tgCMDLineStructure() {
    if (this->Scan1Data) {
      Scan1Data->Delete();
      this->Scan1Data = NULL;
    }
    if (this->Scan2Data) {
      Scan2Data->Delete();
      this->Scan2Data = NULL;
    }
    this->interp = NULL;
  } 

  tgSetDataMacro(Scan1Data);
  tgSetDataMacro(Scan2Data);

  void SetWorkingDir(vtkKWApplication *app, const char* fileNameScan1) {
    this->WorkingDir = vtksys::SystemTools::GetFilenamePath(fileNameScan1) + "-TGcmd";
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

  vtkImageData *Scan2Data; 
  std::string Scan2DataTcl;

  std::string WorkingDir;

  private:
  int tgReadVolume(const char *fileName, vtkImageData *Output);

  Tcl_Interp *interp;

};


int tgCMDLineStructure::tgReadVolume(const char *fileName, vtkImageData *Output) {
  // Currently only works with nrrd files bc I do not know how to converd itk::ImageFileReader vtkImageData
  vtkNRRDReader *reader = vtkNRRDReader::New();
  reader->SetFileName(fileName);
  reader->Update();
  Output->DeepCopy(reader->GetOutput());
  if (reader->GetReadStatus()) {
    cout << "ERROR: tgReadVolume: could not read " << fileName << endl;
    return 1;
  }
  reader->Delete();
  return 0;
}

void tgWriteVolume(const char *fileName, vtkImageData *Output) {
  vtkMatrix4x4 *export_matrix = vtkMatrix4x4::New(); 
  int *EXTENT = Output->GetExtent();
  int dims[3] = {EXTENT[1] - EXTENT[0] + 1, EXTENT[3] - EXTENT[2] + 1,EXTENT[5] - EXTENT[4] + 1};
  vtkMRMLVolumeNode::ComputeIJKToRASFromScanOrder("IS",Output->GetSpacing(),dims,1,export_matrix);
  export_matrix->Invert();

  vtkITKImageWriter *iwriter = vtkITKImageWriter::New(); 
  iwriter->SetInput(Output); 
  iwriter->SetFileName(fileName);
  iwriter->SetRasToIJKMatrix(export_matrix);
  iwriter->SetUseCompression(1);
  iwriter->Write();
  iwriter->Delete();
  export_matrix->Delete();

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
  if ( !vtksys::SystemTools::GetEnv("SLICER_HOME", slicerHome) )
  {
    std::string programPath;
    std::string errorMessage;
    if ( !vtksys::SystemTools::FindProgramPath(argv[0], programPath, errorMessage) ) return 1;

    std::string homeEnv = "SLICER_HOME=";
    homeEnv += vtksys::SystemTools::GetFilenamePath(programPath.c_str()) + "/../../../";
   
    // cout << "Set environment: " << homeEnv.c_str() << endl;
    vtkKWApplication::PutEnv(const_cast <char *> (homeEnv.c_str()));
  }
  return 0;
}

int tgRegisterAG(vtkKWApplication *app, std::string Target, std::string Source, std::string Transform, std::string WorkingDir, std::string Output) {
  std::string CMD = "::TumorGrowthReg::RegistrationAG " +  Target + " IS " + Source + " IS 1 0 0 50 mono 3 " + Transform;
   
  if (!app->Script(CMD.c_str())) {
    cout << "Error:  Could not perform Global Registration";
    return 1; 
  }
   
  CMD = "::TumorGrowthReg::ResampleAG_GUI "+ Source + " " + Target + " " +  Transform + " " + Output;
  app->Script(CMD.c_str());
   
  CMD = "::TumorGrowthReg::WriteTransformationAG "  +  Transform + " " + WorkingDir;
  app->Script(CMD.c_str());

  return 0; 
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

    // SLICER_HOME
    if (tgSetSLICER_HOME(argv)) {
      cout << "Error: Cannot find executable" << endl;
      return EXIT_FAILURE; 
    }

    // When I include the following line I get the leak message 
    // vtkSlicerApplication *app   = vtkSlicerApplication::GetInstance();
    vtkKWApplication *app   = vtkKWApplication::New();

    vtkTumorGrowthLogic  *logic = vtkTumorGrowthLogic::New();
    std::string logicTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,logic));
    logic->SourceAnalyzeTclScripts(app);
 
    // -------------------------------------
    // Load Parameters for pipeline 
    // -------------------------------------
    tgCMDLineStructure tg(interp);   

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
 
    // -------------------------------------
    // Run pipeline 
    // -------------------------------------
 
    // Create Working Directory 
    tg.SetWorkingDir(app,tgScan1.c_str()); 
    tgVtkDefineMacro(Scan2Global,vtkImageData); 

    // 
    // -----------GLOBAL REGISTRATION --------
    // 
    if (1) {
      // -------------------------------------
      cout << "=== Global Rigid Registration ===" << endl;
      vtkGeneralTransform* globalTransform = logic->CreateGlobalTransform(); 
      std::string globalTransformTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,globalTransform));

      if (tgRegisterAG( app, tg.Scan1DataTcl, tg.Scan2DataTcl , globalTransformTcl, tg.GetWorkingDir(), Scan2GlobalTcl)) return EXIT_FAILURE;

      std::string CMD = "catch { exec mv " + tg.WorkingDir + "/LinearRegistration.txt " + tg.WorkingDir + "/GlobalLinearRegistration.txt }";
      app->Script(CMD.c_str());

      CMD = "catch { ::TumorGrowthReg::DeleteTransformAG " + globalTransformTcl + " }";
      app->Script(CMD.c_str());

      CMD = tg.WorkingDir + "/TG_scan2_Global.nhdr";
      tgWriteVolume(CMD.c_str(),Scan2Global);    
       
    } else {
     cout << "Debugging - jump over global registration" << endl;
     Scan2Global->DeepCopy(tg.Scan1Data);
    }
 
    // 
    // --------------- ROI --------------------
    // 
    double *Spacing;
    double SuperSampleSpacing; 
    double SuperSampleVol;     
    double Scan1Vol;     
    double SuperSampleRatio;
 
    tgVtkDefineMacro(Scan1SuperSample,vtkImageData); 
    tgVtkDefineMacro(Scan2SuperSample,vtkImageData); 
    std::string Scan1SuperSampleFileName = tg.WorkingDir + "/TG_scan1_SuperSampled.nhdr";

    if (1) {
      // -------------------------------------
      cout << "=== Define ROI for each scan ===" << endl;
      Spacing =  tg.Scan1Data->GetSpacing();
      int ROIMin[3] = {tgROIMin[0], tgROIMin[1],  tgROIMin[2]};
      int ROIMax[3] = {tgROIMax[0], tgROIMax[1],  tgROIMax[2]};
 
      SuperSampleSpacing = logic->DefineSuperSampleSize(Spacing, ROIMin, ROIMax);
      SuperSampleVol     = SuperSampleSpacing*SuperSampleSpacing*SuperSampleSpacing;
      Scan1Vol           = (Spacing[0]*Spacing[1]*Spacing[2]);
      SuperSampleRatio   = SuperSampleVol/Scan1Vol;
 
      if (logic->CreateSuperSampleFct(tg.Scan1Data,ROIMin, ROIMax, SuperSampleSpacing,Scan1SuperSample)) {
       cout << "ERROR: Could not super sample scan1 " << endl;
       return EXIT_FAILURE; 
      }
      tgWriteVolume(Scan1SuperSampleFileName.c_str(),Scan1SuperSample);    
 
      if (logic->CreateSuperSampleFct(Scan2Global,ROIMin, ROIMax, SuperSampleSpacing,Scan2SuperSample)) {
       cout << "ERROR: Could not super sample scan1 " << endl;
       return EXIT_FAILURE; 
      }
       std::string NAME = tg.WorkingDir + "/TG_scan2_Global_SuperSampled.nhdr";
      tgWriteVolume(NAME.c_str(),Scan2SuperSample);    
    } else {
     cout << "Debugging - jump over super sampling" << endl;      
    }
 
    // 
    // ------------- SEGMENTATION --------------------
    //

    tgVtkDefineMacro(Scan1PreSegment,vtkImageThreshold); 
    tgVtkDefineMacro(Scan1Segment,vtkImageIslandFilter); 
    std::string Scan1SegmentFileName;

    if (1) {
      // -------------------------------------
      cout << "=== Segment Scan1 ===" << endl;
      int range[2] = {tgThreshold[0],tgThreshold[1]};
      vtkTumorGrowthLogic::DefinePreSegment(Scan1SuperSample,range,Scan1PreSegment);
      vtkTumorGrowthLogic::DefineSegment(Scan1PreSegment->GetOutput(),Scan1Segment);

      Scan1SegmentFileName = tg.WorkingDir + "/TG_scan1_Segment.nhdr";
      tgWriteVolume(Scan1SegmentFileName.c_str(),Scan1Segment->GetOutput());
    }
    char *Scan1SegmentOutputTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,Scan1Segment->GetOutput()));

    // 
    // ------------- NORMALIZE  --------------------
    //

    tgVtkDefineMacro(Scan2Normalized,vtkImageData);
    if (1) {
       // -------------------------------------
       cout << "=== Normalize Scan2 ===" << endl;
       std::string CMD = "::TumorGrowthTcl::HistogramNormalization_FCT " + Scan1SuperSampleTcl + " " + Scan1SegmentOutputTcl + " " 
                                                                     + Scan2SuperSampleTcl + " " + Scan2NormalizedTcl;
       app->Script(CMD.c_str()); 
       std::string NAME = tg.WorkingDir + "/TG_scan2_norm.nhdr";
       tgWriteVolume(NAME.c_str(),Scan2Normalized);
    }

    // 
    // ------------- LOCAL REGISTRATION  --------------------
    //
    tgVtkDefineMacro(Scan2Local,vtkImageData); 
    std::string Scan2LocalFileName = tg.WorkingDir + "/TG_scan2_Local.nhdr";
    
    if (1) {
      // -------------------------------------
      cout << "=== Local Rigid Registration ===" << endl;
      vtkGeneralTransform* localTransform = logic->CreateLocalTransform(); 
      std::string localTransformTcl = vtksys::SystemTools::DuplicateString(vtkKWTkUtilities::GetTclNameFromPointer(interp,localTransform));

      if (tgRegisterAG( app, Scan1SuperSampleTcl, Scan2NormalizedTcl, localTransformTcl, tg.GetWorkingDir(), Scan2LocalTcl)) return EXIT_FAILURE;
      std::string CMD = "catch { exec mv " + tg.WorkingDir + "/LinearRegistration.txt " + tg.WorkingDir + "/LocalLinearRegistration.txt }";
      app->Script(CMD.c_str());

      CMD = "catch { ::TumorGrowthReg::DeleteTransformAG " + localTransformTcl + " }";
      app->Script(CMD.c_str());
   
      tgWriteVolume(Scan2LocalFileName.c_str(),Scan2Local);

    } else {
     cout << "Debugging - jump over local registration" << endl;
     Scan2Local->DeepCopy(Scan1SuperSample);
    }

    // 
    // ------------- ANALYZE TYPE: INTENSITY  --------------------
    //
    tgVtkDefineMacro(Scan1Intensity,vtkImageData); 
    tgVtkDefineMacro(Scan2Intensity,vtkImageData); 
    double Analysis_Intensity_Growth = -1;

    if (tgIntensityAnalysisFlag) { 
      cout << "=== Intensity Based Analysis ===" << endl;
      // ------------- INTENSITY THRESHOLDING  --------------------
      char ThreshString[1024];
      sprintf(ThreshString," %i %i ", tgThreshold[0], tgThreshold[1]);
      std::string CMD = "::TumorGrowthTcl::IntensityThresholding_Fct " + Scan1SuperSampleTcl + " " + Scan1SuperSampleTcl + ThreshString + Scan1IntensityTcl;
      app->Script(CMD.c_str());

      std::string Scan1IntensityFileName = tg.WorkingDir + "/TG_scan1_Thr.nhdr";
      tgWriteVolume(Scan1IntensityFileName.c_str(),Scan1Intensity);

      CMD = "::TumorGrowthTcl::IntensityThresholding_Fct " + Scan2LocalTcl + " " + Scan1SuperSampleTcl + ThreshString + Scan2IntensityTcl;
      app->Script(CMD.c_str());

      std::string Scan2IntensityFileName = tg.WorkingDir + "/TG_scan2_Thr.nhdr";
      tgWriteVolume(Scan2IntensityFileName.c_str(),Scan2Intensity);
      cout <<  "---!!!- > "  << Scan2IntensityFileName.c_str() << endl;

      // ------------- ANALYSIS  --------------------
      char Sensitivity[100];
      sprintf(Sensitivity, " %f" ,tgSensitivity);

      CMD = "::TumorGrowthTcl::Analysis_Intensity_CMD " + logicTcl + " " + Scan1IntensityTcl + " " + Scan1SegmentOutputTcl + " " + Scan2IntensityTcl + Sensitivity;
      app->Script(CMD.c_str());

      CMD = tg.WorkingDir + "/TG_Analysis_Intensity.nhdr";
      tgWriteVolume(CMD.c_str(),logic->GetAnalysis_Intensity_ROIBinReal());
      cout << "=========================" << endl;    
      Analysis_Intensity_Growth  = logic->MeassureGrowth();
      cout << "Analysis Intensity Growth: " <<  Analysis_Intensity_Growth  << " Super sample " << SuperSampleVol << endl;
      printf("Intensity Metric: %.3f mm^3 (%d Voxels)\n",  Analysis_Intensity_Growth *SuperSampleVol,int( Analysis_Intensity_Growth *SuperSampleRatio));

      app->Script("::TumorGrowthTcl::Analysis_Intensity_DeleteOutput_FCT");
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

      std::string CMD =  "::TumorGrowthTcl::Analysis_Deformable_Fct " + Scan1SuperSampleFileName + " " + Scan1SegmentFileName + " " + Scan2LocalFileName + " "
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
        outFile  << "  Jacobian Metric:     "<<  floor(Analysis_SEGM_Growth*1000)/1000.0 << "mm^3 (" << int(Analysis_SEGM_Growth/Scan1Vol ) << " Voxels)\n";
      }
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
    if (Scan2Normalized)  Scan2Normalized->Delete();
    if (Scan1PreSegment)  Scan1PreSegment->Delete();
    if (Scan1Segment)     Scan1Segment->Delete();
    if (Scan1SuperSample) Scan1SuperSample->Delete();
    if (Scan2SuperSample) Scan2SuperSample->Delete();
    if (Scan2Global)      Scan2Global->Delete();
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
