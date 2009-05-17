/*=========================================================================

Program:   Visualization Toolkit
Module:    $RCSfile: vtkRegistration.cxx,v $
Language:  C++
Date:      $Date: 2007/07/12 16:24:29 $
Version:   $Revision: 1.5 $

=========================================================================*/

#include <vtkRegistration.h>

vtkCxxRevisionMacro(vtkRegistration, "$Revision: 1.5 $");
vtkStandardNewMacro(vtkRegistration);

vtkRegistration::vtkRegistration() {

  // Initialize registration matrix.
  this->RegistrationMatrix = vtkMatrix4x4::New();
  this->RegistrationMatrix->Identity();

  // Initialize calibration matrix.
  this->CalibrationMatrix = vtkMatrix4x4::New();
  this->CalibrationMatrix->Identity();

}

vtkRegistration::~vtkRegistration() {

  // Delete registration matrix.
  this->RegistrationMatrix->Delete();

  // Delete calibration matrix.
  this->CalibrationMatrix->Delete();

}

void vtkRegistration::PrintSelf(ostream &os, vtkIndent indent) {

  // Print registration matrix.
  os << "Registration Matrix: " << endl;
  for(int i=0; i < 4; i++) {
  os << this->RegistrationMatrix->GetElement(i,0) << " " 
     << this->RegistrationMatrix->GetElement(i,1) << " " 
     << this->RegistrationMatrix->GetElement(i,2) << " " 
     << this->RegistrationMatrix->GetElement(i,3) << endl;
  }
  cout << "\n";

  // Print stylus calibration matrix.
  os << "Stylus calibration matrix: " << endl;
  for(int i=0; i < 4; i++) {
  os << this->CalibrationMatrix->GetElement(i,0) << " " 
     << this->CalibrationMatrix->GetElement(i,1) << " " 
     << this->CalibrationMatrix->GetElement(i,2) << " " 
     << this->CalibrationMatrix->GetElement(i,3) << endl;
  } 

}

void vtkRegistration::CalibrateStylus(const char *stylusBirdMatsFile) {

  vtkMatrix4x4 *stylus_matrix = vtkMatrix4x4::New(); // holds current stylus sensor matrix
  vtkDoubleArray *inv_stylus_mats = vtkDoubleArray::New(); // stores inverse of stylus matrices
  inv_stylus_mats->SetNumberOfComponents(4);
  vtkDoubleArray *X = vtkDoubleArray::New();   // stores location of sensor in transmitter frame for
  X->SetNumberOfComponents(4); // each reading
  vtkDoubleArray *stylus_mats = vtkDoubleArray::New(); // holds all stylus sensor matrices in file
  stylus_mats->SetNumberOfComponents(4);

  // open necessary file stream
  ifstream ifs(stylusBirdMatsFile);
  if(!ifs.is_open()) return;
  
  int mNumber = 0;
  ifs >> mNumber;
  for(int q=0; q < mNumber; q++) { 

  // read next stylus sensor matrix and get the x,y,z part of the translation column
  double mat_element; // holds individual matrix element
  double xyz[4]; // holds translation column
  double row[4]; // holds a row of the matrix
  double **stylus_mat = new double *[4]; // holds whole matrix
  for(int i=0; i < 4; i++) stylus_mat[i] = new double[4];
  for(int i=0; i < 4; i++) {
  for(int j=0; j < 4; j++) {
  ifs >> mat_element;
  if((j == 3)) xyz[i] = mat_element;
  stylus_mat[i][j] = mat_element;
  row[j] = mat_element;
  }
  stylus_mats->InsertNextTuple(row);
  }
  X->InsertNextTuple(xyz);

  double **inv_stylus_mat = new double *[4];
  for(int i=0; i < 4; i++) inv_stylus_mat[i] = new double[4];
  vtkMath::InvertMatrix(stylus_mat, inv_stylus_mat, 4);
  double inv_stylus_mat_tuple[4];
  for(int i=0; i < 4; i++) {
  for(int j=0; j < 4; j++) inv_stylus_mat_tuple[j] = inv_stylus_mat[i][j];
  inv_stylus_mats->InsertNextTuple(inv_stylus_mat_tuple);
  }
  } 

  double *beta = new double[4];
  for(int i=0; i < 3; i++) beta[i] = 0;
  beta[3] = 1;
  double *y = new double[X->GetNumberOfTuples()];
  for(int i=0; i < X->GetNumberOfTuples(); i++) y[i] = 0;

  beta = FindCalibrationParameters(X, y, &vtkRegistration::CalibrateStylusAux, beta);
  beta[3] = 1;
    
  // using center, determine the location of the stylus tip in the receiver frame
  // for each sensor reading. add the x,y,z values for these points and then determine
  // the mean x,y,z values and use this value for the final calibration stylus 
  // calibration matrix.
  double *points = new double[inv_stylus_mats->GetNumberOfTuples()];
  for(int i=0; i < inv_stylus_mats->GetNumberOfTuples(); i++) {
  points[i] = 0;
  for(int j=0; j < 4; j++) points[i] += inv_stylus_mats->GetComponent(i, j) * beta[j];
  }
  
  double mean[3] = {0};
  for(int i=0; i < inv_stylus_mats->GetNumberOfTuples(); i += 4) {
  mean[0] += points[i];
  mean[1] += points[i+1];
  mean[2] += points[i+2];
  }

  double center[3];
  for(int i=0; i < 3; i++) {
  mean[i] = mean[i] / (inv_stylus_mats->GetNumberOfTuples()/4);
  center[i] = mean[i];
  }

  vtkMatrix4x4 *calib_matrix = vtkMatrix4x4::New();
  for(int i=0; i < 3; i++) calib_matrix->SetElement(i, 3, center[i]);

  this->CalibrationMatrix = calib_matrix;

}

double *vtkRegistration::CalibrateStylusAux(vtkDoubleArray *X, double *beta) {
  double *temp = new double[X->GetNumberOfTuples()];

  for(int i=0; i < X->GetNumberOfTuples(); i++) 
    temp[i] = pow(X->GetComponent(i, 0) - beta[0],2) + pow(X->GetComponent(i, 1) - beta[1],2) + pow(X->GetComponent(i, 2) - beta[2],2) - pow(beta[3],2);
      
  return temp;

}

double *vtkRegistration::FindCalibrationParameters(vtkDoubleArray *X, double *y, double *(vtkRegistration::*CalibFunPtr)(vtkDoubleArray *, double *), double *beta) {
  double **Jacobian = new double *[X->GetNumberOfTuples()];
  for(int i=0; i < X->GetNumberOfTuples(); i++) Jacobian[i] = new double[X->GetNumberOfComponents()];
  double *betanew = new double[X->GetNumberOfComponents()];
  for(int i=0; i < X->GetNumberOfComponents(); i++) betanew[i] = beta[i] + 1;

  double sse = 1, sseold = sse, eps = pow(2.0,-52), betatol = 0.0001, rtol = 0.0001;
  int iter = 0, maxiter = 100;

  // temporarily holds value that is passed to vtkMath::Norm()
  float *norm_temp = new float[X->GetNumberOfComponents()];
  for(int i=0; i < X->GetNumberOfComponents(); i++) 
    norm_temp[i] = (betanew[i] - beta[i]) / (beta[i] + sqrt(eps));

  while((vtkMath::Norm(norm_temp, X->GetNumberOfComponents()) > betatol || abs(sseold - sse)/(sse + sqrt(eps)) > rtol) && (iter < maxiter)) {

  // update 'beta'
  if(iter > 0) {for(int i=0; i < X->GetNumberOfComponents(); i++) beta[i] = betanew[i];}

  // increment iteration count
  ++iter;
  
  // update 'yfit'
  double *yfit = (this->*CalibFunPtr)(X, beta);

  // update residuals
  double *r = new double[X->GetNumberOfTuples()];
  for(int i=0; i < X->GetNumberOfTuples(); i++) r[i] = y[i] - yfit[i];
  
  // perform  r' * r, sum of squares
  sseold = 0;
  for(int i=0; i < X->GetNumberOfTuples(); i++) sseold += pow(r[i],2);

  // update Jacobian
  for(int i=0; i < X->GetNumberOfComponents(); i++) {
      
  double *delta = new double[X->GetNumberOfComponents()];
  for(int j=0; j < X->GetNumberOfComponents(); j++) delta[j] = 0;

  if(beta[i] == 0) {
  double nb = sqrt(vtkMath::Norm((float *) beta,X->GetNumberOfComponents()));
  delta[i] = sqrt(eps) * (nb + (nb == 0));
  }

  else { 
  delta[i] = sqrt(eps) * beta[i]; 
  }

  double *temp = new double[X->GetNumberOfComponents()];
  for(int j=0; j < X->GetNumberOfComponents(); j++) 
    temp[j] = beta[j] + delta[j];

  double *yplus = (this->*CalibFunPtr)(X, temp);

  for(int j=0; j < X->GetNumberOfTuples(); j++) 
    Jacobian[j][i] = (yplus[j] - yfit[j]) / (delta[i]);
  }

  // Gauss-Newton step

  // create'Jplus', which is 'Jacobian' with additional (beta * beta)-sized identity matrix
  double **Jplus = new double *[X->GetNumberOfTuples() + X->GetNumberOfComponents()];
  for(int i=0; i < X->GetNumberOfTuples() + X->GetNumberOfComponents(); i++)
    Jplus[i] = new double[X->GetNumberOfComponents()];

  // assign 'Jacobian' to 'Jplus' and set addition to 'Jacobian' to identity matrix
  for(int i=0; i < X->GetNumberOfTuples() + X->GetNumberOfComponents(); i++) {
  if(i < X->GetNumberOfTuples())
    for(int j=0; j < X->GetNumberOfComponents(); j++) Jplus[i][j] = Jacobian[i][j];
  else if(i >= X->GetNumberOfTuples()) {
  for(int j=0; j < X->GetNumberOfComponents(); j++) {
  if((i - X->GetNumberOfTuples()) == j) Jplus[i][j] = 0.01;
  else Jplus[i][j] = 0;
  }
  }
  }
    
  // create 'rplus', which is 'r' with added length equal to 'beta'
  double **rplus = new double *[X->GetNumberOfTuples() + X->GetNumberOfComponents()];
  for(int i=0; i < X->GetNumberOfTuples() + X->GetNumberOfComponents(); i++) 
    rplus[i] = new double[1];

  // assign 'r' to 'rplus' and set addition to 'r' to 0
  for(int i=0; i < X->GetNumberOfTuples() + X->GetNumberOfComponents(); i++) {
  if(i < X->GetNumberOfTuples()) rplus[i][0] = r[i];
  else if(i >= X->GetNumberOfTuples()) rplus[i][0] = 0;
  }

  // create the 'step' array, the size of which is # columns 'Jplus' by # columns 'rplus'
  double **step = new double *[X->GetNumberOfComponents()];
  for(int i=0; i < X->GetNumberOfComponents(); i++) step[i] = new double[1];

  vtkMath::SolveLeastSquares(X->GetNumberOfTuples() + X->GetNumberOfComponents(), Jplus, X->GetNumberOfComponents(), rplus, 1, step);

  // calculate 'betanew' by adding step to 'beta'
  for(int i=0; i < X->GetNumberOfComponents(); i++) betanew[i] = beta[i] + step[i][0];  

  // calculate 'yfitnew'
  double *yfitnew = (this->*CalibFunPtr)(X, betanew);

  // calculate 'rnew'
  double *rnew  = new double[X->GetNumberOfTuples()];
  for(int i=0; i < X->GetNumberOfTuples(); i++) rnew[i] = y[i] - yfitnew[i];

  // calculate sum of squares
  sse = 0;
  for(int i=0; i < X->GetNumberOfTuples(); i++) sse += pow(rnew[i],2);

  int int_iter = 0;
  while((sse > sseold) && (int_iter < 12)) {

  // update 'step'
  for(int i=0; i < X->GetNumberOfComponents(); i++) step[i][0] = step[i][0] / sqrt(10.0);

  // calculate 'betanew' by adding 'step' to 'beta'
  for(int i=0; i < X->GetNumberOfComponents(); i++) betanew[i] = beta[i] + step[i][0];

  // update 'yfitnew'
  yfitnew = (this->*CalibFunPtr)(X, betanew);

  // update 'rnew'
  for(int i=0; i < X->GetNumberOfTuples(); i++) rnew[i] = y[i] - yfitnew[i];

  // calculate sum of squares
  sse = 0;
  for(int i=0; i < X->GetNumberOfTuples(); i++) sse += pow(rnew[i], 2);

  // increment iteration count
  ++int_iter;
  }
    
  // update 'norm_temp'
  for(int i=0; i < X->GetNumberOfComponents(); i++) 
    norm_temp[i] = (betanew[i] - beta[i]) / (beta[i] + sqrt(eps));    
  }

  return beta;

}

void vtkRegistration::ComputeRegistration1(const char *frame_one_mtxs_file, const char *frame_two_pnts_file) {

  vtkPoints *source, *target; 
  source = vtkPoints::New();
  target = vtkPoints::New();
  double xyz[3];

  // take values from files and assign them to the appropriate array

  // source...
  ifstream fs;
  fs.open(frame_one_mtxs_file);
  if(!fs.is_open()) return;

  // target...
  ifstream ft;
  ft.open(frame_two_pnts_file);
  if(!ft.is_open()) return;

  // read sensor matrices
  vtkMatrix4x4 *birdMatrix = vtkMatrix4x4::New();
  double value;
  int count = 0;
  while( !fs.eof() && !ft.eof() ) {

  // get sensor matrix data
  for(int i=0; i < 4; i++) {
  for(int j=0; j < 4; j++) {
  fs >> value;
  birdMatrix->SetElement(i, j, value);
  }
  }

  // calculate stylus tip in sensor frame and 
  // assign to array
  for(int i=0; i < 3; i++) {
  xyz[i] = 0;
  for(int j=0; j < 4; j++) {
  xyz[i] += birdMatrix->GetElement(i, j)* this->CalibrationMatrix->GetElement(j, 3);
  }
  }
  source->InsertPoint(count, xyz);

  // get corresponding target point
  ft >> xyz[0] >> xyz[1] >> xyz[2];
  target->InsertPoint(count, xyz);

  ++count;
  }

  // close file streams
  fs.close();
  ft.close();

  // calculate the registration matrix
  vtkLandmarkTransform *tempTransform;
  tempTransform = vtkLandmarkTransform::New();

  tempTransform->SetModeToRigidBody();

  tempTransform->SetSourceLandmarks(source);
  tempTransform->SetTargetLandmarks(target);

  // assign value to class member
  this->RegistrationMatrix->DeepCopy(tempTransform->GetMatrix());
  
  tempTransform->Delete();

}

void vtkRegistration::ComputeRegistration2(vtkDoubleArray *frame_one_mtxs, vtkPoints *frame_two_pnts) {

  vtkPoints *source = vtkPoints::New(),
    *target = frame_two_pnts;

  double xyz[3];

  // read sensor matrices
  vtkMatrix4x4 *birdMatrix = vtkMatrix4x4::New();
  double value;
  int count = frame_one_mtxs->GetNumberOfTuples();

  for (int c=0 ; c < frame_one_mtxs->GetNumberOfTuples(); c++) {

  // get sensor matrix data
  for(int i=0; i < 4; i++) {
  for(int j=0; j < 4; j++) {
  value = frame_one_mtxs->GetComponent(c,j+i*4);
  birdMatrix->SetElement(i, j, value);
  }
  }
  // calculate stylus tip in sensor frame and 
  // assign to array
  for(int i=0; i < 3; i++) {
  xyz[i] = 0;
  for(int j=0; j < 4; j++) {
  xyz[i] += birdMatrix->GetElement(i, j)* this->CalibrationMatrix->GetElement(j, 3);
  }
  }
  source->InsertPoint(c, xyz);
  }

  // calculate the registration matrix
  vtkLandmarkTransform *tempTransform;
  tempTransform = vtkLandmarkTransform::New();

  tempTransform->SetModeToRigidBody();

  tempTransform->SetSourceLandmarks(source);
  tempTransform->SetTargetLandmarks(target);

  // assign value to class member
  this->RegistrationMatrix->DeepCopy(tempTransform->GetMatrix());
  
  tempTransform->Delete();

}

