/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkRegistration.h,v $
  Language:  C++
  Date:      $Date: 2007/07/12 16:24:29 $
  Version:   $Revision: 1.5 $

  =========================================================================*/
// .NAME vtkRegistration
// .SECTION Description
// Faciliates calibration of the stylus and registration of two different 
// reference systems.
//=========================================================================

#ifndef __vtkRegistration_h
#define __vtkRegistration_h

//#include "vtkLapUSNavSysConfigure.h"
#include "vtkSystemIncludes.h"
#include "vtkObjectFactory.h"
#include "vtkTransform.h"
#include "vtkMatrix4x4.h"
#include "vtkDoubleArray.h"
#include "vtkPoints.h"
#include "vtkMath.h"
#include "vtkLandmarkTransform.h"

class  vtkRegistration: public vtkTransform {
public:
  // Description:
  /// Create a new instance of the class.
  static vtkRegistration *New();

  // Description:
  /// Defines some useful class identity functions.
  vtkTypeRevisionMacro(vtkRegistration, vtkTransform);

  // Description:
  /// Print class information.
  void PrintSelf(ostream &os, vtkIndent indent);

  // Description:
  /// Creates the stylus calibration matrix by computing the sensor coordinates
  /// of the center of a sphere created by a set of bird matrices passed from the stylus 
  /// (which is moved around while keeping the tip at a fixed point), using the nonlinear 
  /// least squares method - expects a file containing the sensor matrices.
  void CalibrateStylus(const char *stylusBirdMatsFile);

  // Description:
  /// Auxilliary function for calibrating the stylus.
  double *CalibrateStylusAux(vtkDoubleArray *X, double *beta);
  
  //BTX
  // Description:
  /// Auxilliary function for calibrating the stylus. It is an implementation of the Gauss-Newton nonlinear 
  /// least squares algorithm.
  double *FindCalibrationParameters(vtkDoubleArray *X, double *y, double *(vtkRegistration::*CalibFunPtr)(vtkDoubleArray *, double *), double *beta);
  //ETX
  
  // Description: 
  /// Calculates the registration matrix based on the two sets of points passed
  /// as arguments, with the first containing points from the initial frame and
  /// the second containing points from the target - expects two files containing 
  /// the points.
  void ComputeRegistration1(const char *frame_one_pts_file, const char *frame_two_pts_file);

  // Description: 
  /// Calculates the registration matrix based on the two sets of points passed
  /// as arguments, with the first being the initial frame and the second being
  /// the target frame.
  void ComputeRegistration2(vtkDoubleArray *frame_one_pts, vtkPoints *frame_two_pts);

  // Description: 
  /// Registration matrix get function.
  vtkGetObjectMacro(RegistrationMatrix, vtkMatrix4x4);

  // Description: 
  /// Calibration matrix get function.
  vtkGetObjectMacro(CalibrationMatrix, vtkMatrix4x4);

  // Description:
  /// Calibration matrix set function.
  vtkSetObjectMacro(CalibrationMatrix, vtkMatrix4x4);

protected: 
  vtkRegistration();
  ~vtkRegistration();
  vtkRegistration(const vtkRegistration&){};  // Not implemented.
  void operator=(const vtkRegistration&) {};  // Not implemented.

private:
  // Description: 
  /// The calculated registration matrix.
  vtkMatrix4x4 *RegistrationMatrix;

  // Description: 
  /// The calibration matrix for the stylus, which is presumably
  /// used in obtaining the location of the stylus' tip. It is set to the 
  /// identity matrix by default.
  vtkMatrix4x4 *CalibrationMatrix;

};

#endif



