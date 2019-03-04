/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Thomas Vaughan, PerkLab, Queen's University.

==============================================================================*/

/**
 * @class   vtkLinearSpline
 * @brief   computes an interpolating spline with piecewise linear segments
 *
 *
 * vtkLinearSpline is a concrete implementation of vtkSpline using
 * piecewise linearly interpolated segments.
 *
 * @sa
 * vtkSpline vtkCardinalSpline vtkKochanekSpline
*/

#ifndef vtkLinearSpline_h
#define vtkLinearSpline_h

#include "vtkSlicerMarkupsModuleMRMLExport.h" // For export macro

#include <vtkSpline.h>

class VTK_SLICER_MARKUPS_MODULE_MRML_EXPORT vtkLinearSpline : public vtkSpline
{
public:
  static vtkLinearSpline *New();

  vtkTypeMacro(vtkLinearSpline,vtkSpline);
  void PrintSelf(ostream& os, vtkIndent indent) VTK_OVERRIDE;

  /**
   * Compute linear splines for each dependent variable
   */
  void Compute () VTK_OVERRIDE;

  /**
   * Evaluate a 1D linear spline.
   */
  double Evaluate (double t) VTK_OVERRIDE;

  /**
   * Deep copy of linear spline data.
   */
  void DeepCopy(vtkSpline *s) VTK_OVERRIDE;

protected:
  vtkLinearSpline();
  ~vtkLinearSpline() VTK_OVERRIDE {}

private:
  vtkLinearSpline(const vtkLinearSpline&) = delete;
  void operator=(const vtkLinearSpline&) = delete;
};

#endif
