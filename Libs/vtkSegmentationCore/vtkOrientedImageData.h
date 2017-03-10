/*==============================================================================

  Copyright (c) Laboratory for Percutaneous Surgery (PerkLab)
  Queen's University, Kingston, ON, Canada. All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Csaba Pinter, PerkLab, Queen's University
  and was supported through the Applied Cancer Research Unit program of Cancer Care
  Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care

==============================================================================*/

#ifndef __vtkOrientedImageData_h
#define __vtkOrientedImageData_h

// Segmentation includes
#include "vtkSegmentationCoreConfigure.h"

#include "vtkImageData.h"

class vtkMatrix4x4;

/// \ingroup SegmentationCore
/// \brief Image data containing orientation information
///
/// This extends vtkImageData to arbitrary grid orientation.
///
class vtkSegmentationCore_EXPORT vtkOrientedImageData : public vtkImageData
{
public:
  static vtkOrientedImageData *New();
  vtkTypeMacro(vtkOrientedImageData,vtkImageData);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  /// Shallow copy
  virtual void ShallowCopy(vtkDataObject *src);
  /// Deep copy
  virtual void DeepCopy(vtkDataObject *src);
  /// Copy orientation information only
  virtual void CopyDirections(vtkDataObject *src);

public:
  /// Set directions only
  void SetDirections(double dirs[3][3]);
  /// Set directions only
  void SetDirections(double ir, double ia, double is,
                     double jr, double ja, double js,
                     double kr, double ka, double ks);

  void GetDirections(double dirs[3][3]);

  /// Utility function that returns the min spacing between the 3 orientations
  double GetMinSpacing();

  /// Utility function that returns the max spacing between the 3 orientations
  double GetMaxSpacing();

  /// Get matrix including directions only
  void GetDirectionMatrix(vtkMatrix4x4* mat);
  /// Set directions by matrix
  void SetDirectionMatrix(vtkMatrix4x4* mat);

  /// Get the geometry matrix that includes the spacing and origin information
  void GetImageToWorldMatrix(vtkMatrix4x4* mat);
  /// Convenience method to set the directions, spacing, and origin from a matrix
  void SetImageToWorldMatrix(vtkMatrix4x4* mat);
  /// Same as SetImageToWorldMatrix. Kept for backward compatibility.
  void SetGeometryFromImageToWorldMatrix(vtkMatrix4x4* mat);

  /// Compute image bounds (xmin,xmax, ymin,ymax, zmin,zmax).
  virtual void ComputeBounds();

  /// Get the inverse of the geometry matrix
  void GetWorldToImageMatrix(vtkMatrix4x4* mat);

  /// Determines whether the image data is empty (if the extent has 0 voxels then it is)
  bool IsEmpty();

public:
  /// Set bounds to an uninitialized state. \sa vtkMath::UninitializeBounds works incorrectly in cases where
  /// the maximum bound of an object along an axis is smaller than -1. In that case \sa vtkSegment::ExtendBounds
  /// keeps -1 as the upper bound, which is incorrect.
  static void UninitializeBounds(double bounds[6])
    {
    bounds[0] = VTK_DOUBLE_MAX;
    bounds[1] = VTK_DOUBLE_MIN;
    bounds[2] = VTK_DOUBLE_MAX;
    bounds[3] = VTK_DOUBLE_MIN;
    bounds[4] = VTK_DOUBLE_MAX;
    bounds[5] = VTK_DOUBLE_MIN;
    };

protected:
  vtkOrientedImageData();
  ~vtkOrientedImageData();

protected:
  /// Direction matrix for the image data
  /// These are unit length direction cosines
  double Directions[3][3];

private:
  vtkOrientedImageData(const vtkOrientedImageData&);  // Not implemented.
  void operator=(const vtkOrientedImageData&);  // Not implemented.
};

#endif
