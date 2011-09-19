/*==============================================================================

  Program: 3D Slicer

  Copyright (c) 2010 Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qSlicerVolumeRenderingIO
#define __qSlicerVolumeRenderingIO

// SlicerQt includes
#include "qSlicerIO.h"

// Volume Rendering includes
class qSlicerVolumeRenderingIOPrivate;
class vtkSlicerVolumeRenderingLogic;

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_VolumeRendering
class qSlicerVolumeRenderingIO: public qSlicerIO
{
  Q_OBJECT
public: 
  qSlicerVolumeRenderingIO(QObject* parent = 0);
  qSlicerVolumeRenderingIO(vtkSlicerVolumeRenderingLogic* logic, QObject* parent = 0);

  void setLogic(vtkSlicerVolumeRenderingLogic* logic);
  vtkSlicerVolumeRenderingLogic* logic()const;

  // Reimplemented for IO specific description
  virtual QString description()const;
  virtual IOFileType fileType()const;
  virtual QStringList extensions()const;

  virtual bool load(const IOProperties& properties);

protected:
  QScopedPointer<qSlicerVolumeRenderingIOPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerVolumeRenderingIO);
  Q_DISABLE_COPY(qSlicerVolumeRenderingIO);
};

#endif
