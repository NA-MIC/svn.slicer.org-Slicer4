/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright 2015 Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Andras Lasso (PerkLab, Queen's
  University) and Kevin Wang (Princess Margaret Hospital, Toronto) and was
  supported through OCAIRO and the Applied Cancer Research Unit program of
  Cancer Care Ontario.

==============================================================================*/

#ifndef __qSlicerTablesReader
#define __qSlicerTablesReader

// SlicerQt includes
#include "qSlicerFileReader.h"

class qSlicerTablesReaderPrivate;
class vtkSlicerTablesLogic;

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_DoubleArray
class qSlicerTablesReader
  : public qSlicerFileReader
{
  Q_OBJECT
public:
  typedef qSlicerFileReader Superclass;
  qSlicerTablesReader(QObject* parent = 0);
  qSlicerTablesReader(vtkSlicerTablesLogic* logic,
                       QObject* parent = 0);
  virtual ~qSlicerTablesReader();

  vtkSlicerTablesLogic* logic()const;
  void setLogic(vtkSlicerTablesLogic* logic);

  virtual QString description()const;
  virtual IOFileType fileType()const;
  virtual QStringList extensions()const;

  virtual bool load(const IOProperties& properties);
protected:
  QScopedPointer<qSlicerTablesReaderPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerTablesReader);
  Q_DISABLE_COPY(qSlicerTablesReader);
};

#endif
