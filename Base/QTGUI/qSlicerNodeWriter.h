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

  This file was originally developed by Julien Finet, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

#ifndef __qSlicerNodeWriter_h
#define __qSlicerNodeWriter_h

// QtCore includes
#include "qSlicerBaseQTGUIExport.h"
#include "qSlicerFileWriter.h"
class qSlicerNodeWriterPrivate;
class vtkMRMLNode;

/// Utility class that is ready to use for most of the nodes.
class Q_SLICER_BASE_QTGUI_EXPORT qSlicerNodeWriter
  : public qSlicerFileWriter
{
  Q_OBJECT
  /// Some storage nodes don't support the compression option
  Q_PROPERTY(bool supportUseCompression READ supportUseCompression WRITE setSupportUseCompression);
public:
  typedef qSlicerFileWriter Superclass;
  qSlicerNodeWriter(const QString& description,
                    const qSlicerIO::IOFileType& fileType,
                    const QStringList& nodeTags,
                    bool useCompression = true,
                    QObject* parent = 0);

  virtual ~qSlicerNodeWriter();

  void setSupportUseCompression(bool useCompression);
  bool supportUseCompression()const;

  virtual QString description()const;
  virtual IOFileType fileType()const;

  /// Return true if the object is handled by the writer.
  virtual bool canWriteObject(vtkObject* object)const;

  /// Return  a list of the supported extensions for a particuliar object.
  /// Please read QFileDialog::nameFilters for the allowed formats
  /// Example: "Image (*.jpg *.png *.tiff)", "Model (*.vtk)"
  virtual QStringList extensions(vtkObject* object)const;

  virtual bool write(const qSlicerIO::IOProperties& properties);

  virtual vtkMRMLNode* getNodeByID(const char *id)const;

  /// Return a qSlicerIONodeWriterOptionsWidget
  virtual qSlicerIOOptions* options()const;

protected:
  void setNodeClassNames(const QStringList& nodeClassNames);
  QStringList nodeClassNames()const;

protected:
  QScopedPointer<qSlicerNodeWriterPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerNodeWriter);
  Q_DISABLE_COPY(qSlicerNodeWriter);
};

#endif
