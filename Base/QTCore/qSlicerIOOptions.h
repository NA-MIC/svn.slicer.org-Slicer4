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

#ifndef __qSlicerIOOptions_h
#define __qSlicerIOOptions_h

/// QtCore includes
#include "qSlicerIO.h"
#include "qSlicerBaseQTCoreExport.h"

class Q_SLICER_BASE_QTCORE_EXPORT qSlicerIOOptions
{
public:
  /// Contructor initialize empty properties
  explicit qSlicerIOOptions();
  virtual ~qSlicerIOOptions();

  const qSlicerIO::IOProperties& properties()const;
protected:
  qSlicerIO::IOProperties Properties;
};

#endif
