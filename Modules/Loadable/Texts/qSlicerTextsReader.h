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

  This file was originally developed by Kyle Sunderland, PerkLab, Queen's University
  and was supported through CANARIE's Research Software Program, and Cancer
  Care Ontario.

==============================================================================*/

#ifndef __qSlicerTextsReader_h
#define __qSlicerTextsReader_h

// SlicerQt includes
#include "qSlicerFileReader.h"
class qSlicerTextsReaderPrivate;

//-----------------------------------------------------------------------------
class qSlicerTextsReader : public qSlicerFileReader
{
  Q_OBJECT
public:
  typedef qSlicerFileReader Superclass;
  qSlicerTextsReader(QObject* parent = 0);
  virtual ~qSlicerTextsReader();

  virtual QString description() const;
  virtual IOFileType fileType() const;
  virtual QStringList extensions() const;

  virtual bool load(const IOProperties& properties);

protected:
  QScopedPointer< qSlicerTextsReaderPrivate > d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerTextsReader);
  Q_DISABLE_COPY(qSlicerTextsReader);
};

#endif
