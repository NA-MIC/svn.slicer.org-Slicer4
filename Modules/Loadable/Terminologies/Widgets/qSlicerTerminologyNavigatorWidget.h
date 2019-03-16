/*==============================================================================

  Program: 3D Slicer

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

#ifndef __qSlicerTerminologyNavigatorWidget_h
#define __qSlicerTerminologyNavigatorWidget_h

// MRMLWidgets includes
#include "qMRMLWidget.h"

// Terminologies includes
#include "qSlicerTerminologiesModuleWidgetsExport.h"

#include "vtkSlicerTerminologyEntry.h"

// CTK includes
#include <ctkPimpl.h>
#include <ctkVTKObject.h>

class qSlicerTerminologyNavigatorWidgetPrivate;

class QColor;
class QTableWidgetItem;
class QItemSelection;
class vtkSlicerTerminologyCategory;
class vtkSlicerTerminologyType;

/// \brief Qt widget for browsing a terminology dictionary.
///   DICOM properties of the selected entry can also be set if enabled.
/// \ingroup SlicerRt_QtModules_Terminologies_Widgets
class Q_SLICER_MODULE_TERMINOLOGIES_WIDGETS_EXPORT qSlicerTerminologyNavigatorWidget : public qMRMLWidget
{
  Q_OBJECT
  QVTK_OBJECT

  Q_PROPERTY(bool anatomicRegionSectionVisible READ anatomicRegionSectionVisible WRITE setAnatomicRegionSectionVisible)

  /// Roles set to the items in the terminology tables uniquely identifying the entries
  enum TerminologyItemDataRole
    {
    CodingSchemeDesignatorRole = Qt::UserRole + 100,
    CodeValueRole,
    // Roles for type items referencing category
    CategoryCodingSchemeDesignatorRole,
    CategoryCodeValueRole,
    CategoryCodeMeaningRole,
    // Last role (always keep last)
    LastTerminologyRole
    };

public:
  /// Constructor
  explicit qSlicerTerminologyNavigatorWidget(QWidget* parent = nullptr);
  /// Destructor
  ~qSlicerTerminologyNavigatorWidget() override;

  class TerminologyInfoBundle
  {
  public:
    TerminologyInfoBundle();
    TerminologyInfoBundle(vtkSlicerTerminologyEntry* entry, QString name, bool nameAutoGenerated, QColor color, bool colorAutoGenerated, QColor generatedColor);
    ~TerminologyInfoBundle();
    const TerminologyInfoBundle& operator=(const TerminologyInfoBundle& other);
    vtkSlicerTerminologyEntry* GetTerminologyEntry();

    QString Name;
    bool NameAutoGenerated;
    QColor Color;
    bool ColorAutoGenerated;
    QColor GeneratedColor;
  private:
    vtkSlicerTerminologyEntry* TerminologyEntry;
  };

  /// Get selection from widget: terminology and meta-data (name, color, auto-generated flags)
  /// \param terminologyInfo Info bundle to get terminology information into
  void terminologyInfo(TerminologyInfoBundle &terminologyInfo);
  /// Set selection to widget: terminology and meta-data (name, color, auto-generated flags)
  void setTerminologyInfo(TerminologyInfoBundle &terminologyInfo);

  /// Populate terminology entry from terminology and anatomy selection
  /// \return Success flag (e.g. fail if no type is selected)
  bool terminologyEntry(vtkSlicerTerminologyEntry* entry);
  /// Update terminology and anatomy widgets and selections from terminology entry
  /// \return Success flag (e.g. fail if no type is specified in entry)
  bool setTerminologyEntry(vtkSlicerTerminologyEntry* entry);

  /// Get whether anatomic region section are visible
  bool anatomicRegionSectionVisible() const;

  /// Generate name for given terminology
  Q_INVOKABLE static QString nameFromTerminology(vtkSlicerTerminologyEntry* entry);
  /// Generate name for current terminology
  QString nameFromCurrentTerminology();
  /// Get recommended color from type (or type modifier if any) of the given terminology entry
  Q_INVOKABLE static QColor recommendedColorFromTerminology(vtkSlicerTerminologyEntry* entry);
  /// Get recommended color from type (or type modifier if any) of the current terminology in the widget
  QColor recommendedColorFromCurrentTerminology();

public slots:
  /// Show/hide anatomic region section section
  void setAnatomicRegionSectionVisible(bool);

protected:
  /// Set current terminology to widget
  void setCurrentTerminology(QString terminologyName);
  /// Set current category to widget.
  /// Only used when setting the category from a given entry to the widget!
  /// \return Flag indicating whether the given category was found in the category table
  bool setCurrentCategory(vtkSlicerTerminologyCategory* category);
  /// Update widget UI based on the current category selection
  void updateWidgetFromCurrentCategory();
  /// Set current type to widget
  /// \return Flag indicating whether the given type was found in the type table
  bool setCurrentType(vtkSlicerTerminologyType* type);
  /// Set current type modifier to widget
  /// \return Flag indicating whether the given modifier was found in the combobox
  bool setCurrentTypeModifier(vtkSlicerTerminologyType* modifier);
  /// Set current anatomic context to widget
  void setCurrentAnatomicContext(QString contextName);
  /// Set current region to widget
  /// \return Flag indicating whether the given region was found in the region table
  bool setCurrentRegion(vtkSlicerTerminologyType* region);
  /// Set current region modifier to widget
  /// \return Flag indicating whether the given modifier was found in the combobox
  bool setCurrentRegionModifier(vtkSlicerTerminologyType* modifier);

protected:
  /// Populate terminology combobox based on current selection
  void populateTerminologyComboBox();
  /// Populate category table based on selected terminology and category search term
  void populateCategoryTable();
  /// Populate type table based on selected category and type search term
  void populateTypeTable();
  /// Populate type modifier combobox based on current selection
  void populateTypeModifierComboBox();

  /// Populate anatomic region context combobox based on current selection
  void populateAnatomicContextComboBox();
  /// Populate region table based on selected anatomic region context and type search term
  void populateRegionTable();
  /// Populate region modifier combobox based on current selection
  void populateRegionModifierComboBox();

  /// Copy terminology or anatomic context file to user folder
  void copyContextToUserDirectory(QString filePath);

protected slots:
  void onTerminologySelectionChanged(int);
  void onCategorySelectionChanged();
  void onTypeSelected(QTableWidgetItem*, QTableWidgetItem*);
  void onTypeModifierSelectionChanged(int);
  void onCategorySearchTextChanged(QString);
  void onTypeSearchTextChanged(QString);

  void onAnatomicContextSelectionChanged(int);
  void onRegionSelected(QTableWidgetItem*,QTableWidgetItem*);
  void onRegionModifierSelectionChanged(int);
  void onRegionSearchTextChanged(QString);

  void onNameChanged(QString);
  void onResetNameClicked();
  void onColorChanged(QColor);
  void onResetColorClicked();

  void onLoadTerminologyClicked();
  void onLoadAnatomicContextClicked();

  void anatomicalRegionExpandButtonUp();
  void anatomicalRegionExpandButtonDown();

  void onLogicModified();

signals:
  /// Emitted when selection becomes valid (true argument) or invalid (false argument)
  void selectionValidityChanged(bool);

protected:
  QScopedPointer<qSlicerTerminologyNavigatorWidgetPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(qSlicerTerminologyNavigatorWidget);
  Q_DISABLE_COPY(qSlicerTerminologyNavigatorWidget);
};

#endif
