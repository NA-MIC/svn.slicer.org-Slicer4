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

// Qt includes
#include <QDropEvent>
#include <QFileInfo>

// PythonQt includes
#include <PythonQt.h>

// SlicerQt includes
#include "qSlicerScriptedFileDialog.h"

// Python includes

//-----------------------------------------------------------------------------
class qSlicerScriptedFileDialogPrivate
{
public:
  typedef qSlicerScriptedFileDialogPrivate Self;
  qSlicerScriptedFileDialogPrivate();
  virtual ~qSlicerScriptedFileDialogPrivate();

  qSlicerIO::IOFileType FileType;
  QString Description;
  qSlicerFileDialog::IOAction Action;
  qSlicerIO::IOProperties Properties;
  mutable const QMimeData* MimeData;
  mutable bool MimeDataAccepted;
  QDropEvent* DropEvent;

  enum {
    ExecMethod = 0,
    IsMimeDataAcceptedMethod,
    DropEventMethod,
    };

  static int          APIMethodCount;
  static const char * APIMethodNames[3];

  PyObject*  PythonAPIMethods[3];
  PyObject*  PythonSelf;
  QString    PythonSource;
};

//-----------------------------------------------------------------------------
// qSlicerScriptedFileDialogPrivate methods

//---------------------------------------------------------------------------
int qSlicerScriptedFileDialogPrivate::APIMethodCount = 3;

//---------------------------------------------------------------------------
const char* qSlicerScriptedFileDialogPrivate::APIMethodNames[3] =
{
  "execDialog",
  "isMimeDataAccepted",
  "dropEvent"
};

//-----------------------------------------------------------------------------
qSlicerScriptedFileDialogPrivate::qSlicerScriptedFileDialogPrivate()
{
  this->Action = qSlicerFileDialog::Read;
  this->MimeData = 0;
  this->MimeDataAccepted = false;
  this->DropEvent = 0;

  this->PythonSelf = 0;
  for (int i = 0; i < Self::APIMethodCount; ++i)
    {
    this->PythonAPIMethods[i] = 0;
    }
}

//-----------------------------------------------------------------------------
qSlicerScriptedFileDialogPrivate::~qSlicerScriptedFileDialogPrivate()
{
  if (this->PythonSelf)
    {
    for (int i = 0; i < Self::APIMethodCount; ++i)
      {
      Py_XDECREF(this->PythonAPIMethods[i]);
      }
    Py_DECREF(this->PythonSelf);
    }
}

//-----------------------------------------------------------------------------
// qSlicerScriptedFileDialog methods

//-----------------------------------------------------------------------------
qSlicerScriptedFileDialog::qSlicerScriptedFileDialog(QObject* parent)
  : Superclass(parent)
  , d_ptr(new qSlicerScriptedFileDialogPrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerScriptedFileDialog::~qSlicerScriptedFileDialog()
{
}

//-----------------------------------------------------------------------------
QString qSlicerScriptedFileDialog::pythonSource()const
{
  Q_D(const qSlicerScriptedFileDialog);
  return d->PythonSource;
}

//-----------------------------------------------------------------------------
bool qSlicerScriptedFileDialog::setPythonSource(const QString& newPythonSource, const QString& className)
{
  Q_D(qSlicerScriptedFileDialog);

  if (!Py_IsInitialized())
    {
    return false;
    }

  Q_ASSERT(newPythonSource.endsWith(".py"));

  // Open the file
#ifdef HAVE_PYRUN_OPENFILE
  FILE* pyfile = PyRun_OpenFile(newPythonSource.toLatin1());
#else
  FILE* pyfile = fopen(newPythonSource.toLatin1(), "r");
#endif
  if (!pyfile)
    {
    PythonQt::self()->handleError();
    qCritical() << "setPythonSource - File" << newPythonSource << "doesn't exist !";
    return false;
    }

  // Extract moduleName from the provided filename
  QString classNameToLoad = className;
  if (classNameToLoad.isEmpty())
    {
    QString moduleName = QFileInfo(newPythonSource).baseName();
    classNameToLoad = moduleName;
    if (!moduleName.endsWith("FileDialog"))
      {
      classNameToLoad.append("FileDialog");
      }
    }

  // Get a reference to the main module and global dictionary
  PyObject * main_module = PyImport_AddModule("__main__");
  PyObject * global_dict = PyModule_GetDict(main_module);

  // Load class definition if needed
  PyObject * classToInstantiate = PyDict_GetItemString(global_dict, classNameToLoad.toLatin1());
  if (!classToInstantiate)
    {/*
    PyRun_File(pyfile, newPythonSource.toLatin1(), Py_file_input, global_dict, global_dict);
    classToInstantiate = PyDict_GetItemString(global_dict, classNameToLoad.toLatin1());
    }

  if (!classToInstantiate)
    {
    PythonQt::self()->handleError();
    PyErr_SetString(PyExc_RuntimeError,
                    QString("qSlicerScriptedFileDialog::setPythonSource - "
                            "Failed to load scripted pythonqt module class definition"
                            " %1 from %2").arg(classNameToLoad).arg(newPythonSource).toLatin1());
    */
    PythonQt::self()->handleError();
    return false;
    }

#ifdef HAVE_PYRUN_CLOSEFILE
  PyRun_CloseFile(pyfile);
#else
  fclose(pyfile);
#endif

  PyObject * wrappedThis = PythonQt::self()->priv()->wrapQObject(this);
  if (!wrappedThis)
    {
    PythonQt::self()->handleError();
    qCritical() << "qSlicerScriptedFileDialog::setPythonSource" << newPythonSource
        << "- Failed to wrap" << this->metaObject()->className();
    return false;
    }

  PyObject * arguments = PyTuple_New(1);
  PyTuple_SET_ITEM(arguments, 0, wrappedThis);

  // Attempt to instantiate the associated python class
  PyObject * self = PyInstance_New(classToInstantiate, arguments, 0);
  Py_DECREF(arguments);
  if (!self)
    {
    PythonQt::self()->handleError();
    qCritical() << "qSlicerScriptedFileDialog::setPythonSource" << newPythonSource
        << " - Failed to instantiate scripted pythonqt class"
        << classNameToLoad << classToInstantiate;
    return false;
    }

  // Retrieve API methods
  for (int i = 0; i < d->APIMethodCount; ++i)
    {
    Q_ASSERT(d->APIMethodNames[i]);
    if (!PyObject_HasAttrString(self, d->APIMethodNames[i]))
      {
      continue;
      }
    PyObject * method = PyObject_GetAttrString(self, d->APIMethodNames[i]);
    d->PythonAPIMethods[i] = method;
    }

  d->PythonSource = newPythonSource;
  d->PythonSelf = self;

  return true;
}

//-----------------------------------------------------------------------------
PyObject* qSlicerScriptedFileDialog::self() const
{
  Q_D(const qSlicerScriptedFileDialog);
  return d->PythonSelf;
}

//-----------------------------------------------------------------------------
bool qSlicerScriptedFileDialog::exec(const qSlicerIO::IOProperties& ioProperties)
{
  Q_D(qSlicerScriptedFileDialog);
  d->Properties = ioProperties;
  PyObject * method = d->PythonAPIMethods[d->ExecMethod];
  if (!method)
    {
    return false;
    }
  PythonQt::self()->clearError();
  PyObject_CallObject(method, 0);
  PythonQt::self()->handleError();
  return true;
}

//-----------------------------------------------------------------------------
bool qSlicerScriptedFileDialog::isMimeDataAccepted(const QMimeData* mimeData)const
{
  Q_D(const qSlicerScriptedFileDialog);
  d->MimeData = mimeData;
  d->MimeDataAccepted = false;
  PyObject * method = d->PythonAPIMethods[d->IsMimeDataAcceptedMethod];
  if (!method)
    {
    return d->MimeDataAccepted;
    }
  PythonQt::self()->clearError();
  PyObject_CallObject(method, 0);
  PythonQt::self()->handleError();
  return d->MimeDataAccepted;
}

//-----------------------------------------------------------------------------
void qSlicerScriptedFileDialog::dropEvent(QDropEvent* event)
{
  Q_D(qSlicerScriptedFileDialog);
  d->DropEvent =  event;
  d->MimeData = event->mimeData();
  PyObject * method = d->PythonAPIMethods[d->DropEventMethod];
  if (!method)
    {
    return;
    }
  PythonQt::self()->clearError();
  PyObject_CallObject(method, 0);
  PythonQt::self()->handleError();
}

//-----------------------------------------------------------------------------
const qSlicerIO::IOProperties& qSlicerScriptedFileDialog::ioProperties()const
{
  Q_D(const qSlicerScriptedFileDialog);
  return d->Properties;
}

//-----------------------------------------------------------------------------
const QMimeData* qSlicerScriptedFileDialog::mimeData()const
{
  Q_D(const qSlicerScriptedFileDialog);
  return d->MimeData;
}

//-----------------------------------------------------------------------------
QDropEvent* qSlicerScriptedFileDialog::dropEvent()const
{
  Q_D(const qSlicerScriptedFileDialog);
  return d->DropEvent;
}

//-----------------------------------------------------------------------------
void qSlicerScriptedFileDialog::acceptMimeData(bool accept)
{
  Q_D(qSlicerScriptedFileDialog);
  d->MimeDataAccepted = accept;
}
