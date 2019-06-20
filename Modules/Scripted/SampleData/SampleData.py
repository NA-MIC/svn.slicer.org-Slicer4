from __future__ import print_function
import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import computeChecksum, extractAlgoAndDigest
import logging
import textwrap

#
# SampleData methods
#

def downloadFromURL(uris=None, fileNames=None, nodeNames=None, checksums=None, loadFiles=None,
  customDownloader=None, loadFileTypes=None, loadFileProperties={}):
  """Download and optionally load data into the application.

  :param uris: Download URL(s).
  :param fileNames: File name(s) that will be downloaded (and loaded).
  :param nodeNames: Node name(s) in the scene.
  :param checksums: Checksum(s) formatted as ``<algo>:<digest>`` to verify the downloaded file(s). For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
  :param loadFiles: Boolean indicating if file(s) should be loaded. By default, the function decides.
  :param customDownloader: Custom function for downloading.
  :param loadFileTypes: file format name(s) ('VolumeFile' by default).
  :param loadFileProperties: custom properties passed to the IO plugin.

  If the given ``fileNames`` are not found in the application cache directory, they
  are downloaded using the associated URIs.
  See ``slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()``

  If not explicitly provided or if set to ``None``, the ``loadFileTypes`` are
  guessed based on the corresponding filename extensions.

  If a given fileName has the ``.mrb`` or ``.mrml`` extension, it will **not** be loaded
  by default. To ensure the file is loaded, ``loadFiles`` must be set.

  The ``loadFileProperties`` are common for all files. If different properties
  need to be associated with files of different types, downloadFromURL must
  be called for each.
  """
  return SampleDataLogic().downloadFromURL(
    uris, fileNames, nodeNames, checksums, loadFiles, customDownloader, loadFileTypes, loadFileProperties)


def downloadSample(sampleName):
  """For a given sample name this will search the available sources
  and load it if it is available.  Returns the first loaded node."""
  return SampleDataLogic().downloadSamples(sampleName)[0]


def downloadSamples(sampleName):
  """For a given sample name this will search the available sources
  and load it if it is available.  Returns the loaded nodes."""
  return SampleDataLogic().downloadSamples(sampleName)

#
# SampleData
#

class SampleData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Sample Data"
    self.parent.categories = ["Informatics"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics), Benjamin Long (Kitware), Jean-Christophe Fillion-Robin (Kitware)"]
    self.parent.helpText = """
The SampleData module can be used to download data for working with in slicer.  Use of this module requires an active network connection.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
<p>This work was was funded by Cancer Care Ontario
and the Ontario Consortium for Adaptive Interventions in Radiation Oncology (OCAIRO)</p>

<p>CTA abdomen (Panoramix) dataset comes from <a href="http://www.osirix-viewer.com/resources/dicom-image-library/">Osirix DICOM image library</a>
and is exclusively available for research and teaching. You are not authorized to redistribute or sell it, or
use it for commercial purposes.</p>
"""

    if slicer.mrmlScene.GetTagByClassName( "vtkMRMLScriptedModuleNode" ) != 'ScriptedModule':
      slicer.mrmlScene.RegisterNodeClass(vtkMRMLScriptedModuleNode())

    # Trigger the menu to be added when application has started up
    if not slicer.app.commandOptions().noMainWindow :
      slicer.app.connect("startupCompleted()", self.addMenu)

    # allow other modules to register sample data sources by appending
    # instances or subclasses SampleDataSource objects on this list
    try:
      slicer.modules.sampleDataSources
    except AttributeError:
      slicer.modules.sampleDataSources = {}


  def addMenu(self):
    actionIcon = self.parent.icon
    a = qt.QAction(actionIcon, 'Download Sample Data', slicer.util.mainWindow())
    a.setToolTip('Go to the SampleData module to download data from the network')
    a.connect('triggered()', self.select)

    fileMenu = slicer.util.lookupTopLevelWidget('FileMenu')
    if fileMenu:
      for action in fileMenu.actions():
        if action.text == 'Save':
          fileMenu.insertAction(action,a)


  def select(self):
    m = slicer.util.mainWindow()
    m.moduleSelector().selectModule('SampleData')

#
# SampleDataSource
#
class SampleDataSource(object):
  """Describe a set of sample data associated with one or multiple URIs and filenames.

  Example::

    import SampleData
    dataSource = SampleData.SampleDataSource(
      nodeNames='fixed',
      fileNames='fixed.nrrd',
      uris='http://slicer.kitware.com/midas3/download/item/157188/small-mr-eye-fixed.nrrd')
    loadedNode = SampleData.SampleDataLogic().downloadFromSource(dataSource)[0]
  """

  def __init__(self, sampleName=None, sampleDescription=None, uris=None, fileNames=None, nodeNames=None,
    checksums=None, loadFiles=None,
    customDownloader=None, thumbnailFileName=None,
    loadFileType=None, loadFileProperties={}):
    """
    :param sampleName: Name identifying the data set.
    :param sampleDescription: Displayed name of data set in SampleData module GUI. (default is ``sampleName``)
    :param thumbnailFileName: Displayed thumbnail of data set in SampleData module GUI,
    :param uris: Download URL(s).
    :param fileNames: File name(s) that will be downloaded (and loaded).
    :param nodeNames: Node name(s) in the scene.
    :param checksums: Checksum(s) formatted as ``<algo>:<digest>`` to verify the downloaded file(s). For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
    :param loadFiles: Boolean indicating if file(s) should be loaded.
    :param customDownloader: Custom function for downloading.
    :param loadFileType: file format name(s) ('VolumeFile' by default if node name is specified).
    :param loadFileProperties: custom properties passed to the IO plugin.
    """
    self.sampleName = sampleName
    if sampleDescription is None:
      sampleDescription = sampleName
    self.sampleDescription = sampleDescription
    if (isinstance(uris, list) or isinstance(uris, tuple)):
      if isinstance(loadFileType, str) or loadFileType is None:
        loadFileType = [loadFileType] * len(uris)
      if nodeNames is None:
        nodeNames = [None] * len(uris)
      if loadFiles is None:
        loadFiles = [None] * len(uris)
      if checksums is None:
        checksums = [None] * len(uris)
    elif isinstance(uris, str):
      uris = [uris,]
      fileNames = [fileNames,]
      nodeNames = [nodeNames,]
      loadFiles = [loadFiles,]
      loadFileType = [loadFileType,]
      checksums = [checksums,]

    updatedFileType = []
    for fileName, nodeName, fileType in zip(fileNames, nodeNames, loadFileType):
      # If not explicitly specified, attempt to guess fileType
      if fileType is None:
        if nodeName is not None:
          # TODO: Use method from Slicer IO logic ?
          fileType = "VolumeFile"
        else:
          ext = os.path.splitext(fileName.lower())[1]
          if ext in [".mrml", ".mrb"]:
            fileType = "SceneFile"
          elif ext in [".zip"]:
            fileType = "ZipFile"
      updatedFileType.append(fileType)

    self.uris = uris
    self.fileNames = fileNames
    self.nodeNames = nodeNames
    self.loadFiles = loadFiles
    self.customDownloader = customDownloader
    self.thumbnailFileName = thumbnailFileName
    self.loadFileType = updatedFileType
    self.loadFileProperties = loadFileProperties
    self.checksums = checksums
    if not len(uris) == len(fileNames) == len(nodeNames) == len(loadFiles) == len(updatedFileType) == len(checksums):
      raise ValueError(
        f"All fields of sample data source must have the same length\n"
        f"  uris                 : {uris}\n"
        f"  len(uris)            : {len(uris)}\n"
        f"  len(fileNames)       : {len(fileNames)}\n"
        f"  len(nodeNames)       : {len(nodeNames)}\n"
        f"  len(loadFiles)       : {len(loadFiles)}\n"
        f"  len(updatedFileType) : {len(updatedFileType)}\n"
        f"  len(checksums)       : {len(checksums)}\n"
        )

  def __str__(self):
    output = [
      "sampleName        : %s" % self.sampleName,
      "sampleDescription : %s" % self.sampleDescription,
      "thumbnailFileName : %s" % self.thumbnailFileName,
      "loadFileProperties: %s" % self.loadFileProperties,
      "customDownloader  : %s" % self.customDownloader,
      ""
    ]
    for fileName, uri, nodeName, loadFile, fileType, checksum in zip(self.fileNames, self.uris, self.nodeNames, self.loadFiles, self.loadFileType, self.checksums):
      output.extend([
        " fileName    : %s" % fileName,
        " uri         : %s" % uri,
        " checksum    : %s" % checksum,
        " nodeName    : %s" % nodeName,
        " loadFile    : %s" % loadFile,
        " loadFileType: %s" % fileType,
        ""
      ])
    return "\n".join(output)


#
# SampleData widget
#

class SampleDataWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # This module is often used in developer mode, therefore
    # collapse reload & test section by default.
    if hasattr(self, "reloadCollapsibleButton"):
      self.reloadCollapsibleButton.collapsed = True

    self.observerTags = []
    self.logic = SampleDataLogic(self.logMessage)

    numberOfColumns = 3
    iconPath = os.path.join(os.path.dirname(__file__).replace('\\','/'), 'Resources','Icons')
    desktop = qt.QDesktopWidget()
    mainScreenSize = desktop.availableGeometry(desktop.primaryScreen)
    iconSize = qt.QSize(int(mainScreenSize.width()/15),int(mainScreenSize.height()/10))

    categories = sorted(slicer.modules.sampleDataSources.keys())
    if self.logic.builtInCategoryName in categories:
      categories.remove(self.logic.builtInCategoryName)
    categories.insert(0,self.logic.builtInCategoryName)
    for category in categories:
      if category == self.logic.developmentCategoryName and self.developerMode is False:
        continue
      frame = ctk.ctkCollapsibleGroupBox(self.parent)
      self.layout.addWidget(frame)
      frame.title = category
      frame.name = '%sCollapsibleGroupBox' % category
      layout = qt.QGridLayout(frame)
      columnIndex = 0
      rowIndex = 0
      for source in slicer.modules.sampleDataSources[category]:
        name = source.sampleDescription
        if not name:
          name = source.nodeNames[0]

        b = qt.QToolButton()
        b.setText(name)

        # Set thumbnail
        if source.thumbnailFileName:
          # Thumbnail provided
          thumbnailImage = source.thumbnailFileName
        else:
          # Look for thumbnail image with the name of any node name with .png extension
          thumbnailImage = None
          for nodeName in source.nodeNames:
            if not nodeName:
              continue
            thumbnailImageAttempt = os.path.join(iconPath, nodeName+'.png')
            if os.path.exists(thumbnailImageAttempt):
              thumbnailImage = thumbnailImageAttempt
              break
        if thumbnailImage and os.path.exists(thumbnailImage):
          b.setIcon(qt.QIcon(thumbnailImage))

        b.setIconSize(iconSize)
        b.setToolButtonStyle(qt.Qt.ToolButtonTextUnderIcon)
        qSize = qt.QSizePolicy()
        qSize.setHorizontalPolicy(qt.QSizePolicy.Expanding)
        b.setSizePolicy(qSize)

        b.name = '%sPushButton' % name
        layout.addWidget(b, rowIndex, columnIndex)
        columnIndex += 1
        if columnIndex==numberOfColumns:
          rowIndex += 1
          columnIndex = 0
        if source.customDownloader:
          b.connect('clicked()', lambda s=source: s.customDownloader(s))
        else:
          b.connect('clicked()', lambda s=source: self.logic.downloadFromSource(s))

    self.log = qt.QTextEdit()
    self.log.readOnly = True
    self.layout.addWidget(self.log)
    self.logMessage('<p>Status: <i>Idle</i>')

    # Add spacer to layout
    self.layout.addStretch(1)

  def logMessage(self, message, logLevel=logging.INFO):
    # Set text color based on log level
    if logLevel >= logging.ERROR:
      message = '<font color="red">' + message + '</font>'
    elif logLevel >= logging.WARNING:
      message = '<font color="orange">' + message + '</font>'
    # Show message in status bar
    doc = qt.QTextDocument()
    doc.setHtml(message)
    slicer.util.showStatusMessage(doc.toPlainText(),3000)
    # Show message in log window at the bottom of the module widget
    self.log.insertHtml(message)
    self.log.insertPlainText('\n')
    self.log.ensureCursorVisible()
    self.log.repaint()
    logging.log(logLevel, message)
    slicer.app.processEvents(qt.QEventLoop.ExcludeUserInputEvents)

  def isCategoryVisible(self, category):
    """Check the visibility of a SampleData category given its name.

    Returns False if the category is not visible or if it does not exist,
    otherwise returns True.
    """
    if not self.isSampleDataSourceRegistered(category):
      return False
    return slicer.util.findChild(self.parent, '%sCollapsibleGroupBox' % category).isVisible()

  def setCategoryVisible(self, category, visible):
    """Update visibility of a SampleData category given its name.

    The function is a no-op if the category does not exist.
    """
    if not self.isSampleDataSourceRegistered(category):
      return
    slicer.util.findChild(self.parent, '%sCollapsibleGroupBox' % category).setVisible(visible)

  def isSampleDataSourceRegistered(self, category):
    """Check if a SampleDataSource is registered given its category name.
    """
    return category in slicer.modules.sampleDataSources.keys()

#
# SampleData logic
#

class SampleDataLogic(object):
  """Manage the slicer.modules.sampleDataSources dictionary.
  The dictionary keys are categories of sample data sources.
  The BuiltIn category is managed here.  Modules or extensions can
  register their own sample data by creating instances of the
  SampleDataSource class.  These instances should be stored in a
  list that is assigned to a category following the model
  used in registerBuiltInSampleDataSources below.

  Checksums are expected to be formatted as a string of the form
  ``<algo>:<digest>``. For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
  """

  @staticmethod
  def registerCustomSampleDataSource(category='Custom',
    sampleName=None, uris=None, fileNames=None, nodeNames=None,
    customDownloader=None, thumbnailFileName=None,
    loadFileType='VolumeFile', loadFiles=None, loadFileProperties={},
    checksums=None):
    """Adds custom data sets to SampleData.
    :param category: Section title of data set in SampleData module GUI.
    :param sampleName: Displayed name of data set in SampleData module GUI.
    :param thumbnailFileName: Displayed thumbnail of data set in SampleData module GUI,
    :param uris: Download URL(s).
    :param fileNames: File name(s) that will be loaded.
    :param nodeNames: Node name(s) in the scene.
    :param customDownloader: Custom function for downloading.
    :param loadFileType: file format name(s) ('VolumeFile' by default).
    :param loadFiles: Boolean indicating if file(s) should be loaded. By default, the function decides.
    :param loadFileProperties: custom properties passed to the IO plugin.
    :param checksums: Checksum(s) formatted as ``<algo>:<digest>`` to verify the downloaded file(s). For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
    """

    try:
      slicer.modules.sampleDataSources
    except AttributeError:
      slicer.modules.sampleDataSources = {}

    if category not in slicer.modules.sampleDataSources:
      slicer.modules.sampleDataSources[category] = []

    slicer.modules.sampleDataSources[category].append(SampleDataSource(
      sampleName=sampleName,
      uris=uris,
      fileNames=fileNames,
      nodeNames=nodeNames,
      thumbnailFileName=thumbnailFileName,
      loadFileType=loadFileType,
      loadFiles=loadFiles,
      loadFileProperties=loadFileProperties,
      checksums=checksums,
      customDownloader=customDownloader,
      ))

  def __init__(self, logMessage=None):
    if logMessage:
      self.logMessage = logMessage
    self.builtInCategoryName = 'BuiltIn'
    self.developmentCategoryName = 'Development'
    self.registerBuiltInSampleDataSources()
    self.registerDevelopmentSampleDataSources()
    if slicer.app.testingEnabled():
      self.registerTestingDataSources()

  def registerBuiltInSampleDataSources(self):
    """Fills in the pre-define sample data sources"""
    sourceArguments = (
        ('MRHead', None, 'http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', 'MR-head.nrrd', 'MRHead', 'SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93'),
        ('CTChest', None, 'http://slicer.kitware.com/midas3/download/item/292307/CT-chest.nrrd', 'CT-chest.nrrd', 'CTChest', 'SHA256:4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e'),
        ('CTACardio', None, 'http://slicer.kitware.com/midas3/download/item/292309/CTA-cardio.nrrd', 'CTA-cardio.nrrd', 'CTACardio', 'SHA256:3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2'),
        ('DTIBrain', None, 'http://slicer.kitware.com/midas3/download/item/292310/DTI-brain.nrrd', 'DTI-Brain.nrrd', 'DTIBrain', 'SHA256:5c78d00c86ae8d968caa7a49b870ef8e1c04525b1abc53845751d8bce1f0b91a'),
        ('MRBrainTumor1', None, 'http://slicer.kitware.com/midas3/download/item/292312/RegLib_C01_1.nrrd', 'RegLib_C01_1.nrrd', 'MRBrainTumor1', 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95'),
        ('MRBrainTumor2', None, 'http://slicer.kitware.com/midas3/download/item/292313/RegLib_C01_2.nrrd', 'RegLib_C01_2.nrrd', 'MRBrainTumor2', 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97'),
        ('BaselineVolume', None, 'http://slicer.kitware.com/midas3/download/?items=2009,1', 'BaselineVolume.nrrd', 'BaselineVolume', 'SHA256:dff28a7711d20b6e16d5416535f6010eb99fd0c8468aaa39be4e39da78e93ec2'),
        ('DTIVolume', None,
          ('http://slicer.kitware.com/midas3/download/?items=2011,1',
            'http://slicer.kitware.com/midas3/download/?items=2010,1', ),
          ('DTIVolume.raw.gz', 'DTIVolume.nhdr'), (None, 'DTIVolume'),
          ('SHA256:d785837276758ddd9d21d76a3694e7fd866505a05bc305793517774c117cb38d', 'SHA256:67564aa42c7e2eec5c3fd68afb5a910e9eab837b61da780933716a3b922e50fe')),
        ('DWIVolume', None,
          ('http://slicer.kitware.com/midas3/download/?items=2142,1', 'http://slicer.kitware.com/midas3/download/?items=2141,1'),
          ('dwi.raw.gz', 'dwi.nhdr'), (None, 'dwi'),
          ('SHA256:cf03fd53583dc05120d3314d0a82bdf5946799b1f72f2a7f08963f3fd24ca692', 'SHA256:7666d83bc205382e418444ea60ab7df6dba6a0bd684933df8809da6b476b0fed')),
        ('CTAAbdomenPanoramix', 'CTA abdomen\n(Panoramix)', 'http://slicer.kitware.com/midas3/download/?items=9073,1', 'Panoramix-cropped.nrrd', 'Panoramix-cropped', 'SHA256:146af87511520c500a3706b7b2bfb545f40d5d04dd180be3a7a2c6940e447433'),
        ('CBCTDentalSurgery', None,
          ('http://slicer.kitware.com/midas3/download/item/94510/Greyscale_presurg.gipl.gz',
            'http://slicer.kitware.com/midas3/download/item/94509/Greyscale_postsurg.gipl.gz',),
          ('PreDentalSurgery.gipl.gz', 'PostDentalSurgery.gipl.gz'), ('PreDentalSurgery', 'PostDentalSurgery'),
          ('SHA256:7bfa16945629c319a439f414cfb7edddd2a97ba97753e12eede3b56a0eb09968', 'SHA256:4cdc3dc35519bb57daeef4e5df89c00849750e778809e94971d3876f95cc7bbd')),
        ('MRUSProstate', 'MR-US Prostate',
          ('http://slicer.kitware.com/midas3/download/item/142475/Case10-MR.nrrd',
            'http://slicer.kitware.com/midas3/download/item/142476/case10_US_resampled.nrrd',),
          ('Case10-MR.nrrd', 'case10_US_resampled.nrrd'), ('MRProstate', 'USProstate'),
          ('SHA256:4843cdc9ea5d7bcce61650d1492ce01035727c892019339dca726380496896aa', 'SHA256:34decf58b1e6794069acbe947b460252262fe95b6858c5e320aeab03bc82ebb2')),
        ('CTMRBrain', 'CT-MR Brain',
          ('http://slicer.kitware.com/midas3/download/item/284192/CTBrain.nrrd',
           'http://slicer.kitware.com/midas3/download/item/330508/MRBrainT1.nrrd',
           'http://slicer.kitware.com/midas3/download/item/330509/MRBrainT2.nrrd',),
          ('CT-brain.nrrd', 'MR-brain-T1.nrrd', 'MR-brain-T2.nrrd'),
          ('CTBrain', 'MRBrainT1', 'MRBrainT2'),
          ('SHA256:6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1',
           'SHA256:2da3f655ed20356ee8cdf32aa0f8f9420385de4b6e407d28e67f9974d7ce1593',
           'SHA256:fa1fe5910a69182f2b03c0150d8151ac6c75df986449fb5a6c5ae67141e0f5e7')),
        )

    if self.builtInCategoryName not in slicer.modules.sampleDataSources:
      slicer.modules.sampleDataSources[self.builtInCategoryName] = []
    for sourceArgument in sourceArguments:
      slicer.modules.sampleDataSources[self.builtInCategoryName].append(SampleDataSource(*sourceArgument))

  def registerDevelopmentSampleDataSources(self):
    """Fills in the sample data sources displayed only if developer mode is enabled."""
    iconPath = os.path.join(os.path.dirname(__file__).replace('\\','/'), 'Resources','Icons')
    self.registerCustomSampleDataSource(
      category=self.developmentCategoryName, sampleName='TinyPatient',
      uris=['http://slicer.kitware.com/midas3/download/item/245205/TinyPatient_CT.nrrd',
            'http://slicer.kitware.com/midas3/download/item/367020/TinyPatient_Structures.seg.nrrd'],
      fileNames=['TinyPatient_CT.nrrd', 'TinyPatient_Structures.seg.nrrd'],
      nodeNames=['TinyPatient_CT', 'TinyPatient_Segments'],
      thumbnailFileName=os.path.join(iconPath, 'TinyPatient.png'),
      loadFileType=['VolumeFile', 'SegmentationFile'],
      checksums=['SHA256:c0743772587e2dd4c97d4e894f5486f7a9a202049c8575e032114c0a5c935c3b', 'SHA256:3243b62bde36b1db1cdbfe204785bd4bc1fbb772558d5f8cac964cda8385d470']
      )

  def registerTestingDataSources(self):
    """Register sample data sources used by SampleData self-test to test module functionalities."""
    self.registerCustomSampleDataSource(**SampleDataTest.CustomDownloaderDataSource)

  def downloadFileIntoCache(self, uri, name, checksum=None):
    """Given a uri and and a filename, download the data into
    a file of the given name in the scene's cache"""
    destFolderPath = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()

    if not os.access(destFolderPath, os.W_OK):
      try:
        os.mkdir(destFolderPath)
      except:
        self.logMessage('<b>Failed to create cache folder %s</b>' % destFolderPath, logging.ERROR)
      if not os.access(destFolderPath, os.W_OK):
        self.logMessage('<b>Cache folder %s is not writable</b>' % destFolderPath, logging.ERROR)
    return self.downloadFile(uri, destFolderPath, name, checksum)

  def downloadSourceIntoCache(self, source):
    """Download all files for the given source and return a
    list of file paths for the results"""
    filePaths = []
    for uri,fileName,checksum in zip(source.uris,source.fileNames,source.checksums):
      filePaths.append(self.downloadFileIntoCache(uri, fileName, checksum))
    return filePaths

  def downloadFromSource(self,source,attemptCount=0):
    """Given an instance of SampleDataSource, downloads the associated data and
    load them into Slicer if it applies.

    The function always returns a list.

    Based on the fileType(s), nodeName(s) and loadFile(s) associated with
    the source, different values may be appended to the returned list:

      - if nodeName is specified, appends loaded nodes but if ``loadFile`` is False appends downloaded filepath
      - if fileType is ``SceneFile``, appends downloaded filepath
      - if fileType is ``ZipFile``, appends directory of extracted archive but if ``loadFile`` is False appends downloaded filepath

    If no ``nodeNames`` and no ``fileTypes`` are specified or if ``loadFiles`` are all False,
    returns the list of all downloaded filepaths.
    """
    nodes = []
    filePaths = []

    for uri,fileName,nodeName,checksum,loadFile,loadFileType in zip(source.uris,source.fileNames,source.nodeNames,source.checksums,source.loadFiles,source.loadFileType):

      current_source = SampleDataSource(uris=uri, fileNames=fileName, nodeNames=nodeName, checksums=checksum, loadFiles=loadFile, loadFileType=loadFileType, loadFileProperties=source.loadFileProperties)
      try:
        filePath = self.downloadFileIntoCache(uri, fileName, checksum)
      except ValueError:
        if attemptCount < 5:
          attemptCount += 1
          self.logMessage('<b>Load failed! Trying to download again (%d of 5 attempts)...</b>' % (attemptCount), logging.ERROR)
          filePath = self.downloadFileIntoCache(uri, fileName, checksum)

      filePaths.append(filePath)

      if loadFileType == 'ZipFile':
        if loadFile == False:
          nodes.append(filePath)
          continue
        outputDir = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory() + "/" + os.path.splitext(os.path.basename(filePath))[0]
        qt.QDir().mkpath(outputDir)
        success = slicer.util.extractArchive(filePath, outputDir)
        if not success and attemptCount < 5:
          file = qt.QFile(filePath)
          if not file.remove():
            self.logMessage('<b>Load failed! Unable to delete and try again loading %s!</b>' % filePath, logging.ERROR)
            nodes.append(None)
            break
          attemptCount += 1
          self.logMessage('<b>Load failed! Trying to download again (%d of 5 attempts)...</b>' % (attemptCount), logging.ERROR)
          outputDir = self.downloadFromSource(current_source,attemptCount)[0]
        nodes.append(outputDir)

      elif loadFileType == 'SceneFile':
        if not loadFile:
          nodes.append(filePath)
          continue
        success = self.loadScene(filePath, source.loadFileProperties)
        if not success and attemptCount < 5:
          file = qt.QFile(filePath)
          if not file.remove():
            self.logMessage('<b>Load failed! Unable to delete and try again loading %s!</b>' % filePath, logging.ERROR)
            nodes.append(None)
            break
          attemptCount += 1
          self.logMessage('<b>Load failed! Trying to download again (%d of 5 attempts)...</b>' % (attemptCount), logging.ERROR)
          filePath = self.downloadFromSource(current_source,attemptCount)[0]
        nodes.append(filePath)

      elif nodeName:
        if loadFile == False:
          nodes.append(filePath)
          continue
        loadedNode = self.loadNode(filePath, nodeName, loadFileType, source.loadFileProperties)
        if loadedNode is None and attemptCount < 5:
          file = qt.QFile(filePath)
          if not file.remove():
            self.logMessage('<b>Load failed! Unable to delete and try again loading %s!</b>' % filePath, logging.ERROR)
            nodes.append(None)
            break
          attemptCount += 1
          self.logMessage('<b>Load failed! Trying to download again (%d of 5 attempts)...</b>' % (attemptCount), logging.ERROR)
          loadedNode = self.downloadFromSource(current_source,attemptCount)[0]
        nodes.append(loadedNode)

    if nodes:
      return nodes
    else:
      return filePaths

  def sourceForSampleName(self,sampleName):
    """For a given sample name this will search the available sources.
    Returns SampleDataSource instance."""
    for category in slicer.modules.sampleDataSources.keys():
      for source in slicer.modules.sampleDataSources[category]:
        if sampleName == source.sampleName:
          return source
    return None

  def downloadFromURL(self, uris=None, fileNames=None, nodeNames=None, checksums=None, loadFiles=None,
    customDownloader=None, loadFileTypes=None, loadFileProperties={}):
    """Download and optionally load data into the application.

    :param uris: Download URL(s).
    :param fileNames: File name(s) that will be downloaded (and loaded).
    :param nodeNames: Node name(s) in the scene.
    :param checksums: Checksum(s) formatted as ``<algo>:<digest>`` to verify the downloaded file(s). For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
    :param loadFiles: Boolean indicating if file(s) should be loaded. By default, the function decides.
    :param customDownloader: Custom function for downloading.
    :param loadFileTypes: file format name(s) ('VolumeFile' by default).
    :param loadFileProperties: custom properties passed to the IO plugin.

    If the given ``fileNames`` are not found in the application cache directory, they
    are downloaded using the associated URIs.
    See ``slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()``

    If not explicitly provided or if set to ``None``, the ``loadFileTypes`` are
    guessed based on the corresponding filename extensions.

    If a given fileName has the ``.mrb`` or ``.mrml`` extension, it will **not** be loaded
    by default. To ensure the file is loaded, ``loadFiles`` must be set.

    The ``loadFileProperties`` are common for all files. If different properties
    need to be associated with files of different types, downloadFromURL must
    be called for each.
    """
    return self.downloadFromSource(SampleDataSource(
      uris=uris, fileNames=fileNames, nodeNames=nodeNames, loadFiles=loadFiles,
      loadFileType=loadFileTypes, loadFileProperties=loadFileProperties, checksums=checksums
    ))

  def downloadSample(self,sampleName):
    """For a given sample name this will search the available sources
    and load it if it is available.  Returns the first loaded node."""
    return self.downloadSamples(sampleName)[0]

  def downloadSamples(self,sampleName):
    """For a given sample name this will search the available sources
    and load it if it is available.  Returns the loaded nodes."""
    source = self.sourceForSampleName(sampleName)
    nodes = []
    if source:
      nodes = self.downloadFromSource(source)
    return nodes

  def logMessage(self,message,logLevel=logging.INFO):
    print(message)

  """Utility methods for backwards compatibility"""
  def downloadMRHead(self):
    return self.downloadSample('MRHead')

  def downloadCTChest(self):
    return self.downloadSample('CTChest')

  def downloadCTACardio(self):
    return self.downloadSample('CTACardio')

  def downloadDTIBrain(self):
    return self.downloadSample('DTIBrain')

  def downloadMRBrainTumor1(self):
    return self.downloadSample('MRBrainTumor1')

  def downloadMRBrainTumor2(self):
    return self.downloadSample('MRBrainTumor2')

  def downloadWhiteMatterExplorationBaselineVolume(self):
    return self.downloadSample('BaselineVolume')

  def downloadWhiteMatterExplorationDTIVolume(self):
    return self.downloadSample('DTIVolume')

  def downloadDiffusionMRIDWIVolume(self):
    return self.downloadSample('DWIVolume')

  def downloadAbdominalCTVolume(self):
    return self.downloadSample('CTAAbdomenPanoramix')

  def downloadDentalSurgery(self):
    # returns list since that's what earlier method did
    return self.downloadSamples('CBCTDentalSurgery')

  def downloadMRUSPostate(self):
    # returns list since that's what earlier method did
    return self.downloadSamples('MRUSProstate')

  def humanFormatSize(self,size):
    """ from http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size"""
    for x in ['bytes','KB','MB','GB']:
      if size < 1024.0 and size > -1024.0:
        return "%3.1f %s" % (size, x)
      size /= 1024.0
    return "%3.1f %s" % (size, 'TB')

  def reportHook(self,blocksSoFar,blockSize,totalSize):
    # we clamp to 100% because the blockSize might be larger than the file itself
    percent = min(int((100. * blocksSoFar * blockSize) / totalSize), 100)
    if percent == 100 or (percent - self.downloadPercent >= 10):
      # we clamp to totalSize when blockSize is larger than totalSize
      humanSizeSoFar = self.humanFormatSize(min(blocksSoFar * blockSize, totalSize))
      humanSizeTotal = self.humanFormatSize(totalSize)
      self.logMessage('<i>Downloaded %s (%d%% of %s)...</i>' % (humanSizeSoFar, percent, humanSizeTotal))
      self.downloadPercent = percent

  def downloadFile(self, uri, destFolderPath, name, checksum=None):
    """
    :param uri: Download URL.
    :param destFolderPath: Folder to download the file into.
    :param name: File name that will be downloaded.
    :param checksum: Checksum formatted as ``<algo>:<digest>`` to verify the downloaded file. For example, ``SHA256:cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93``.
    """
    filePath = destFolderPath + '/' + name
    (algo, digest) = extractAlgoAndDigest(checksum)
    if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
      import urllib.request, urllib.parse, urllib.error
      self.logMessage('<b>Requesting download</b> <i>%s</i> from %s ...' % (name, uri))
      # add a progress bar
      self.downloadPercent = 0
      try:
        urllib.request.urlretrieve(uri, filePath, self.reportHook)
        self.logMessage('<b>Download finished</b>')
      except IOError as e:
        self.logMessage('<b>\tDownload failed: %s</b>' % e, logging.ERROR)

      if algo is not None:
        self.logMessage('<b>Verifying checksum</b>')
        current_digest = computeChecksum(algo, filePath)
        if current_digest != digest:
          self.logMessage('<b>Checksum verification failed. Computed checksum %s different from expected checksum %s</b>' % (current_digest, digest))
          qt.QFile(filePath).remove()
        else:
          self.logMessage('<b>Checksum OK</b>')
    else:
      if algo is not None:
        self.logMessage('<b>Verifying checksum</b>')
        current_digest = computeChecksum(algo, filePath)
        if current_digest != digest:
          self.logMessage('<b>File already exists in cache but checksum is different - re-downloading it.</b>')
          qt.QFile(filePath).remove()
          return self.downloadFile(uri, destFolderPath, name, checksum)
        else:
          self.logMessage('<b>File already exists and checksum is OK - reusing it.</b>')
      else:
        self.logMessage('<b>File already exists in cache - reusing it.</b>')
    return filePath

  def loadScene(self, uri,  fileProperties = {}):
    self.logMessage('<b>Requesting load</b> %s ...' % uri)
    fileProperties['fileName'] = uri
    success = slicer.app.coreIOManager().loadNodes('SceneFile', fileProperties)
    if not success:
      self.logMessage('<b>\tLoad failed!</b>', logging.ERROR)
      return False
    self.logMessage('<b>Load finished</b>')
    return True

  def loadNode(self, uri, name, fileType = 'VolumeFile', fileProperties = {}):
    self.logMessage('<b>Requesting load</b> <i>%s</i> from %s ...' % (name, uri))

    fileProperties['fileName'] = uri
    fileProperties['name'] = name
    firstLoadedNode = None
    loadedNodes = vtk.vtkCollection()
    success = slicer.app.coreIOManager().loadNodes(fileType, fileProperties, loadedNodes)

    if not success or loadedNodes.GetNumberOfItems()<1:
      self.logMessage('<b>\tLoad failed!</b>', logging.ERROR)
      return None

    self.logMessage('<b>Load finished</b>')

    # since nodes were read from a temp directory remove the storage nodes
    for i in range(loadedNodes.GetNumberOfItems()):
      loadedNode = loadedNodes.GetItemAsObject(i)
      if not loadedNode.IsA("vtkMRMLStorableNode"):
        continue
      storageNode = loadedNode.GetStorageNode()
      if not storageNode:
        continue
      slicer.mrmlScene.RemoveNode(storageNode)
      loadedNode.SetAndObserveStorageNodeID(None)

    return loadedNodes.GetItemAsObject(0)


class SampleDataTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  customDownloads = []

  def setUp(self):
    slicer.mrmlScene.Clear(0)
    SampleDataTest.customDownloads = []

  def runTest(self):
    for test in [
      self.test_downloadFromSource_downloadFiles,
      self.test_downloadFromSource_downloadZipFile,
      self.test_downloadFromSource_loadMRBFile,
      self.test_downloadFromSource_loadMRMLFile,
      self.test_downloadFromSource_downloadMRBFile,
      self.test_downloadFromSource_downloadMRMLFile,
      self.test_downloadFromSource_loadNode,
      self.test_downloadFromSource_loadNodeFromMultipleFiles,
      self.test_downloadFromSource_loadNodes,
      self.test_downloadFromSource_loadNodesWithLoadFileFalse,
      self.test_isSampleDataSourceRegistered,
      self.test_categoryVisibility,
      self.test_customDownloader,
    ]:
      self.setUp()
      test()

  @staticmethod
  def path2uri(path):
    """Gets an ur from a local file path.
    Typically it prefixes the received path by file:// or file:///.
    """
    import urllib.parse, urllib.request, urllib.parse, urllib.error
    return urllib.parse.urljoin('file:', urllib.request.pathname2url(path))

  def test_downloadFromSource_downloadFiles(self):
    """Specifying URIs and fileNames without nodeNames is expected to download the files
    without loading into Slicer.
    """
    logic = SampleDataLogic()

    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris='http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', fileNames='MR-head.nrrd'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertEqual(sceneMTime, slicer.mrmlScene.GetMTime())

    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris=['http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', 'http://slicer.kitware.com/midas3/download/item/292307/CT-chest.nrrd'],
      fileNames=['MR-head.nrrd', 'CT-chest.nrrd']))
    self.assertEqual(len(filePaths), 2)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertTrue(os.path.exists(filePaths[1]))
    self.assertTrue(os.path.isfile(filePaths[1]))
    self.assertEqual(sceneMTime, slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_downloadZipFile(self):
    logic = SampleDataLogic()
    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris='http://slicer.kitware.com/midas3/download/folder/3763/TinyPatient_Seg.zip', fileNames='TinyPatient_Seg.zip'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isdir(filePaths[0]))
    self.assertEqual(sceneMTime, slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_loadMRBFile(self):
    logic = SampleDataLogic()
    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris='http://slicer.kitware.com/midas3/download?items=8466', loadFiles=True, fileNames='slicer4minute.mrb'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertTrue(sceneMTime < slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_loadMRMLFile(self):
    logic = SampleDataLogic()
    tempFile = qt.QTemporaryFile(slicer.app.temporaryPath + "/SampleDataTest-loadSceneFile-XXXXXX.mrml");
    tempFile.open()
    tempFile.write(textwrap.dedent("""
      <?xml version="1.0" encoding="ISO-8859-1"?>
      <MRML  version="Slicer4.4.0" userTags="">
      </MRML>
      """).strip())
    tempFile.close()
    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris=self.path2uri(tempFile.fileName()), loadFiles=True, fileNames='scene.mrml'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertTrue(sceneMTime < slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_downloadMRBFile(self):
    logic = SampleDataLogic()
    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris='http://slicer.kitware.com/midas3/download?items=8466', fileNames='slicer4minute.mrb'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertEqual(sceneMTime, slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_downloadMRMLFile(self):
    logic = SampleDataLogic()
    tempFile = qt.QTemporaryFile(slicer.app.temporaryPath + "/SampleDataTest-loadSceneFile-XXXXXX.mrml");
    tempFile.open()
    tempFile.write(textwrap.dedent("""
      <?xml version="1.0" encoding="ISO-8859-1"?>
      <MRML  version="Slicer4.4.0" userTags="">
      </MRML>
      """).strip())
    tempFile.close()
    sceneMTime = slicer.mrmlScene.GetMTime()
    filePaths = logic.downloadFromSource(SampleDataSource(
      uris=self.path2uri(tempFile.fileName()), fileNames='scene.mrml'))
    self.assertEqual(len(filePaths), 1)
    self.assertTrue(os.path.exists(filePaths[0]))
    self.assertTrue(os.path.isfile(filePaths[0]))
    self.assertEqual(sceneMTime, slicer.mrmlScene.GetMTime())

  def test_downloadFromSource_loadNode(self):
    logic = SampleDataLogic()
    nodes = logic.downloadFromSource(SampleDataSource(
      uris='http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', fileNames='MR-head.nrrd', nodeNames='MRHead'))
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0], slicer.mrmlScene.GetFirstNodeByName("MRHead"))

  def test_downloadFromSource_loadNodeFromMultipleFiles(self):
    logic = SampleDataLogic()
    nodes = logic.downloadFromSource(SampleDataSource(
      uris=['http://slicer.kitware.com/midas3/download/?items=2011,1', 'http://slicer.kitware.com/midas3/download/?items=2010,1'],
      fileNames=['DTIVolume.raw.gz', 'DTIVolume.nhdr'],
      nodeNames=[None, 'DTIVolume']))
    self.assertEqual(len(nodes), 1)
    self.assertEqual(nodes[0], slicer.mrmlScene.GetFirstNodeByName("DTIVolume"))

  def test_downloadFromSource_loadNodesWithLoadFileFalse(self):
    logic = SampleDataLogic()
    nodes = logic.downloadFromSource(SampleDataSource(
      uris=['http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', 'http://slicer.kitware.com/midas3/download/item/292307/CT-chest.nrrd'],
      fileNames=['MR-head.nrrd', 'CT-chest.nrrd'],
      nodeNames=['MRHead', 'CTChest'],
      loadFiles=[False, True]))
    self.assertEqual(len(nodes), 2)
    self.assertTrue(os.path.exists(nodes[0]))
    self.assertTrue(os.path.isfile(nodes[0]))
    self.assertEqual(nodes[1], slicer.mrmlScene.GetFirstNodeByName("CTChest"))

  def test_downloadFromSource_loadNodes(self):
    logic = SampleDataLogic()
    nodes = logic.downloadFromSource(SampleDataSource(
      uris=['http://slicer.kitware.com/midas3/download/item/292308/MR-head.nrrd', 'http://slicer.kitware.com/midas3/download/item/292307/CT-chest.nrrd'],
      fileNames=['MR-head.nrrd', 'CT-chest.nrrd'],
      nodeNames=['MRHead', 'CTChest']))
    self.assertEqual(len(nodes), 2)
    self.assertEqual(nodes[0], slicer.mrmlScene.GetFirstNodeByName("MRHead"))
    self.assertEqual(nodes[1], slicer.mrmlScene.GetFirstNodeByName("CTChest"))

  def test_isSampleDataSourceRegistered(self):
    widget = slicer.modules.SampleDataWidget
    self.assertTrue(widget.isSampleDataSourceRegistered('BuiltIn'))
    self.assertFalse(widget.isSampleDataSourceRegistered('Not_A_Registered_Category'))

  def test_categoryVisibility(self):
    slicer.util.selectModule("SampleData")
    widget = slicer.modules.SampleDataWidget
    widget.setCategoryVisible('BuiltIn', False)
    self.assertFalse(widget.isCategoryVisible('BuiltIn'))
    widget.setCategoryVisible('BuiltIn', True)
    self.assertTrue(widget.isCategoryVisible('BuiltIn'))

  class CustomDownloader(object):
    def __call__(self, source):
      SampleDataTest.customDownloads.append(source)

  CustomDownloaderDataSource = {
    'category': "Testing",
    'sampleName': 'customDownloader',
    'uris': 'http://down.load/test',
    'fileNames': 'cust.om',
    'customDownloader': CustomDownloader()
  }

  def test_customDownloader(self):
    if not slicer.app.testingEnabled():
      return
    slicer.util.selectModule("SampleData")
    widget = slicer.modules.SampleDataWidget
    button = slicer.util.findChild(widget.parent,'customDownloaderPushButton')

    self.assertEqual(self.customDownloads, [])

    button.click()

    self.assertEqual(len(self.customDownloads), 1)
    self.assertEqual(self.customDownloads[0].sampleName, 'customDownloader')
