import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# ScreenCapture
#

class ScreenCapture(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Screen Capture"
    self.parent.categories = ["Utilities"]
    self.parent.dependencies = []
    self.parent.contributors = ["Andras Lasso (PerkLab Queen's University)"]
    self.parent.helpText = """
This module captures image sequences and videos
from dynamic contents shown in 3D and slice viewers.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This work was was funded by Cancer Care Ontario
and the Ontario Consortium for Adaptive Interventions in Radiation Oncology (OCAIRO)
"""

#
# ScreenCaptureWidget
#

VIEW_SLICE = 'slice'
VIEW_3D = '3d'

AXIS_YAW = 0
AXIS_PITCH = 1

class ScreenCaptureWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = ScreenCaptureLogic()
    self.logic.logCallback = self.addLog
    self.viewNodeType = None
    self.animationMode = None
    self.createdOutputFile = None

    self.snapshotIndex = 0 # this counter is used for determining file names for single-image snapshots
    self.snapshotOutputDir = None
    self.snapshotFileNamePattern = None

    # Instantiate and connect widgets ...

    #
    # Input area
    #
    self.inputCollapsibleButton = ctk.ctkCollapsibleButton()
    self.inputCollapsibleButton.text = "Input"
    self.layout.addWidget(self.inputCollapsibleButton)
    inputFormLayout = qt.QFormLayout(self.inputCollapsibleButton)

    # Input view selector
    self.viewNodeSelector = slicer.qMRMLNodeComboBox()
    self.viewNodeSelector.nodeTypes = ["vtkMRMLSliceNode", "vtkMRMLViewNode"]
    self.viewNodeSelector.addEnabled = False
    self.viewNodeSelector.removeEnabled = False
    self.viewNodeSelector.noneEnabled = False
    self.viewNodeSelector.showHidden = False
    self.viewNodeSelector.showChildNodeTypes = False
    self.viewNodeSelector.setMRMLScene( slicer.mrmlScene )
    self.viewNodeSelector.setToolTip("This slice or 3D view will be updated during capture."
      "Only this view will be captured unless 'Capture of all views' option in output section is enabled." )
    inputFormLayout.addRow("Master view: ", self.viewNodeSelector)

    # Mode
    self.animationModeWidget = qt.QComboBox()
    self.animationModeWidget.setToolTip("Select the property that will be adjusted")
    inputFormLayout.addRow("Animation mode:", self.animationModeWidget)

    # Slice start offset position
    self.sliceStartOffsetSliderLabel = qt.QLabel("Start sweep offset:")
    self.sliceStartOffsetSliderWidget = ctk.ctkSliderWidget()
    self.sliceStartOffsetSliderWidget.singleStep = 30
    self.sliceStartOffsetSliderWidget.minimum = -100
    self.sliceStartOffsetSliderWidget.maximum = 100
    self.sliceStartOffsetSliderWidget.value = 0
    self.sliceStartOffsetSliderWidget.setToolTip("Start slice sweep offset.")
    inputFormLayout.addRow(self.sliceStartOffsetSliderLabel, self.sliceStartOffsetSliderWidget)

    # Slice end offset position
    self.sliceEndOffsetSliderLabel = qt.QLabel("End sweep offset:")
    self.sliceEndOffsetSliderWidget = ctk.ctkSliderWidget()
    self.sliceEndOffsetSliderWidget.singleStep = 5
    self.sliceEndOffsetSliderWidget.minimum = -100
    self.sliceEndOffsetSliderWidget.maximum = 100
    self.sliceEndOffsetSliderWidget.value = 0
    self.sliceEndOffsetSliderWidget.setToolTip("End slice sweep offset.")
    inputFormLayout.addRow(self.sliceEndOffsetSliderLabel, self.sliceEndOffsetSliderWidget)

    # 3D rotation range
    self.rotationSliderLabel = qt.QLabel("Rotation range:")
    self.rotationSliderWidget = ctk.ctkRangeWidget()
    self.rotationSliderWidget.singleStep = 5
    self.rotationSliderWidget.minimum = -180
    self.rotationSliderWidget.maximum = 180
    self.rotationSliderWidget.minimumValue = -180
    self.rotationSliderWidget.maximumValue = 180
    self.rotationSliderWidget.setToolTip("View rotation range, relative to current view orientation.")
    inputFormLayout.addRow(self.rotationSliderLabel, self.rotationSliderWidget)

    # 3D rotation axis
    self.rotationAxisLabel = qt.QLabel("Rotation axis:")
    self.rotationAxisWidget = ctk.ctkRangeWidget()
    self.rotationAxisWidget = qt.QComboBox()
    self.rotationAxisWidget.addItem("Yaw", AXIS_YAW)
    self.rotationAxisWidget.addItem("Pitch", AXIS_PITCH)
    inputFormLayout.addRow(self.rotationAxisLabel, self.rotationAxisWidget)


    # Sequence browser node selector
    self.sequenceBrowserNodeSelectorLabel = qt.QLabel("Sequence:")
    self.sequenceBrowserNodeSelectorWidget = slicer.qMRMLNodeComboBox()
    self.sequenceBrowserNodeSelectorWidget.nodeTypes = ["vtkMRMLSequenceBrowserNode"]
    self.sequenceBrowserNodeSelectorWidget.addEnabled = False
    self.sequenceBrowserNodeSelectorWidget.removeEnabled = False
    self.sequenceBrowserNodeSelectorWidget.noneEnabled = False
    self.sequenceBrowserNodeSelectorWidget.showHidden = False
    self.sequenceBrowserNodeSelectorWidget.setMRMLScene( slicer.mrmlScene )
    self.sequenceBrowserNodeSelectorWidget.setToolTip( "Items defined by this sequence browser will be replayed." )
    inputFormLayout.addRow(self.sequenceBrowserNodeSelectorLabel, self.sequenceBrowserNodeSelectorWidget)

    # Sequence start index
    self.sequenceStartItemIndexLabel = qt.QLabel("Start index:")
    self.sequenceStartItemIndexWidget = ctk.ctkSliderWidget()
    self.sequenceStartItemIndexWidget.minimum = 0
    self.sequenceStartItemIndexWidget.decimals = 0
    self.sequenceStartItemIndexWidget.setToolTip("First item in the sequence to capture.")
    inputFormLayout.addRow(self.sequenceStartItemIndexLabel, self.sequenceStartItemIndexWidget)

    # Sequence end index
    self.sequenceEndItemIndexLabel = qt.QLabel("End index:")
    self.sequenceEndItemIndexWidget = ctk.ctkSliderWidget()
    self.sequenceEndItemIndexWidget.minimum = 0
    self.sequenceEndItemIndexWidget.decimals = 0
    self.sequenceEndItemIndexWidget.setToolTip("Last item in the sequence to capture.")
    inputFormLayout.addRow(self.sequenceEndItemIndexLabel, self.sequenceEndItemIndexWidget)

    #
    # Output area
    #
    self.outputCollapsibleButton = ctk.ctkCollapsibleButton()
    self.outputCollapsibleButton.text = "Output"
    self.layout.addWidget(self.outputCollapsibleButton)
    outputFormLayout = qt.QFormLayout(self.outputCollapsibleButton)

    # Number of steps value
    self.numberOfStepsSliderWidget = ctk.ctkSliderWidget()
    self.numberOfStepsSliderWidget.singleStep = 1
    self.numberOfStepsSliderWidget.pageStep = 10
    self.numberOfStepsSliderWidget.minimum = 1
    self.numberOfStepsSliderWidget.maximum = 600
    self.numberOfStepsSliderWidget.value = 31
    self.numberOfStepsSliderWidget.decimals = 0
    self.numberOfStepsSliderWidget.setToolTip("Number of images extracted between start and stop positions.")

    # Single step toggle button
    self.singleStepButton = qt.QToolButton()
    self.singleStepButton.setText("single")
    self.singleStepButton.setCheckable(True)
    self.singleStepButton.toolTip = "Capture a single image of current state only.\n" + \
      "New filename is generated for each captured image (no files are overwritten)."

    hbox = qt.QHBoxLayout()
    hbox.addWidget(self.singleStepButton)
    hbox.addWidget(self.numberOfStepsSliderWidget)
    outputFormLayout.addRow("Number of images:", hbox)

    # Output directory selector
    self.outputDirSelector = ctk.ctkPathLineEdit()
    self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirSelector.settingKey = 'ScreenCaptureOutputDir'
    outputFormLayout.addRow("Output directory:", self.outputDirSelector)
    if not self.outputDirSelector.currentPath:
      defaultOutputPath = os.path.abspath(os.path.join(slicer.app.defaultScenePath,'SlicerCapture'))
      self.outputDirSelector.setCurrentPath(defaultOutputPath)

    self.captureAllViewsCheckBox = qt.QCheckBox(" ")
    self.captureAllViewsCheckBox.checked = False
    self.captureAllViewsCheckBox.setToolTip("If checked, all views will be captured. If unchecked then only the selected view will be captured.")
    outputFormLayout.addRow("Capture all views:", self.captureAllViewsCheckBox)

    self.videoExportCheckBox = qt.QCheckBox(" ")
    self.videoExportCheckBox.checked = False
    self.videoExportCheckBox.setToolTip("If checked, exported images will be written as a video file."
      " Requires setting of ffmpeg executable path in Advanced section.")

    self.videoFormatWidget = qt.QComboBox()
    self.videoFormatWidget.enabled = False
    self.videoFormatWidget.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
    for videoFormatPreset in self.logic.videoFormatPresets:
      self.videoFormatWidget.addItem(videoFormatPreset["name"])

    hbox = qt.QHBoxLayout()
    hbox.addWidget(self.videoExportCheckBox)
    hbox.addWidget(self.videoFormatWidget)
    outputFormLayout.addRow("Video export:", hbox)

    self.videoFileNameWidget = qt.QLineEdit()
    self.videoFileNameWidget.setToolTip("String that defines file name and type.")
    self.videoFileNameWidget.text = "SlicerCapture.avi"
    self.videoFileNameWidget.setEnabled(False)
    outputFormLayout.addRow("Video file name:", self.videoFileNameWidget)

    self.videoLengthSliderWidget = ctk.ctkSliderWidget()
    self.videoLengthSliderWidget.singleStep = 0.1
    self.videoLengthSliderWidget.minimum = 0.1
    self.videoLengthSliderWidget.maximum = 30
    self.videoLengthSliderWidget.value = 5
    self.videoLengthSliderWidget.suffix = "s"
    self.videoLengthSliderWidget.decimals = 1
    self.videoLengthSliderWidget.setToolTip("Length of the exported video in seconds (without backward steps and repeating).")
    self.videoLengthSliderWidget.setEnabled(False)
    outputFormLayout.addRow("Video length:", self.videoLengthSliderWidget)

    self.videoFrameRateSliderWidget = ctk.ctkSliderWidget()
    self.videoFrameRateSliderWidget.singleStep = 0.1
    self.videoFrameRateSliderWidget.minimum = 0.1
    self.videoFrameRateSliderWidget.maximum = 60
    self.videoFrameRateSliderWidget.value = 5.0
    self.videoFrameRateSliderWidget.suffix = "fps"
    self.videoFrameRateSliderWidget.decimals = 3
    self.videoFrameRateSliderWidget.setToolTip("Frame rate in frames per second.")
    self.videoFrameRateSliderWidget.setEnabled(False)
    outputFormLayout.addRow("Video frame rate:", self.videoFrameRateSliderWidget)

    #
    # Advanced area
    #
    self.advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    self.advancedCollapsibleButton.text = "Advanced"
    self.advancedCollapsibleButton.collapsed = True
    outputFormLayout.addRow(self.advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(self.advancedCollapsibleButton)

    self.forwardBackwardCheckBox = qt.QCheckBox(" ")
    self.forwardBackwardCheckBox.checked = False
    self.forwardBackwardCheckBox.setToolTip("If checked, image series will be generated playing forward and then backward.")
    advancedFormLayout.addRow("Forward-backward:", self.forwardBackwardCheckBox)

    self.repeatSliderWidget = ctk.ctkSliderWidget()
    self.repeatSliderWidget.decimals = 0
    self.repeatSliderWidget.singleStep = 1
    self.repeatSliderWidget.minimum = 1
    self.repeatSliderWidget.maximum = 50
    self.repeatSliderWidget.value = 1
    self.repeatSliderWidget.setToolTip("Number of times image series are repeated. Useful for making short videos longer for playback in software"
      " that does not support looped playback.")
    advancedFormLayout.addRow("Repeat:", self.repeatSliderWidget)

    ffmpegPath = self.logic.getFfmpegPath()
    self.ffmpegPathSelector = ctk.ctkPathLineEdit()
    self.ffmpegPathSelector.setCurrentPath(ffmpegPath)
    self.ffmpegPathSelector.nameFilters = [self.logic.getFfmpegExecutableFilename()]
    self.ffmpegPathSelector.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Preferred)
    self.ffmpegPathSelector.setToolTip("Set the path to ffmpeg executable. Download from: https://www.ffmpeg.org/")
    advancedFormLayout.addRow("ffmpeg executable:", self.ffmpegPathSelector)

    self.videoExportFfmpegWarning = qt.QLabel('<qt><b><font color="red">Set valid ffmpeg executable path! '+
      '<a href="http://wiki.slicer.org/slicerWiki/index.php/Documentation/Nightly/Modules/ScreenCapture#Setting_up_ffmpeg">Help...</a></font></b></qt>')
    self.videoExportFfmpegWarning.connect('linkActivated(QString)', self.openURL)
    self.videoExportFfmpegWarning.setVisible(False)
    advancedFormLayout.addRow("", self.videoExportFfmpegWarning)

    self.extraVideoOptionsWidget = qt.QLineEdit()
    self.extraVideoOptionsWidget.setToolTip('Additional video conversion options passed to ffmpeg. Parameters -i (input files), -y'
      +'(overwrite without asking), -r (frame rate), -start_number are specified by the module and therefore'
      +'should not be included in this list.')
    advancedFormLayout.addRow("Video extra options:", self.extraVideoOptionsWidget)

    self.fileNamePatternWidget = qt.QLineEdit()
    self.fileNamePatternWidget.setToolTip(
      "String that defines file name, type, and numbering scheme. Default: image%05d.png.")
    self.fileNamePatternWidget.text = "image_%05d.png"
    advancedFormLayout.addRow("Image file name pattern:", self.fileNamePatternWidget)

    self.maxFramesWidget = qt.QSpinBox()
    self.maxFramesWidget.setRange(1, 9999)
    self.maxFramesWidget.setValue(600)
    self.maxFramesWidget.setToolTip(
      "Maximum number of images to be captured (without backward steps and repeating).")
    advancedFormLayout.addRow("Maximum number of images:", self.maxFramesWidget)

    self.transparentBackgroundCheckBox = qt.QCheckBox(" ")
    self.transparentBackgroundCheckBox.checked = False
    self.transparentBackgroundCheckBox.setToolTip("If checked, images will be captured with transparent background.")
    advancedFormLayout.addRow("Transparent background:", self.transparentBackgroundCheckBox)

    # Capture button
    self.captureButtonLabelCapture = "Capture"
    self.captureButtonLabelCancel = "Cancel"
    self.captureButton = qt.QPushButton(self.captureButtonLabelCapture)
    self.captureButton.toolTip = "Capture slice sweep to image sequence."
    self.showCreatedOutputFileButton = qt.QPushButton()
    self.showCreatedOutputFileButton.setIcon(qt.QIcon(':Icons/Go.png'))
    self.showCreatedOutputFileButton.setMaximumWidth(60)
    self.showCreatedOutputFileButton.enabled = False
    self.showCreatedOutputFileButton.toolTip = "Show created output file."
    hbox = qt.QHBoxLayout()
    hbox.addWidget(self.captureButton)
    hbox.addWidget(self.showCreatedOutputFileButton)
    self.layout.addLayout(hbox)

    self.statusLabel = qt.QPlainTextEdit()
    self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
    self.statusLabel.setCenterOnScroll(True)
    self.layout.addWidget(self.statusLabel)

    #
    # Add vertical spacer
    # self.layout.addStretch(1)

    # connections
    self.captureButton.connect('clicked(bool)', self.onCaptureButton)
    self.showCreatedOutputFileButton.connect('clicked(bool)', self.onShowCreatedOutputFile)
    self.viewNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateViewOptions)
    self.animationModeWidget.connect("currentIndexChanged(int)", self.updateViewOptions)
    self.sliceStartOffsetSliderWidget.connect('valueChanged(double)', self.setSliceOffset)
    self.sliceEndOffsetSliderWidget.connect('valueChanged(double)', self.setSliceOffset)
    self.sequenceBrowserNodeSelectorWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.updateViewOptions)
    self.sequenceStartItemIndexWidget.connect('valueChanged(double)', self.setSequenceItemIndex)
    self.sequenceEndItemIndexWidget.connect('valueChanged(double)', self.setSequenceItemIndex)
    self.videoExportCheckBox.connect('toggled(bool)', self.fileNamePatternWidget, 'setDisabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoFileNameWidget, 'setEnabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoFormatWidget, 'setEnabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoLengthSliderWidget, 'setEnabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoFrameRateSliderWidget, 'setEnabled(bool)')
    self.videoFormatWidget.connect("currentIndexChanged(int)", self.updateVideoFormat)
    self.singleStepButton.connect('toggled(bool)', self.numberOfStepsSliderWidget, 'setDisabled(bool)')
    self.maxFramesWidget.connect('valueChanged(int)', self.maxFramesChanged)
    self.videoLengthSliderWidget.connect('valueChanged(double)', self.setVideoLength)
    self.videoFrameRateSliderWidget.connect('valueChanged(double)', self.setVideoFrameRate)
    self.numberOfStepsSliderWidget.connect('valueChanged(double)', self.setNumberOfSteps)

    self.setVideoLength() # update frame rate based on video length
    self.updateVideoFormat(0)
    self.updateViewOptions()

  def maxFramesChanged(self):
    self.numberOfStepsSliderWidget.maximum = self.maxFramesWidget.value

  def openURL(self, URL):
    qt.QDesktopServices().openUrl(qt.QUrl(URL))
    QDesktopServices

  def onShowCreatedOutputFile(self):
    if not self.createdOutputFile:
      return
    qt.QDesktopServices().openUrl(qt.QUrl("file:///"+self.createdOutputFile, qt.QUrl.TolerantMode));

  def updateVideoFormat(self, selectionIndex):
    videoFormatPreset = self.logic.videoFormatPresets[selectionIndex]

    import os
    filenameExt = os.path.splitext(self.videoFileNameWidget.text)
    self.videoFileNameWidget.text = filenameExt[0] + "." + videoFormatPreset["fileExtension"]

    self.extraVideoOptionsWidget.text = videoFormatPreset["extraVideoOptions"]

  def currentViewNodeType(self):
    viewNode = self.viewNodeSelector.currentNode()
    if not viewNode:
      return None
    elif viewNode.IsA("vtkMRMLSliceNode"):
      return VIEW_SLICE
    elif viewNode.IsA("vtkMRMLViewNode"):
      return VIEW_3D
    else:
      return None

  def addLog(self, text):
    """Append text to log window
    """
    self.statusLabel.appendPlainText(text)
    self.statusLabel.ensureCursorVisible()
    slicer.app.processEvents() # force update

  def cleanup(self):
    pass

  def updateViewOptions(self):

    sequencesModuleAvailable = hasattr(slicer.modules, 'sequences')

    if self.viewNodeType !=  self.currentViewNodeType():
      self.viewNodeType = self.currentViewNodeType()

      self.animationModeWidget.clear()
      if self.viewNodeType == VIEW_SLICE:
        self.animationModeWidget.addItem("slice sweep")
        self.animationModeWidget.addItem("slice fade")
      if self.viewNodeType == VIEW_3D:
        self.animationModeWidget.addItem("3D rotation")
      if sequencesModuleAvailable:
        self.animationModeWidget.addItem("sequence")

    if self.animationMode != self.animationModeWidget.currentText:
      self.animationMode = self.animationModeWidget.currentText

    # slice sweep
    self.sliceStartOffsetSliderLabel.visible = (self.animationMode == "slice sweep")
    self.sliceStartOffsetSliderWidget.visible = (self.animationMode == "slice sweep")
    self.sliceEndOffsetSliderLabel.visible = (self.animationMode == "slice sweep")
    self.sliceEndOffsetSliderWidget.visible = (self.animationMode == "slice sweep")
    if self.animationMode == "slice sweep":
      offsetResolution = self.logic.getSliceOffsetResolution(self.viewNodeSelector.currentNode())
      sliceOffsetMin, sliceOffsetMax = self.logic.getSliceOffsetRange(self.viewNodeSelector.currentNode())

      wasBlocked = self.sliceStartOffsetSliderWidget.blockSignals(True)
      self.sliceStartOffsetSliderWidget.singleStep = offsetResolution
      self.sliceStartOffsetSliderWidget.minimum = sliceOffsetMin
      self.sliceStartOffsetSliderWidget.maximum = sliceOffsetMax
      self.sliceStartOffsetSliderWidget.value = sliceOffsetMin
      self.sliceStartOffsetSliderWidget.blockSignals(wasBlocked)

      wasBlocked = self.sliceEndOffsetSliderWidget.blockSignals(True)
      self.sliceEndOffsetSliderWidget.singleStep = offsetResolution
      self.sliceEndOffsetSliderWidget.minimum = sliceOffsetMin
      self.sliceEndOffsetSliderWidget.maximum = sliceOffsetMax
      self.sliceEndOffsetSliderWidget.value = sliceOffsetMax
      self.sliceEndOffsetSliderWidget.blockSignals(wasBlocked)

    # 3D rotation
    self.rotationSliderLabel.visible = (self.animationMode == "3D rotation")
    self.rotationSliderWidget.visible = (self.animationMode == "3D rotation")
    self.rotationAxisLabel.visible = (self.animationMode == "3D rotation")
    self.rotationAxisWidget.visible = (self.animationMode == "3D rotation")

    # Sequence
    self.sequenceBrowserNodeSelectorLabel.visible = (self.animationMode == "sequence")
    self.sequenceBrowserNodeSelectorWidget.visible = (self.animationMode == "sequence")
    self.sequenceStartItemIndexLabel.visible = (self.animationMode == "sequence")
    self.sequenceStartItemIndexWidget.visible = (self.animationMode == "sequence")
    self.sequenceEndItemIndexLabel.visible = (self.animationMode == "sequence")
    self.sequenceEndItemIndexWidget.visible = (self.animationMode == "sequence")
    if self.animationMode == "sequence":
      sequenceBrowserNode = self.sequenceBrowserNodeSelectorWidget.currentNode()

      sequenceItemCount = 0
      if sequenceBrowserNode and sequenceBrowserNode.GetMasterSequenceNode():
        sequenceItemCount = sequenceBrowserNode.GetMasterSequenceNode().GetNumberOfDataNodes()

      if sequenceItemCount>0:
        wasBlocked = self.sequenceStartItemIndexWidget.blockSignals(True)
        self.sequenceStartItemIndexWidget.maximum = sequenceItemCount-1
        self.sequenceStartItemIndexWidget.value = 0
        self.sequenceStartItemIndexWidget.blockSignals(wasBlocked)

        wasBlocked = self.sequenceEndItemIndexWidget.blockSignals(True)
        self.sequenceEndItemIndexWidget.maximum = sequenceItemCount-1
        self.sequenceEndItemIndexWidget.value = sequenceItemCount-1
        self.sequenceEndItemIndexWidget.blockSignals(wasBlocked)

      self.sequenceStartItemIndexWidget.enabled = sequenceItemCount>0
      self.sequenceEndItemIndexWidget.enabled = sequenceItemCount>0

    numberOfSteps = int(self.numberOfStepsSliderWidget.value)
    self.forwardBackwardCheckBox.enabled = (numberOfSteps > 1)
    self.repeatSliderWidget.enabled = (numberOfSteps > 1)

  def setSliceOffset(self, offset):
    sliceLogic = self.logic.getSliceLogicFromSliceNode(self.viewNodeSelector.currentNode())
    sliceLogic.SetSliceOffset(offset)

  def setVideoLength(self, lengthSec=None):
    wasBlocked = self.videoFrameRateSliderWidget.blockSignals(True)
    self.videoFrameRateSliderWidget.value = self.numberOfStepsSliderWidget.value / self.videoLengthSliderWidget.value
    self.videoFrameRateSliderWidget.blockSignals(wasBlocked)

  def setVideoFrameRate(self, frameRateFps):
    wasBlocked = self.videoFrameRateSliderWidget.blockSignals(True)
    self.videoLengthSliderWidget.value = self.numberOfStepsSliderWidget.value / self.videoFrameRateSliderWidget.value
    self.videoFrameRateSliderWidget.blockSignals(wasBlocked)

  def setNumberOfSteps(self, frameRateFps):
    self.setVideoLength()

  def setSequenceItemIndex(self, index):
    sequenceBrowserNode = self.sequenceBrowserNodeSelectorWidget.currentNode()
    sequenceBrowserNode.SetSelectedItemNumber(int(index))

  def enableInputOutputWidgets(self, enable):
    self.inputCollapsibleButton.setEnabled(enable)
    self.outputCollapsibleButton.setEnabled(enable)

  def onCaptureButton(self):

    # Disable capture button to prevent multiple clicks
    self.captureButton.setEnabled(False)
    self.enableInputOutputWidgets(False)
    slicer.app.processEvents()

    if self.captureButton.text == self.captureButtonLabelCancel:
      self.logic.requestCancel()
      return

    self.logic.setFfmpegPath(self.ffmpegPathSelector.currentPath)

    self.statusLabel.plainText = ''

    videoOutputRequested = self.videoExportCheckBox.checked
    viewNode = self.viewNodeSelector.currentNode()
    numberOfSteps = int(self.numberOfStepsSliderWidget.value)
    if self.singleStepButton.checked:
      numberOfSteps = 1
    if numberOfSteps < 2:
      # If a single image is selected
      videoOutputRequested = False
    outputDir = self.outputDirSelector.currentPath

    self.videoExportFfmpegWarning.setVisible(False)
    if videoOutputRequested:
      if not self.logic.isFfmpegPathValid():
        # ffmpeg not found, try to automatically find it at common locations
        self.logic.findFfmpeg()
      if not self.logic.isFfmpegPathValid() and os.name == 'nt': # TODO: implement download for Linux/MacOS?
        # ffmpeg not found, offer downloading it
        if slicer.util.confirmOkCancelDisplay(
          'Video encoder not detected on your system. '
          'Download ffmpeg video encoder?',
          windowTitle='Download confirmation'):
          if not self.logic.ffmpegDownload():
            slicer.util.errorDisplay("ffmpeg download failed")
      if not self.logic.isFfmpegPathValid():
        # still not found, user has to specify path manually
        self.videoExportFfmpegWarning.setVisible(True)
        self.advancedCollapsibleButton.collapsed = False
        self.captureButton.setEnabled(True)
        self.enableInputOutputWidgets(True)
        return
      self.ffmpegPathSelector.currentPath = self.logic.getFfmpegPath()

    # Need to create a new random file pattern if video output is requested to make sure that new image files are not mixed up with
    # existing files in the output directory
    imageFileNamePattern = self.logic.getRandomFilePattern() if videoOutputRequested else self.fileNamePatternWidget.text

    self.captureButton.setEnabled(True)
    self.captureButton.text = self.captureButtonLabelCancel
    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    captureAllViews = self.captureAllViewsCheckBox.checked
    transparentBackground = self.transparentBackgroundCheckBox.checked
    if captureAllViews:
      self.logic.showViewControllers(False)
    try:
      if numberOfSteps < 2:
        if imageFileNamePattern != self.snapshotFileNamePattern or outputDir != self.snapshotOutputDir:
          self.snapshotIndex = 0
        [filename, self.snapshotIndex] = self.logic.getNextAvailableFileName(outputDir, imageFileNamePattern, self.snapshotIndex)
        view = None if captureAllViews else self.logic.viewFromNode(viewNode)
        self.logic.captureImageFromView(view, filename, transparentBackground)
        self.logic.addLog("Write "+filename)
      elif self.animationModeWidget.currentText == "slice sweep":
        self.logic.captureSliceSweep(viewNode, self.sliceStartOffsetSliderWidget.value,
          self.sliceEndOffsetSliderWidget.value, numberOfSteps, outputDir, imageFileNamePattern,
          captureAllViews = captureAllViews, transparentBackground = transparentBackground)
      elif self.animationModeWidget.currentText == "slice fade":
        self.logic.captureSliceFade(viewNode, numberOfSteps, outputDir, imageFileNamePattern,
        captureAllViews = captureAllViews, transparentBackground = transparentBackground)
      elif self.animationModeWidget.currentText == "3D rotation":
        self.logic.capture3dViewRotation(viewNode, self.rotationSliderWidget.minimumValue,
          self.rotationSliderWidget.maximumValue, numberOfSteps,
          self.rotationAxisWidget.itemData(self.rotationAxisWidget.currentIndex),
          outputDir, imageFileNamePattern,
          captureAllViews = captureAllViews, transparentBackground = transparentBackground)
      elif self.animationModeWidget.currentText == "sequence":
        self.logic.captureSequence(viewNode, self.sequenceBrowserNodeSelectorWidget.currentNode(),
          self.sequenceStartItemIndexWidget.value, self.sequenceEndItemIndexWidget.value,
          numberOfSteps, outputDir, imageFileNamePattern,
          captureAllViews = captureAllViews, transparentBackground = transparentBackground)
      else:
        raise ValueError('Unsupported view node type.')

      import shutil

      fps = self.videoFrameRateSliderWidget.value

      if numberOfSteps > 1:
        forwardBackward = self.forwardBackwardCheckBox.checked
        numberOfRepeats = int(self.repeatSliderWidget.value)
        filePathPattern = os.path.join(outputDir, imageFileNamePattern)
        fileIndex = numberOfSteps
        for repeatIndex in range(numberOfRepeats):
          if forwardBackward:
            for step in reversed(range(1, numberOfSteps-1)):
              sourceFilename = filePathPattern % step
              destinationFilename = filePathPattern % fileIndex
              self.logic.addLog("Copy to "+destinationFilename)
              shutil.copyfile(sourceFilename, destinationFilename)
              fileIndex += 1
          if repeatIndex < numberOfRepeats - 1:
            for step in range(numberOfSteps):
              sourceFilename = filePathPattern % step
              destinationFilename = filePathPattern % fileIndex
              self.logic.addLog("Copy to "+destinationFilename)
              shutil.copyfile(sourceFilename, destinationFilename)
              fileIndex += 1
        if forwardBackward and (numberOfSteps > 2):
          numberOfSteps += numberOfSteps - 2
        numberOfSteps *= numberOfRepeats

      if videoOutputRequested:
        try:
          self.logic.createVideo(fps, self.extraVideoOptionsWidget.text,
            outputDir, imageFileNamePattern, self.videoFileNameWidget.text)
        except Exception as e:
          self.logic.deleteTemporaryFiles(outputDir, imageFileNamePattern, numberOfSteps)
          raise
        self.logic.deleteTemporaryFiles(outputDir, imageFileNamePattern, numberOfSteps)

      self.addLog("Done.")
      self.createdOutputFile = os.path.join(outputDir, self.videoFileNameWidget.text) if videoOutputRequested else outputDir
      self.showCreatedOutputFileButton.enabled = True
    except Exception as e:
      self.addLog("Error: {0}".format(str(e)))
      import traceback
      traceback.print_exc()
      self.showCreatedOutputFileButton.enabled = False
      self.createdOutputFile = None
    if captureAllViews:
      self.logic.showViewControllers(True)
    slicer.app.restoreOverrideCursor()
    self.captureButton.text = self.captureButtonLabelCapture
    self.captureButton.setEnabled(True)
    self.enableInputOutputWidgets(True)

#
# ScreenCaptureLogic
#

class ScreenCaptureLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    self.logCallback = None
    self.cancelRequested = False

    self.videoFormatPresets = [
      {"name": "H.264",                    "fileExtension": "mp4", "extraVideoOptions": "-codec libx264 -preset slower -pix_fmt yuv420p"},
      {"name": "H.264 (high-quality)",     "fileExtension": "mp4", "extraVideoOptions": "-codec libx264 -preset slower -crf 18 -pix_fmt yuv420p"},
      {"name": "MPEG-4",                   "fileExtension": "mp4", "extraVideoOptions": "-codec mpeg4 -qscale 5"},
      {"name": "MPEG-4 (high-quality)",    "fileExtension": "mp4", "extraVideoOptions": "-codec mpeg4 -qscale 3"},
      {"name": "Animated GIF",             "fileExtension": "gif", "extraVideoOptions": "-filter_complex palettegen,[v]paletteuse"},
      {"name": "Animated GIF (grayscale)", "fileExtension": "gif", "extraVideoOptions": "-vf format=gray"} ]

  def requestCancel(self):
    logging.info("User requested cancelling of capture")
    self.cancelRequested = True

  def addLog(self, text):
    logging.info(text)
    if self.logCallback:
      self.logCallback(text)

  def showViewControllers(self, show):
    lm = slicer.app.layoutManager()
    for viewIndex in range(lm.threeDViewCount):
      lm.threeDWidget(viewIndex).threeDController().setVisible(show)
    for sliceViewName in lm.sliceViewNames():
      lm.sliceWidget(sliceViewName).sliceController().setVisible(show)

  def getRandomFilePattern(self):
    import string
    import random
    numberOfRandomChars=5
    randomString = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(numberOfRandomChars))
    filePathPattern = "tmp-"+randomString+"-%05d.png"
    return filePathPattern

  def isFfmpegPathValid(self):
    import os
    ffmpegPath = self.getFfmpegPath()
    return os.path.isfile(ffmpegPath)

  def getDownloadedFfmpegDirectory(self):
    return os.path.dirname(slicer.app.slicerUserSettingsFilePath)+'/ffmpeg'

  def getFfmpegExecutableFilename(self):
    if os.name == 'nt':
      return 'ffmpeg.exe'
    else:
      return 'ffmpeg'

  def findFfmpeg(self):
    # Try to find the executable at specific paths
    commonFfmpegPaths = [
      '/usr/local/bin/ffmpeg',
      '/usr/bin/ffmpeg'
      ]
    for ffmpegPath in commonFfmpegPaths:
      if os.path.isfile(ffmpegPath):
        # found one
        self.setFfmpegPath(ffmpegPath)
        return True
    # Search for the executable in directories
    commonFfmpegDirs = [
      self.getDownloadedFfmpegDirectory()
      ]
    for ffmpegDir in commonFfmpegDirs:
      if self.findFfmpegInDirectory(ffmpegDir):
        # found it
        return True
    # Not found
    return False

  def findFfmpegInDirectory(self, ffmpegDir):
    ffmpegExecutableFilename = self.getFfmpegExecutableFilename()
    for dirpath, dirnames, files in os.walk(ffmpegDir):
      for name in files:
        if name==ffmpegExecutableFilename:
          ffmpegExecutablePath = (dirpath + '/' + name).replace('\\','/')
          self.setFfmpegPath(ffmpegExecutablePath)
          return True
    return False

  def unzipFfmpeg(self, filePath, ffmpegTargetDirectory):
    if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
      logging.info('ffmpeg package is not found at ' + filePath)
      return False

    logging.info('Unzipping ffmpeg package ' + filePath)
    qt.QDir().mkpath(ffmpegTargetDirectory)
    slicer.app.applicationLogic().Unzip(filePath, ffmpegTargetDirectory)
    success = self.findFfmpegInDirectory(ffmpegTargetDirectory)
    return success

  def ffmpegDownload(self):
    ffmpegTargetDirectory = self.getDownloadedFfmpegDirectory()
    filePath = slicer.app.temporaryPath + '/ffmpeg-package.zip'
    success = self.unzipFfmpeg(filePath, ffmpegTargetDirectory)
    if success:
      # there was a valid downloaded package already
      return True

    # List of mirror sites to attempt download ffmpeg pre-built binaries from
    urls = []
    if os.name == 'nt':
      urls.append('http://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-3.2.4-win64-static.zip')
    else:
      # TODO: implement downloading for Linux/MacOS?
      pass

    success = False
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    for url in urls:

      success = True
      try:
        logging.info('Requesting download ffmpeg from %s...' % url)
        import urllib.request, urllib.error, urllib.parse
        req = urllib.request.Request(url, headers={ 'User-Agent': 'Mozilla/5.0' })
        data = urllib.request.urlopen(req).read()
        with open(filePath, "wb") as f:
          f.write(data)

        success = self.unzipFfmpeg(filePath, ffmpegTargetDirectory)
      except:
        success = False

      if success:
        break

    qt.QApplication.restoreOverrideCursor()
    return success

  def getFfmpegPath(self):
    settings = qt.QSettings()
    if settings.contains('General/ffmpegPath'):
      return settings.value('General/ffmpegPath')
    return ''

  def setFfmpegPath(self, ffmpegPath):
    # don't save it if already saved
    settings = qt.QSettings()
    if settings.contains('General/ffmpegPath'):
      if ffmpegPath == settings.value('General/ffmpegPath'):
        return
    settings.setValue('General/ffmpegPath',ffmpegPath)

  def getSliceLogicFromSliceNode(self, sliceNode):
    lm = slicer.app.layoutManager()
    sliceLogic = lm.sliceWidget(sliceNode.GetLayoutName()).sliceLogic()
    return sliceLogic

  def getSliceOffsetRange(self, sliceNode):
    sliceLogic = self.getSliceLogicFromSliceNode(sliceNode)

    sliceBounds = [0, -1, 0, -1, 0, -1]
    sliceLogic.GetLowestVolumeSliceBounds(sliceBounds)
    sliceOffsetMin = sliceBounds[4]
    sliceOffsetMax = sliceBounds[5]

    # increase range if it is empty
    # to allow capturing even when no volumes are shown in slice views
    if sliceOffsetMin == sliceOffsetMax:
      sliceOffsetMin = sliceLogic.GetSliceOffset()-100
      sliceOffsetMax = sliceLogic.GetSliceOffset()+100

    return sliceOffsetMin, sliceOffsetMax

  def getSliceOffsetResolution(self, sliceNode):
    sliceLogic = self.getSliceLogicFromSliceNode(sliceNode)

    sliceOffsetResolution = 1.0
    sliceSpacing = sliceLogic.GetLowestVolumeSliceSpacing();
    if sliceSpacing is not None and sliceSpacing[2]>0:
      sliceOffsetResolution = sliceSpacing[2]

    return sliceOffsetResolution

  def captureImageFromView(self, view, filename, transparentBackground=False):

    slicer.app.processEvents()
    if view:
      view.forceRender()
    else:
      # force rendering of all views
      lm = slicer.app.layoutManager()
      for viewIndex in range(lm.threeDViewCount):
        lm.threeDWidget(viewIndex).threeDView().forceRender()
      for sliceViewName in lm.sliceViewNames():
        lm.sliceWidget(sliceViewName).sliceView().forceRender()

    if view is None:
      if transparentBackground:
        logging.warning("Transparent background is only available for single-view capture")

      # no view is specified, capture the entire view layout

      # Simply using grabwidget on the view layout frame would grab the screen without background:
      # img = ctk.ctkWidgetsUtils.grabWidget(slicer.app.layoutManager().viewport())

      # Grab the main window and use only the viewport's area
      allViews = slicer.app.layoutManager().viewport()
      topLeft = allViews.mapTo(slicer.util.mainWindow(),allViews.rect.topLeft())
      bottomRight = allViews.mapTo(slicer.util.mainWindow(),allViews.rect.bottomRight())
      imageSize = bottomRight - topLeft

      if imageSize.x()<2 or imageSize.y()<2:
        # image is too small, most likely it is invalid
        raise ValueError('Capture image from view failed')

      # Make sure image witdth and height is even, otherwise encoding may fail
      if (imageSize.x() & 1 == 1):
        imageSize.setX(imageSize.x()-1)
      if (imageSize.y() & 1 == 1):
        imageSize.setY(imageSize.y()-1)

      img = ctk.ctkWidgetsUtils.grabWidget(slicer.util.mainWindow(), qt.QRect(topLeft.x(), topLeft.y(), imageSize.x(), imageSize.y()))
      img.save(filename)
      return

    rw = view.renderWindow()
    wti = vtk.vtkWindowToImageFilter()

    if transparentBackground:
      originalAlphaBitPlanes = rw.GetAlphaBitPlanes()
      rw.SetAlphaBitPlanes(1)
      ren=rw.GetRenderers().GetFirstRenderer()
      originalGradientBackground = ren.GetGradientBackground()
      ren.SetGradientBackground(False)
      wti.SetInputBufferTypeToRGBA()
      rw.Render() # need to render after changing bit planes

    wti.SetInput(rw)
    wti.Update()

    if transparentBackground:
      rw.SetAlphaBitPlanes(originalAlphaBitPlanes)
      ren.SetGradientBackground(originalGradientBackground)

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    outputImage = wti.GetOutput()
    imageSize = outputImage.GetDimensions()

    if imageSize[0]<2 or imageSize[1]<2:
      # image is too small, most likely it is invalid
      raise ValueError('Capture image from view failed')

    # Make sure image witdth and height is even, otherwise encoding may fail
    imageWidthOdd = (imageSize[0] & 1 == 1)
    imageHeightOdd = (imageSize[1] & 1 == 1)
    if imageWidthOdd or imageHeightOdd:
      imageClipper = vtk.vtkImageClip()
      imageClipper.SetInputConnection(wti.GetOutputPort())
      extent = outputImage.GetExtent()
      imageClipper.SetOutputWholeExtent(extent[0], extent[1]-1 if imageWidthOdd else extent[1],
                                        extent[2], extent[3]-1 if imageHeightOdd else extent[3],
                                        extent[4], extent[5])
      writer.SetInputConnection(imageClipper.GetOutputPort())
    else:
      writer.SetInputConnection(wti.GetOutputPort())

    writer.Write()

  def viewFromNode(self, viewNode):
    if not viewNode:
      raise ValueError('Invalid view node.')
    elif viewNode.IsA("vtkMRMLSliceNode"):
      return slicer.app.layoutManager().sliceWidget(viewNode.GetLayoutName()).sliceView()
    elif viewNode.IsA("vtkMRMLViewNode"):
      renderView = None
      lm = slicer.app.layoutManager()
      for widgetIndex in range(lm.threeDViewCount):
        view = lm.threeDWidget(widgetIndex).threeDView()
        if viewNode == view.mrmlViewNode():
          renderView = view
          break
      if not renderView:
        raise ValueError('Selected 3D view is not visible in the current layout.')
      return renderView
    else:
      raise ValueError('Invalid view node.')

  def captureSliceSweep(self, sliceNode, startSliceOffset, endSliceOffset, numberOfImages,
                        outputDir, outputFilenamePattern, captureAllViews = None, transparentBackground = False):

    self.cancelRequested = False

    if not captureAllViews and not sliceNode.IsMappedInLayout():
      raise ValueError('Selected slice view is not visible in the current layout.')

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir,outputFilenamePattern)

    sliceLogic = self.getSliceLogicFromSliceNode(sliceNode)
    originalSliceOffset = sliceLogic.GetSliceOffset()

    sliceView = self.viewFromNode(sliceNode)
    compositeNode = sliceLogic.GetSliceCompositeNode()
    offsetStepSize = (endSliceOffset-startSliceOffset)/(numberOfImages-1)
    for offsetIndex in range(numberOfImages):
      filename = filePathPattern % offsetIndex
      self.addLog("Write "+filename)
      sliceLogic.SetSliceOffset(startSliceOffset+offsetIndex*offsetStepSize)
      self.captureImageFromView(None if captureAllViews else sliceView, filename, transparentBackground)
      if self.cancelRequested:
        break

    sliceLogic.SetSliceOffset(originalSliceOffset)
    if self.cancelRequested:
      raise ValueError('User requested cancel.')

  def captureSliceFade(self, sliceNode, numberOfImages, outputDir,
                        outputFilenamePattern, captureAllViews = None, transparentBackground = False):

    self.cancelRequested = False

    if not captureAllViews and not sliceNode.IsMappedInLayout():
      raise ValueError('Selected slice view is not visible in the current layout.')

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir, outputFilenamePattern)

    sliceLogic = self.getSliceLogicFromSliceNode(sliceNode)
    sliceView = self.viewFromNode(sliceNode)
    compositeNode = sliceLogic.GetSliceCompositeNode()
    originalForegroundOpacity = compositeNode.GetForegroundOpacity()
    startForegroundOpacity = 0.0
    endForegroundOpacity = 1.0
    opacityStepSize = (endForegroundOpacity - startForegroundOpacity) / (numberOfImages - 1)
    for offsetIndex in range(numberOfImages):
      filename = filePathPattern % offsetIndex
      self.addLog("Write "+filename)
      compositeNode.SetForegroundOpacity(startForegroundOpacity + offsetIndex * opacityStepSize)
      self.captureImageFromView(None if captureAllViews else sliceView, filename, transparentBackground)
      if self.cancelRequested:
        break

    compositeNode.SetForegroundOpacity(originalForegroundOpacity)

    if self.cancelRequested:
      raise ValueError('User requested cancel.')

  def capture3dViewRotation(self, viewNode, startRotation, endRotation, numberOfImages, rotationAxis,
    outputDir, outputFilenamePattern, captureAllViews = None, transparentBackground = False):
    """
    Acquire a set of screenshots of the 3D view while rotating it.
    """

    self.cancelRequested = False

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir, outputFilenamePattern)

    renderView = self.viewFromNode(viewNode)

    # Save original orientation and go to start orientation
    originalPitchRollYawIncrement = renderView.pitchRollYawIncrement
    originalDirection = renderView.pitchDirection
    renderView.setPitchRollYawIncrement(-startRotation)
    if rotationAxis == AXIS_YAW:
      renderView.yawDirection = renderView.YawRight
      renderView.yaw()
    else:
      renderView.pitchDirection = renderView.PitchDown
      renderView.pitch()

    # Rotate step-by-step
    rotationStepSize = (endRotation - startRotation) / (numberOfImages - 1)
    renderView.setPitchRollYawIncrement(rotationStepSize)
    if rotationAxis == AXIS_YAW:
      renderView.yawDirection = renderView.YawLeft
    else:
      renderView.pitchDirection = renderView.PitchUp
    for offsetIndex in range(numberOfImages):
      if not self.cancelRequested:
        filename = filePathPattern % offsetIndex
        self.addLog("Write " + filename)
        self.captureImageFromView(None if captureAllViews else renderView, filename, transparentBackground)
      if rotationAxis == AXIS_YAW:
        renderView.yaw()
      else:
        renderView.pitch()

    # Restore original orientation and rotation step size & direction
    if rotationAxis == AXIS_YAW:
      renderView.yawDirection = renderView.YawRight
      renderView.yaw()
      renderView.setPitchRollYawIncrement(endRotation)
      renderView.yaw()
      renderView.setPitchRollYawIncrement(originalPitchRollYawIncrement)
      renderView.yawDirection = originalDirection
    else:
      renderView.pitchDirection = renderView.PitchDown
      renderView.pitch()
      renderView.setPitchRollYawIncrement(endRotation)
      renderView.pitch()
      renderView.setPitchRollYawIncrement(originalPitchRollYawIncrement)
      renderView.pitchDirection = originalDirection

    if self.cancelRequested:
      raise ValueError('User requested cancel.')

  def captureSequence(self, viewNode, sequenceBrowserNode, sequenceStartIndex,
                        sequenceEndIndex, numberOfImages, outputDir, outputFilenamePattern,
                        captureAllViews = None, transparentBackground = False):
    """
    Acquire a set of screenshots of a view while iterating through a sequence.
    """

    self.cancelRequested = False

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir, outputFilenamePattern)

    originalSelectedItemNumber = sequenceBrowserNode.GetSelectedItemNumber()

    renderView = self.viewFromNode(viewNode)
    stepSize = (sequenceEndIndex - sequenceStartIndex) / (numberOfImages - 1)
    for offsetIndex in range(numberOfImages):
      sequenceBrowserNode.SetSelectedItemNumber(int(sequenceStartIndex+offsetIndex*stepSize))
      filename = filePathPattern % offsetIndex
      self.addLog("Write " + filename)
      self.captureImageFromView(None if captureAllViews else renderView, filename, transparentBackground)
      if self.cancelRequested:
        break

    sequenceBrowserNode.SetSelectedItemNumber(originalSelectedItemNumber)
    if self.cancelRequested:
      raise ValueError('User requested cancel.')

  def createVideo(self, frameRate, extraOptions, outputDir, imageFileNamePattern, videoFileName):
    self.addLog("Export to video...")

    # Get ffmpeg
    import os.path
    ffmpegPath = os.path.abspath(self.getFfmpegPath())
    if not ffmpegPath:
      raise ValueError("Video creation failed: ffmpeg executable path is not defined")
    if not os.path.isfile(ffmpegPath):
      raise ValueError("Video creation failed: ffmpeg executable path is invalid: "+ffmpegPath)

    filePathPattern = os.path.join(outputDir, imageFileNamePattern)
    outputVideoFilePath = os.path.join(outputDir, videoFileName)
    ffmpegParams = [ffmpegPath,
                    "-y", # overwrite without asking
                    "-r", str(frameRate),
                    "-start_number", "0",
                    "-i", str(filePathPattern)]
    ffmpegParams += [_f for _f in extraOptions.split(' ') if _f]
    ffmpegParams.append(outputVideoFilePath)

    self.addLog("Start ffmpeg:\n"+' '.join(ffmpegParams))

    import subprocess
    p = subprocess.Popen(ffmpegParams, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=outputDir)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
      self.addLog("ffmpeg error output: " + stderr.decode())
      raise ValueError("ffmpeg returned with error")
    else:
      self.addLog("Video export succeeded to file: "+outputVideoFilePath)
      logging.debug("ffmpeg standard output: " + stdout.decode())
      logging.debug("ffmpeg error output: " + stderr.decode())

  def deleteTemporaryFiles(self, outputDir, imageFileNamePattern, numberOfImages):
    """
    Delete files after a video has been created from them.
    """
    import os
    filePathPattern = os.path.join(outputDir, imageFileNamePattern)
    for imageIndex in range(numberOfImages):
      filename = filePathPattern % imageIndex
      logging.debug("Delete temporary file " + filename)
      os.remove(filename)

  def getNextAvailableFileName(self, outputDir, outputFilenamePattern, snapshotIndex):
    """
    Find a file index that does not overwrite any existing file.
    """
    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir,outputFilenamePattern)
    filename = None
    while True:
      filename = filePathPattern % snapshotIndex
      if not os.path.exists(filename):
        # found an available file name
        break
      snapshotIndex += 1
    return [filename, snapshotIndex]

class ScreenCaptureTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)
    import SampleData
    self.image1 = SampleData.downloadSample('MRBrainTumor1')
    self.image2 = SampleData.downloadSample('MRBrainTumor2')

    # make the output volume appear in all the slice views
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetActiveVolumeID(self.image1.GetID())
    selectionNode.SetSecondaryVolumeID(self.image2.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection(1)

    # Show slice and 3D views
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    for sliceViewNodeId in ['vtkMRMLSliceNodeRed', 'vtkMRMLSliceNodeYellow', 'vtkMRMLSliceNodeGreen']:
      slicer.mrmlScene.GetNodeByID(sliceViewNodeId).SetSliceVisible(True)

    self.tempDir= slicer.app.temporaryPath + '/ScreenCaptureTest'
    self.numberOfImages = 10
    self.imageFileNamePattern = "image_%05d.png"

    self.logic = ScreenCaptureLogic()

  def verifyAndDeleteWrittenFiles(self):
    import os
    filePathPattern = os.path.join(self.tempDir, self.imageFileNamePattern)
    for imageIndex in range(self.numberOfImages):
      filename = filePathPattern % imageIndex
      self.assertTrue(os.path.exists(filename))
    self.logic.deleteTemporaryFiles(self.tempDir, self.imageFileNamePattern, self.numberOfImages)
    for imageIndex in range(self.numberOfImages):
      filename = filePathPattern % imageIndex
      self.assertFalse(os.path.exists(filename))

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SliceSweep()
    self.test_SliceFade()
    self.test_3dViewRotation()

  def test_SliceSweep(self):
    self.delayDisplay("Testing SliceSweep")
    viewNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
    self.assertIsNotNone(viewNode)
    self.logic.captureSliceSweep(viewNode, -125, 75, self.numberOfImages, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing SliceSweep completed successfully')

  def test_SliceFade(self):
    self.delayDisplay("Testing SliceFade")
    viewNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
    self.assertIsNotNone(viewNode)
    self.logic.captureSliceFade(viewNode, self.numberOfImages, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing SliceFade completed successfully')

  def test_3dViewRotation(self):
    self.delayDisplay("Testing 3D view rotation")
    viewNode = slicer.mrmlScene.GetNodeByID('vtkMRMLViewNode1')
    self.assertIsNotNone(viewNode)
    self.logic.capture3dViewRotation(viewNode, -180, 180, self.numberOfImages, AXIS_YAW, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing 3D view rotation completed successfully')
