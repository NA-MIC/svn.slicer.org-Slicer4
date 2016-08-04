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
    self.parent.helpText = """Capture image sequences from 2D and 3D viewers."""
    self.parent.acknowledgementText = """This work was was funded by Cancer Care Ontario
 and the Ontario Consortium for Adaptive Interventions in Radiation Oncology (OCAIRO)"""

#
# ScreenCaptureWidget
#

VIEW_SLICE = 'slice'
VIEW_3D = '3d'

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

    # Instantiate and connect widgets ...

    #
    # Input area
    #
    inputCollapsibleButton = ctk.ctkCollapsibleButton()
    inputCollapsibleButton.text = "Input"
    self.layout.addWidget(inputCollapsibleButton)
    inputFormLayout = qt.QFormLayout(inputCollapsibleButton)

    # Input view selector
    self.viewNodeSelector = slicer.qMRMLNodeComboBox()
    self.viewNodeSelector.nodeTypes = ["vtkMRMLSliceNode", "vtkMRMLViewNode"]
    self.viewNodeSelector.addEnabled = False
    self.viewNodeSelector.removeEnabled = False
    self.viewNodeSelector.noneEnabled = False
    self.viewNodeSelector.showHidden = False
    self.viewNodeSelector.showChildNodeTypes = False
    self.viewNodeSelector.setMRMLScene( slicer.mrmlScene )
    self.viewNodeSelector.setToolTip( "Contents of this slice or 3D view will be captured." )
    inputFormLayout.addRow("View to capture: ", self.viewNodeSelector)

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

    # 3D start rotation
    self.startRotationSliderLabel = qt.QLabel("Start rotation angle:")
    self.startRotationSliderWidget = ctk.ctkSliderWidget()
    self.startRotationSliderWidget.singleStep = 5
    self.startRotationSliderWidget.minimum = 0
    self.startRotationSliderWidget.maximum = 180
    self.startRotationSliderWidget.value = 180
    self.startRotationSliderWidget.setToolTip("Rotation angle for the first image, relative to current orientation.")
    inputFormLayout.addRow(self.startRotationSliderLabel, self.startRotationSliderWidget)

    # 3D end rotation
    self.endRotationSliderLabel = qt.QLabel("End rotation angle:")
    self.endRotationSliderWidget = ctk.ctkSliderWidget()
    self.endRotationSliderWidget.singleStep = 5
    self.endRotationSliderWidget.minimum = 0
    self.endRotationSliderWidget.maximum = 180
    self.endRotationSliderWidget.value = 180
    self.endRotationSliderWidget.setToolTip("Rotation angle for the last image, relative to current orientation.")
    inputFormLayout.addRow(self.endRotationSliderLabel, self.endRotationSliderWidget)

    # Sequence browser node selector
    self.sequenceBrowserNodeSelectorLabel = qt.QLabel("End rotation angle:")
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
    self.sequenceEndItemIndexWidget.singleStep = 10
    self.sequenceEndItemIndexWidget.minimum = 0
    self.sequenceEndItemIndexWidget.decimals = 0
    self.sequenceEndItemIndexWidget.setToolTip("Last item in the sequence to capture.")
    inputFormLayout.addRow(self.sequenceEndItemIndexLabel, self.sequenceEndItemIndexWidget)

    #
    # Output area
    #
    outputCollapsibleButton = ctk.ctkCollapsibleButton()
    outputCollapsibleButton.text = "Output"
    self.layout.addWidget(outputCollapsibleButton)
    outputFormLayout = qt.QFormLayout(outputCollapsibleButton)

    # Number of steps value
    self.numberOfStepsSliderWidget = ctk.ctkSliderWidget()
    self.numberOfStepsSliderWidget.singleStep = 10
    self.numberOfStepsSliderWidget.minimum = 2
    self.numberOfStepsSliderWidget.maximum = 150
    self.numberOfStepsSliderWidget.value = 31
    self.numberOfStepsSliderWidget.decimals = 0
    self.numberOfStepsSliderWidget.setToolTip("Number of images extracted between start and stop positions.")
    outputFormLayout.addRow("Number of images:", self.numberOfStepsSliderWidget)

    # Output directory selector
    self.outputDirSelector = ctk.ctkPathLineEdit()
    self.outputDirSelector.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirSelector.settingKey = 'ScreenCaptureOutputDir'
    outputFormLayout.addRow("Output directory:", self.outputDirSelector)
    if not self.outputDirSelector.currentPath:
      defaultOutputPath = os.path.abspath(os.path.join(slicer.app.defaultScenePath,'SlicerCapture'))
      self.outputDirSelector.setCurrentPath(defaultOutputPath)

    self.videoExportCheckBox = qt.QCheckBox()
    self.videoExportCheckBox.checked = False
    self.videoExportCheckBox.setToolTip("If checked, exported images will be written as a video file."
      " Requires setting of ffmpeg executable path in Advanced section.")
    outputFormLayout.addRow("Video export:", self.videoExportCheckBox)

    self.videoFileNameWidget = qt.QLineEdit()
    self.videoFileNameWidget.setToolTip("String that defines file name, type, and numbering scheme. Default: capture.avi.")
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
    self.videoLengthSliderWidget.setToolTip("Length of the exported video in seconds.")
    self.videoLengthSliderWidget.setEnabled(False)
    outputFormLayout.addRow("Video length:", self.videoLengthSliderWidget)

    #
    # Advanced area
    #
    self.advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    self.advancedCollapsibleButton.text = "Advanced"
    self.advancedCollapsibleButton.collapsed = True
    outputFormLayout.addRow(self.advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(self.advancedCollapsibleButton)

    self.fileNamePatternWidget = qt.QLineEdit()
    self.fileNamePatternWidget.setToolTip(
      "String that defines file name, type, and numbering scheme. Default: image%05d.png.")
    self.fileNamePatternWidget.text = "image_%05d.png"
    advancedFormLayout.addRow("Image file name pattern:", self.fileNamePatternWidget)

    ffmpegPath = self.logic.getFfmpegPath()
    self.ffmpegPathSelector = ctk.ctkPathLineEdit()
    self.ffmpegPathSelector.setCurrentPath(ffmpegPath)
    self.ffmpegPathSelector.nameFilters = ['ffmpeg.exe', 'ffmpeg']
    self.ffmpegPathSelector.setMaximumWidth(300)
    self.ffmpegPathSelector.setToolTip("Set the path to ffmpeg executable. Download from: https://www.ffmpeg.org/")
    advancedFormLayout.addRow("ffmpeg executable:", self.ffmpegPathSelector)

    self.extraVideoOptionsWidget = qt.QComboBox()
    self.extraVideoOptionsWidget.addItem("-c:v mpeg4 -qscale:v 5")
    self.extraVideoOptionsWidget.addItem("-c:v libx264 -preset veryslow -qp 0")
    self.extraVideoOptionsWidget.addItem("-f mp4 -vcodec libx264 -pix_fmt yuv420p")
    self.extraVideoOptionsWidget.setEditable(True)
    self.extraVideoOptionsWidget.setToolTip(
      '<html>\n'
      '  <p>Additional video conversion options passed to ffmpeg.</p>'
      '  <p><b>Examples:</b>'
      '  <ul>'
      '    <li><b>MPEG4: </b>-c:v mpeg4 -qscale:v 5</li>'
      '    <li><b>H264: </b>-c:v libx264 -preset veryslow -qp 0</li>'
      '    <li><b>Quicktime: </b>-f mp4 -vcodec libx264 -pix_fmt yuv420p</li>'
      '  </ul></p>'
      '  <p>See more encoding options at:'
      '  <i>https://trac.ffmpeg.org/wiki/Encode/H.264</i> and'
      '  <i>https://trac.ffmpeg.org/wiki/Encode/MPEG-4</i></p>'
      '</html>')
    advancedFormLayout.addRow("Video extra options:", self.extraVideoOptionsWidget)

    # Capture button
    self.captureButton = qt.QPushButton("Capture")
    self.captureButton.toolTip = "Capture slice sweep to image sequence."
    outputFormLayout.addRow(self.captureButton)

    self.statusLabel = qt.QPlainTextEdit()
    self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
    self.statusLabel.setCenterOnScroll(True)
    outputFormLayout.addRow(self.statusLabel)

    #
    # Add vertical spacer
    #self.layout.addStretch(1)

    # connections
    self.captureButton.connect('clicked(bool)', self.onCaptureButton)
    self.viewNodeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateViewOptions)
    self.animationModeWidget.connect("currentIndexChanged(int)", self.updateViewOptions)
    self.sliceStartOffsetSliderWidget.connect('valueChanged(double)', self.setSliceOffset)
    self.sliceEndOffsetSliderWidget.connect('valueChanged(double)', self.setSliceOffset)
    self.sequenceBrowserNodeSelectorWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.updateViewOptions)
    self.sequenceStartItemIndexWidget.connect('valueChanged(double)', self.setSequenceItemIndex)
    self.sequenceEndItemIndexWidget.connect('valueChanged(double)', self.setSequenceItemIndex)
    self.videoExportCheckBox.connect('toggled(bool)', self.fileNamePatternWidget, 'setDisabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoFileNameWidget, 'setEnabled(bool)')
    self.videoExportCheckBox.connect('toggled(bool)', self.videoLengthSliderWidget, 'setEnabled(bool)')

    self.updateViewOptions()

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
    self.startRotationSliderLabel.visible = (self.animationMode == "3D rotation")
    self.startRotationSliderWidget.visible = (self.animationMode == "3D rotation")
    self.endRotationSliderLabel.visible = (self.animationMode == "3D rotation")
    self.endRotationSliderWidget.visible = (self.animationMode == "3D rotation")

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

  def setSliceOffset(self, offset):
    sliceLogic = self.logic.getSliceLogicFromSliceNode(self.viewNodeSelector.currentNode())
    sliceLogic.SetSliceOffset(offset)

  def setSequenceItemIndex(self, index):
    sequenceBrowserNode = self.sequenceBrowserNodeSelectorWidget.currentNode()
    sequenceBrowserNode.SetSelectedItemNumber(int(index))

  def onSelect(self):
    self.captureButton.enabled = self.viewNodeSelector.currentNode()

  def onCaptureButton(self):
    self.logic.setFfmpegPath(self.ffmpegPathSelector.currentPath)

    slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
    self.statusLabel.plainText = ''

    videoOutputRequested = self.videoExportCheckBox.checked
    viewNode = self.viewNodeSelector.currentNode()
    numberOfSteps = int(self.numberOfStepsSliderWidget.value)
    outputDir = self.outputDirSelector.currentPath

    # Need to create a new random file pattern if video output is requested to make sure that new image files are not mixed up with
    # existing files in the output directory
    imageFileNamePattern = self.logic.getRandomFilePattern() if videoOutputRequested else self.fileNamePatternWidget.text

    try:
      if self.animationModeWidget.currentText == "slice sweep":
        self.logic.captureSliceSweep(viewNode, self.sliceStartOffsetSliderWidget.value,
          self.sliceEndOffsetSliderWidget.value, numberOfSteps, outputDir, imageFileNamePattern)
      elif self.animationModeWidget.currentText == "slice fade":
        self.logic.captureSliceFade(viewNode, numberOfSteps, outputDir, imageFileNamePattern)
      elif self.animationModeWidget.currentText == "3D rotation":
        self.logic.capture3dViewRotation(viewNode, self.startRotationSliderWidget.value,
          self.endRotationSliderWidget.value, numberOfSteps, outputDir, imageFileNamePattern)
      elif self.animationModeWidget.currentText == "sequence":
        self.logic.captureSequence(viewNode, self.sequenceBrowserNodeSelectorWidget.currentNode(),
          self.sequenceStartItemIndexWidget.value, self.sequenceEndItemIndexWidget.value,
          numberOfSteps, outputDir, imageFileNamePattern)
      else:
        raise ValueError('Unsupported view node type.')

      if videoOutputRequested:
        fps = numberOfSteps / self.videoLengthSliderWidget.value
        try:
          self.logic.createVideo(fps, self.extraVideoOptionsWidget.currentText,
            outputDir, imageFileNamePattern, self.videoFileNameWidget.text)
        except Exception as e:
          self.logic.deleteTemporaryFiles(outputDir, imageFileNamePattern, numberOfSteps)
          raise ValueError(e)
        self.logic.deleteTemporaryFiles(outputDir, imageFileNamePattern, numberOfSteps)

      self.addLog("Done.")
    except Exception as e:
      self.addLog("Unexpected error: {0}".format(e.message))
      import traceback
      traceback.print_exc()
    slicer.app.restoreOverrideCursor()

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

  def addLog(self, text):
    logging.info(text)
    if self.logCallback:
      self.logCallback(text)

  def getRandomFilePattern(self):
    import string
    import random
    numberOfRandomChars=5
    randomString = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(numberOfRandomChars))
    filePathPattern = "tmp-"+randomString+"-%05d.png"
    return filePathPattern

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

  def captureImageFromView(self, view, filename):
    view.forceRender()
    # qt.QPixmap().grabWidget(...) would not grab the background
    rw = view.renderWindow()
    wti = vtk.vtkWindowToImageFilter()
    wti.SetInput(rw)
    wti.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    outputImage = wti.GetOutput()
    imageSize = outputImage.GetDimensions()

    if imageSize[0]<2 or imageSize[1]<2:
      # image is too small, most likely it is invalid
      raise ValueError('Capture image from view failed')

    # Make sure image witdth is even, otherwise encoding may fail
    if imageSize[0] & 1 == 1:
      imageClipper = vtk.vtkImageClip()
      imageClipper.SetInputConnection(wti.GetOutputPort())
      extent = outputImage.GetExtent()
      imageClipper.SetOutputWholeExtent(extent[0], extent[1]-1, extent[2], extent[3], extent[4], extent[5])
      writer.SetInputConnection(imageClipper.GetOutputPort())
    else:
      writer.SetInputConnection(wti.GetOutputPort())

    writer.Write()

  def viewFromNode(self, viewNode):
    if not viewNode:
      raise ValueError('Invalid view node.')
    elif viewNode.IsA("vtkMRMLSliceNode"):
      if not viewNode.IsMappedInLayout():
        raise ValueError('Selected slice view is not visible in the current layout.')
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

  def captureSliceSweep(self, sliceNode, startSliceOffset, endSliceOffset, numberOfImages, outputDir, outputFilenamePattern):
    if not sliceNode.IsMappedInLayout():
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
      self.captureImageFromView(sliceView, filename)

    sliceLogic.SetSliceOffset(originalSliceOffset)

  def captureSliceFade(self, sliceNode, numberOfImages, outputDir,
                        outputFilenamePattern):

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir, outputFilenamePattern)

    sliceLogic = self.getSliceLogicFromSliceNode(sliceNode)
    sliceView = self.viewFromNode(sliceNode)
    compositeNode = sliceLogic.GetSliceCompositeNode()
    originalForegroundOpacity = compositeNode.GetForegroundOpacity()
    startForegroundOpacity = 0.0
    endForegroundOpacity = 1.0
    opacityStepSize = 2 * (endForegroundOpacity - startForegroundOpacity) / (numberOfImages - 1)
    for offsetIndex in range(numberOfImages):
      filename = filePathPattern % offsetIndex
      self.addLog("Write "+filename)
      if offsetIndex < numberOfImages/2:
        # fade from start
        compositeNode.SetForegroundOpacity(startForegroundOpacity + offsetIndex * opacityStepSize)
      else:
        # fade to start
        compositeNode.SetForegroundOpacity(startForegroundOpacity + (numberOfImages-offsetIndex) * opacityStepSize)
      self.captureImageFromView(sliceView, filename)

    compositeNode.SetForegroundOpacity(originalForegroundOpacity)

  def capture3dViewRotation(self, viewNode, startRotation, endRotation, numberOfImages, outputDir,
                        outputFilenamePattern):
    """
    Acquire a set of screenshots of the 3D view while rotating it.
    """

    if not os.path.exists(outputDir):
      os.makedirs(outputDir)
    filePathPattern = os.path.join(outputDir, outputFilenamePattern)

    renderView = self.viewFromNode(viewNode)

    # Save original orientation and go to start orientation
    originalPitchRollYawIncrement = renderView.pitchRollYawIncrement
    originalYawDirection = renderView.yawDirection
    renderView.setPitchRollYawIncrement(startRotation)
    renderView.yawDirection = renderView.YawLeft
    renderView.yaw()

    # Rotate step-by-step
    rotationStepSize = (endRotation + startRotation) / (numberOfImages - 1)
    renderView.setPitchRollYawIncrement(rotationStepSize)
    renderView.yawDirection = renderView.YawRight
    for offsetIndex in range(numberOfImages):
      filename = filePathPattern % offsetIndex
      self.addLog("Write " + filename)
      self.captureImageFromView(renderView, filename)
      renderView.yaw()

    # Restore original orientation and rotation step size & direction
    renderView.yawDirection = renderView.YawLeft
    renderView.yaw()
    renderView.setPitchRollYawIncrement(endRotation)
    renderView.yaw()
    renderView.setPitchRollYawIncrement(originalPitchRollYawIncrement)
    renderView.yawDirection = originalYawDirection

  def captureSequence(self, viewNode, sequenceBrowserNode, sequenceStartIndex,
                        sequenceEndIndex, numberOfImages, outputDir, outputFilenamePattern):
    """
    Acquire a set of screenshots of a view while iterating through a sequence.
    """

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
      self.captureImageFromView(renderView, filename)

    sequenceBrowserNode.SetSelectedItemNumber(originalSelectedItemNumber)


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
    ffmpegParams += filter(None, extraOptions.split(' '))
    ffmpegParams.append(outputVideoFilePath)

    logging.debug("ffmpeg parameters: "+repr(ffmpegParams))

    import subprocess
    p = subprocess.Popen(ffmpegParams, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate()
    if p.returncode != 0:
      self.addLog("ffmpeg error output: " + output[1])
      raise ValueError("ffmpeg returned with error")
    else:
      self.addLog("Video export succeeded to file: "+outputVideoFilePath)
      logging.debug("ffmpeg standard output: " + output[0])
      logging.debug("ffmpeg error output: " + output[1])

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
    sampleDataLogic = SampleData.SampleDataLogic()
    self.image1 = sampleDataLogic.downloadMRBrainTumor1()
    self.image2 = sampleDataLogic.downloadMRBrainTumor2()

    # make the output volume appear in all the slice views
    selectionNode = slicer.app.applicationLogic().GetSelectionNode()
    selectionNode.SetReferenceActiveVolumeID(self.image1.GetID())
    selectionNode.SetReferenceSecondaryVolumeID(self.image2.GetID())
    slicer.app.applicationLogic().PropagateVolumeSelection(1)

    # Show slice and 3D views
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    for sliceViewNodeId in ['vtkMRMLSliceNodeRed', 'vtkMRMLSliceNodeYellow', 'vtkMRMLSliceNodeGreen']:
      slicer.util.getNode(sliceViewNodeId).SetSliceVisible(True)

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
    viewNode = slicer.util.getNode('vtkMRMLSliceNodeRed')
    self.assertIsNotNone(viewNode)
    self.logic.captureSliceSweep(viewNode, -125, 75, self.numberOfImages, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing SliceSweep completed successfully')

  def test_SliceFade(self):
    self.delayDisplay("Testing SliceFade")
    viewNode = slicer.util.getNode('vtkMRMLSliceNodeRed')
    self.assertIsNotNone(viewNode)
    self.logic.captureSliceFade(viewNode, self.numberOfImages, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing SliceFade completed successfully')

  def test_3dViewRotation(self):
    self.delayDisplay("Testing 3D view rotation")
    viewNode = slicer.util.getNode('vtkMRMLViewNode1')
    self.assertIsNotNone(viewNode)
    self.logic.capture3dViewRotation(viewNode, 180, 180, self.numberOfImages, self.tempDir, self.imageFileNamePattern)
    self.verifyAndDeleteWrittenFiles()
    self.delayDisplay('Testing 3D view rotation completed successfully')
