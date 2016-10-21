import os
import vtk, qt, ctk, slicer
import logging
from SegmentEditorEffects import *

class SegmentEditorAutoCompleteEffect(AbstractScriptedSegmentEditorEffect):
  """ AutoCompleteEffect is an effect that can create a full segmentation
      from a partial segmentation (not all slices are segmented or only
      part of the target structures are painted).
  """

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Auto-complete'
    # Indicates that effect does not operate on one segment, but the whole segmentation.
    # This means that while this effect is active, no segment can be selected
    scriptedEffect.perSegment = False
    AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)
    # Stores merged labelmap image geometry (voxel data is not allocated)
    self.mergedLabelmapGeometryImage = None
    self.selectedSegmentIds = None
    self.clippedMasterImageData = None
    self.growCutFilter = None
    # Observation for auto-update
    self.observedSegmentation = None
    self.segmentationNodeObserverTags = []

  def __del__(self, scriptedEffect):
    super(SegmentEditorAutoCompleteEffect,self).__del__()
    self.observeSegmentation(False)

  def clone(self):
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self):
    iconPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons/AutoComplete.png')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """Create a complete segmentation from a partial segmentation from partial segmentation \
created using other effects. Masking settings are bypassed. If segments overlap, segment higher in \
the segments table will have priority."""

  def setupOptionsFrame(self):
    self.methodSelectorComboBox = qt.QComboBox()
    self.methodSelectorComboBox.addItem("Fill between parallel slices", MORPHOLOGICAL_SLICE_INTERPOLATION)
    self.methodSelectorComboBox.addItem("Grow from seeds", GROWCUT)
    self.methodSelectorComboBox.setToolTip("""<html>Auto-complete methods:<ul style="margin: 0">
<li><b>Fill between slices:</b> Perform complete segmentation on selected slices using any editor effect.
The complete segmentation will be created by interpolating segmentations on slices that were skipped.</li>
<li><b>Expand segments:</b> Create segments using any editor effect: one segment inside
each each region that should belong to a separate segment. Segments will be expanded to create
a complete segmentation, taking into account the master volume content. Minimum two segments are required.
(see http://insight-journal.org/browse/publication/977)</li>
</ul></html>""")
    self.scriptedEffect.addLabeledOptionsWidget("Method:", self.methodSelectorComboBox)

    self.autoUpdateCheckBox = qt.QCheckBox("Auto-update")
    self.autoUpdateCheckBox.setToolTip("Auto-update results preview when input segments change.")
    self.autoUpdateCheckBox.setChecked(False)
    self.autoUpdateCheckBox.setEnabled(False)

    self.previewButton = qt.QPushButton("Initialize")
    self.previewButton.objectName = self.__class__.__name__ + 'Preview'
    self.previewButton.setToolTip("Preview complete segmentation")
    # qt.QSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    # fails on some systems, therefore set the policies using separate method calls
    qSize = qt.QSizePolicy()
    qSize.setHorizontalPolicy(qt.QSizePolicy.Expanding)
    self.previewButton.setSizePolicy(qSize)

    previewFrame = qt.QHBoxLayout()
    previewFrame.addWidget(self.autoUpdateCheckBox)
    previewFrame.addWidget(self.previewButton)
    self.scriptedEffect.addLabeledOptionsWidget("Preview:", previewFrame)

    self.previewOpacitySlider = ctk.ctkSliderWidget()
    self.previewOpacitySlider.setToolTip("Adjust visibility of results preview.")
    self.previewOpacitySlider.minimum = 0
    self.previewOpacitySlider.maximum = 1.0
    self.previewOpacitySlider.value = 0.0
    self.previewOpacitySlider.singleStep = 0.05
    self.previewOpacitySlider.pageStep = 0.1
    self.previewOpacitySlider.spinBoxVisible = False

    displayFrame = qt.QHBoxLayout()
    displayFrame.addWidget(qt.QLabel("inputs"))
    displayFrame.addWidget(self.previewOpacitySlider)
    displayFrame.addWidget(qt.QLabel("results"))
    self.scriptedEffect.addLabeledOptionsWidget("Display:", displayFrame)

    self.cancelButton = qt.QPushButton("Cancel")
    self.cancelButton.objectName = self.__class__.__name__ + 'Cancel'
    self.cancelButton.setToolTip("Clear preview and cancel auto-complete")

    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.objectName = self.__class__.__name__ + 'Apply'
    self.applyButton.setToolTip("Replace segments by previewed result")

    finishFrame = qt.QHBoxLayout()
    finishFrame.addWidget(self.cancelButton)
    finishFrame.addWidget(self.applyButton)
    self.scriptedEffect.addOptionsWidget(finishFrame)

    self.methodSelectorComboBox.connect("currentIndexChanged(int)", self.updateMRMLFromGUI)
    self.previewButton.connect('clicked()', self.onPreview)
    self.cancelButton.connect('clicked()', self.onCancel)
    self.applyButton.connect('clicked()', self.onApply)
    self.previewOpacitySlider.connect("valueChanged(double)", self.updateMRMLFromGUI)
    self.autoUpdateCheckBox.connect("stateChanged(int)", self.updateMRMLFromGUI)

  def createCursor(self, widget):
    # Turn off effect-specific cursor for this effect
    return slicer.util.mainWindow().cursor

  def setMRMLDefaults(self):
    self.scriptedEffect.setParameterDefault("AutoCompleteMethod", MORPHOLOGICAL_SLICE_INTERPOLATION)

  def onSegmentationModified(self, caller, event):
    import vtkSegmentationCorePython as vtkSegmentationCore

    if event == vtkSegmentationCore.vtkSegmentation.SegmentAdded or event == vtkSegmentationCore.vtkSegmentation.SegmentRemoved:
      self.onCancel()
      return

    if not self.autoUpdateCheckBox.isChecked():
      # just in case a queued request comes through
      return

    self.onPreview()

  def observeSegmentation(self, observationEnabled):
    import vtkSegmentationCorePython as vtkSegmentationCore
    segmentation = self.scriptedEffect.parameterSetNode().GetSegmentationNode().GetSegmentation()
    if observationEnabled and self.observedSegmentation == segmentation:
      return
    if not observationEnabled and not self.observedSegmentation:
      return
    # Need to update the observer
    # Remove old observer
    if self.observedSegmentation:
      for tag in self.segmentationNodeObserverTags:
        self.observedSegmentation.RemoveObserver(tag)
      self.segmentationNodeObserverTags = []
      self.observedSegmentation = None
    # Add new observer
    if observationEnabled and segmentation is not None:
      self.observedSegmentation = segmentation
      observedEvents = [
        vtkSegmentationCore.vtkSegmentation.SegmentAdded,
        vtkSegmentationCore.vtkSegmentation.SegmentRemoved,
        vtkSegmentationCore.vtkSegmentation.SegmentModified,
        vtkSegmentationCore.vtkSegmentation.MasterRepresentationModified ]
      for eventId in observedEvents:
        self.segmentationNodeObserverTags.append(self.observedSegmentation.AddObserver(eventId, self.onSegmentationModified))

  def updateGUIFromMRML(self):
    methodIndex = self.methodSelectorComboBox.findData(self.scriptedEffect.parameter("AutoCompleteMethod"))
    wasBlocked = self.methodSelectorComboBox.blockSignals(True)
    self.methodSelectorComboBox.setCurrentIndex(methodIndex)
    self.methodSelectorComboBox.blockSignals(wasBlocked)

    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)

    self.methodSelectorComboBox.setEnabled(previewNode is None)
    self.cancelButton.setEnabled(previewNode is not None)
    self.applyButton.setEnabled(previewNode is not None)

    self.previewOpacitySlider.setEnabled(previewNode is not None)
    if previewNode:
      wasBlocked = self.previewOpacitySlider.blockSignals(True)
      self.previewOpacitySlider.value = self.getPreviewOpacity()
      self.previewOpacitySlider.blockSignals(wasBlocked)
      self.previewButton.text = "Update"
      self.autoUpdateCheckBox.setEnabled(True)
      self.observeSegmentation(self.autoUpdateCheckBox.isChecked())
    else:
      self.previewButton.text = "Initialize"
      self.autoUpdateCheckBox.setEnabled(False)
      self.observeSegmentation(False)

    autoUpdate = qt.Qt.Unchecked if self.scriptedEffect.integerParameter("AutoUpdate") == 0 else qt.Qt.Checked
    wasBlocked = self.autoUpdateCheckBox.blockSignals(True)
    self.autoUpdateCheckBox.setCheckState(autoUpdate)
    self.autoUpdateCheckBox.blockSignals(wasBlocked)

  def updateMRMLFromGUI(self):
    methodIndex = self.methodSelectorComboBox.currentIndex
    method = self.methodSelectorComboBox.itemData(methodIndex)
    self.scriptedEffect.setParameter("AutoCompleteMethod", method)

    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)
    if previewNode:
      self.setPreviewOpacity(self.previewOpacitySlider.value)

    autoUpdate = 1 if self.autoUpdateCheckBox.isChecked() else 0
    self.scriptedEffect.setParameter("AutoUpdate", autoUpdate)

  def onPreview(self):
    slicer.util.showStatusMessage("Running auto-complete...", 2000)
    try:
      # This can be a long operation - indicate it to the user
      qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
      self.preview()
    finally:
      qt.QApplication.restoreOverrideCursor()

  def reset(self):
    self.observeSegmentation(False)
    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)
    if previewNode:
      self.scriptedEffect.parameterSetNode().SetNodeReferenceID(ResultPreviewNodeReferenceRole, None)
      slicer.mrmlScene.RemoveNode(previewNode)
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentationNode.GetDisplayNode().SetOpacity(1.0)
    self.mergedLabelmapGeometryImage = None
    self.selectedSegmentIds = None
    self.clippedMasterImageData = None
    self.growCutFilter = None
    self.updateGUIFromMRML()

  def onCancel(self):
    self.reset()

  def onApply(self):
    import vtkSegmentationCorePython as vtkSegmentationCore
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)

    self.scriptedEffect.saveStateForUndo()

    # Move segments from preview into current segmentation
    segmentIDs = vtk.vtkStringArray()
    previewNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    for index in xrange(segmentIDs.GetNumberOfValues()):
      segmentID = segmentIDs.GetValue(index)
      previewSegment = previewNode.GetSegmentation().GetSegment(segmentID)
      previewSegmentLabelmap = previewSegment.GetRepresentation(vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(previewSegmentLabelmap, segmentationNode, segmentID)
      previewNode.GetSegmentation().RemoveSegment(segmentID) # delete now to limit memory usage

    self.reset()

  def setPreviewOpacity(self, opacity):
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentationNode.GetDisplayNode().SetOpacity(1.0-opacity)
    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)
    if previewNode:
      previewNode.GetDisplayNode().SetOpacity(opacity)

    # Make sure the GUI is up-to-date
    wasBlocked = self.previewOpacitySlider.blockSignals(True)
    self.previewOpacitySlider.value = opacity
    self.previewOpacitySlider.blockSignals(wasBlocked)

  def getPreviewOpacity(self):
    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)
    return previewNode.GetDisplayNode().GetOpacity() if previewNode else 0.6 # default opacity for preview

  def preview(self):
    # Get master volume image data
    import vtkSegmentationCorePython as vtkSegmentationCore
    masterImageData = self.scriptedEffect.masterVolumeImageData()

    # Get segmentation
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()

    method = self.scriptedEffect.parameter("AutoCompleteMethod")

    previewNode = self.scriptedEffect.parameterSetNode().GetNodeReference(ResultPreviewNodeReferenceRole)
    if not previewNode or not self.clippedMasterImageData or not self.mergedLabelmapGeometryImage:
      self.reset()
      # Compute merged labelmap extent (effective extent slightly expanded)
      self.selectedSegmentIds = vtk.vtkStringArray()
      segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(self.selectedSegmentIds)
      minimumNumberOfSegments = 2 if method == GROWCUT else 1
      if self.selectedSegmentIds.GetNumberOfValues() < minimumNumberOfSegments:
        logging.info("Auto-complete operation skipped: at least {0} visible segments are required".format(minimumNumberOfSegments))
        return
      if not self.mergedLabelmapGeometryImage:
        self.mergedLabelmapGeometryImage = vtkSegmentationCore.vtkOrientedImageData()
      commonGeometryString = segmentationNode.GetSegmentation().DetermineCommonLabelmapGeometry(
        vtkSegmentationCore.vtkSegmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS, self.selectedSegmentIds)
      if not commonGeometryString:
        logging.info("Auto-complete operation skipped: all visible segments are empty")
        return
      vtkSegmentationCore.vtkSegmentationConverter.DeserializeImageGeometry(commonGeometryString, self.mergedLabelmapGeometryImage)

      masterImageExtent = masterImageData.GetExtent()
      labelsEffectiveExtent = self.mergedLabelmapGeometryImage.GetExtent()
      margin = [17, 17, 17]
      labelsExpandedExtent = [
        max(masterImageExtent[0], labelsEffectiveExtent[0]-margin[0]),
        min(masterImageExtent[1], labelsEffectiveExtent[1]+margin[0]),
        max(masterImageExtent[2], labelsEffectiveExtent[2]-margin[1]),
        min(masterImageExtent[3], labelsEffectiveExtent[3]+margin[1]),
        max(masterImageExtent[4], labelsEffectiveExtent[4]-margin[2]),
        min(masterImageExtent[5], labelsEffectiveExtent[5]+margin[2]) ]
      self.mergedLabelmapGeometryImage.SetExtent(labelsExpandedExtent)

      previewNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLSegmentationNode')
      previewNode = slicer.mrmlScene.AddNode(previewNode)
      previewNode.CreateDefaultDisplayNodes()
      previewNode.GetDisplayNode().SetVisibility2DOutline(False)
      if segmentationNode.GetParentTransformNode():
        previewNode.SetAndObserveTransformNodeID(segmentationNode.GetParentTransformNode().GetID())
      self.scriptedEffect.parameterSetNode().SetNodeReferenceID(ResultPreviewNodeReferenceRole, previewNode.GetID())
      self.setPreviewOpacity(0.6)

      self.clippedMasterImageData = vtkSegmentationCore.vtkOrientedImageData()
      masterImageClipper = vtk.vtkImageConstantPad()
      masterImageClipper.SetInputData(masterImageData)
      masterImageClipper.SetOutputWholeExtent(self.mergedLabelmapGeometryImage.GetExtent())
      masterImageClipper.Update()
      self.clippedMasterImageData.ShallowCopy(masterImageClipper.GetOutput())
      self.clippedMasterImageData.CopyDirections(self.mergedLabelmapGeometryImage)

    previewNode.GetSegmentation().RemoveAllSegments()
    previewNode.SetName(segmentationNode.GetName()+" preview")

    mergedImage = vtkSegmentationCore.vtkOrientedImageData()
    segmentationNode.GenerateMergedLabelmapForAllSegments(mergedImage,
      vtkSegmentationCore.vtkSegmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS, self.mergedLabelmapGeometryImage, self.selectedSegmentIds)

    # Make a zero-valued volume for the output
    outputLabelmap = vtkSegmentationCore.vtkOrientedImageData()

    if method == MORPHOLOGICAL_SLICE_INTERPOLATION:
        import vtkITK
        interpolator = vtkITK.vtkITKMorphologicalContourInterpolator()
        interpolator.SetInputData(mergedImage)
        interpolator.Update()
        outputLabelmap.DeepCopy(interpolator.GetOutput())

    elif method == GROWCUT:

      import vtkSlicerSegmentationsModuleLogicPython as vtkSlicerSegmentationsModuleLogic
      # if self.growCutFilter exists but previewNode is empty then probably the scene was closed
      # while in preview mode
      if self.growCutFilter:
        self.growCutFilter.SetSeedLabelVolume(mergedImage)
        self.growCutFilter.Update()
      else:
        self.growCutFilter = vtkSlicerSegmentationsModuleLogic.vtkImageGrowCutSegment()
        self.growCutFilter.SetIntensityVolume(self.clippedMasterImageData)
        self.growCutFilter.SetSeedLabelVolume(mergedImage)
        self.growCutFilter.Update()

      outputLabelmap.DeepCopy( self.growCutFilter.GetOutput() )
    else:
      logging.error("Invalid auto-complete method {0}".format(smoothingMethod))
      return

    # Write output segmentation results in segments
    segmentIDs = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(segmentIDs)
    for index in xrange(segmentIDs.GetNumberOfValues()):
      segmentID = segmentIDs.GetValue(index)
      segment = segmentationNode.GetSegmentation().GetSegment(segmentID)
      # Disable save with scene?

      # Get only the label of the current segment from the output image
      thresh = vtk.vtkImageThreshold()
      thresh.ReplaceInOn()
      thresh.ReplaceOutOn()
      thresh.SetInValue(1)
      thresh.SetOutValue(0)
      labelValue = index + 1 # n-th segment label value = n + 1 (background label value is 0)
      thresh.ThresholdBetween(labelValue, labelValue);
      thresh.SetOutputScalarType(vtk.VTK_UNSIGNED_CHAR)
      thresh.SetInputData(outputLabelmap)
      thresh.Update()

      # Write label to segment
      newSegmentLabelmap = vtkSegmentationCore.vtkOrientedImageData()
      newSegmentLabelmap.ShallowCopy(thresh.GetOutput())
      newSegmentLabelmap.CopyDirections(mergedImage)
      newSegment = vtkSegmentationCore.vtkSegment()
      newSegment.SetName(segment.GetName())
      previewNode.GetSegmentation().AddSegment(newSegment, segmentID)
      previewNode.GetDisplayNode().SetSegmentColor(segmentID, segmentationNode.GetDisplayNode().GetSegmentColor(segmentID))
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(newSegmentLabelmap, previewNode, segmentID)

    self.updateGUIFromMRML()


MORPHOLOGICAL_SLICE_INTERPOLATION = 'MORPHOLOGICAL_SLICE_INTERPOLATION'
GROWCUT = 'GROWCUT'

ResultPreviewNodeReferenceRole = "AutoCompleteSegmentationResultPreview"
