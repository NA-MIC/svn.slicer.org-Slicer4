import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

#
# SubjectHierarchyCorePluginsSelfTest
#

class SubjectHierarchyCorePluginsSelfTest(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    parent.title = "SubjectHierarchyCorePluginsSelfTest"
    parent.categories = ["Testing.TestCases"]
    parent.dependencies = ["SubjectHierarchy"]
    parent.contributors = ["Csaba Pinter (Queen's)"]
    parent.helpText = """
    This is a self test for the Subject hierarchy core plugins.
    """
    parent.acknowledgementText = """This file was originally developed by Csaba Pinter, PerkLab, Queen's University and was supported through the Applied Cancer Research Unit program of Cancer Care Ontario with funds provided by the Ontario Ministry of Health and Long-Term Care""" # replace with organization, grant and thanks.
    self.parent = parent

    # Add this test to the SelfTest module's list for discovery when the module
    # is created.  Since this module may be discovered before SelfTests itself,
    # create the list if it doesn't already exist.
    try:
      slicer.selfTests
    except AttributeError:
      slicer.selfTests = {}
    slicer.selfTests['SubjectHierarchyCorePluginsSelfTest'] = self.runTest

  def runTest(self):
    tester = SubjectHierarchyCorePluginsSelfTestTest()
    tester.runTest()

#
# SubjectHierarchyCorePluginsSelfTestWidget
#

class SubjectHierarchyCorePluginsSelfTestWidget(ScriptedLoadableModuleWidget):
  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

#
# SubjectHierarchyCorePluginsSelfTestLogic
#

class SubjectHierarchyCorePluginsSelfTestLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget
  """
  def __init__(self):
    pass


class SubjectHierarchyCorePluginsSelfTestTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

    self.delayMs = 700

    #TODO: Comment out (sample code for debugging)
    #logFile = open('d:/pyTestLog.txt', 'w')
    #logFile.write(repr(slicer.modules.SubjectHierarchyCorePluginsSelfTest) + '\n')
    #logFile.write(repr(slicer.modules.subjecthierarchy) + '\n')
    #logFile.close()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_SubjectHierarchyCorePluginsSelfTest_FullTest1()

  # ------------------------------------------------------------------------------
  def test_SubjectHierarchyCorePluginsSelfTest_FullTest1(self):
    # Check for SubjectHierarchy module
    self.assertTrue( slicer.modules.subjecthierarchy )

    # Switch to subject hierarchy module so that the changes can be seen as the test goes
    slicer.util.selectModule('SubjectHierarchy')

    self.section_SetupPathsAndNames()
    self.section_MarkupRole()
    self.section_ChartRole()
    self.section_CloneNode()

  # ------------------------------------------------------------------------------
  def section_SetupPathsAndNames(self):
    # Set constants
    self.invalidItemID = slicer.vtkMRMLSubjectHierarchyNode.GetInvalidItemID()
    self.sampleMarkupName = 'SampleMarkup'
    self.sampleChartName = 'SampleChart'
    self.studyItemID = self.invalidItemID
    self.cloneNodeNamePostfix = slicer.qSlicerSubjectHierarchyCloneNodePlugin().getCloneNodeNamePostfix()

  # ------------------------------------------------------------------------------
  def section_MarkupRole(self):
    self.delayDisplay("Markup role",self.delayMs)

    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    self.assertIsNotNone( shNode )

    # Create sample markups node
    markupsNode = slicer.vtkMRMLMarkupsFiducialNode()
    slicer.mrmlScene.AddNode(markupsNode)
    markupsNode.SetName(self.sampleMarkupName)
    fiducialPosition = [100.0, 0.0, 0.0]
    markupsNode.AddFiducialFromArray(fiducialPosition)
    markupsShItemID = shNode.GetItemByDataNode(markupsNode)
    self.assertNotEqual( markupsShItemID, self.invalidItemID )
    self.assertEqual( shNode.GetItemOwnerPluginName(markupsShItemID), 'Markups' )

    # Create patient and study
    patientItemID = shNode.CreateItem(shNode.GetSceneItemID(), 'Patient', slicer.vtkMRMLSubjectHierarchyConstants.GetDICOMLevelPatient())
    self.studyItemID = shNode.CreateItem(patientItemID, 'Study', slicer.vtkMRMLSubjectHierarchyConstants.GetDICOMLevelStudy())

    # Add markups under study
    markupsShItemID2 = shNode.CreateItem(self.studyItemID, markupsNode)
    self.assertEqual( markupsShItemID, markupsShItemID2 )
    self.assertEqual( shNode.GetItemParent(markupsShItemID), self.studyItemID )
    self.assertEqual( shNode.GetItemOwnerPluginName(markupsShItemID), 'Markups' )

  # ------------------------------------------------------------------------------
  def section_ChartRole(self):
    self.delayDisplay("Chart role",self.delayMs)

    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    self.assertIsNotNone( shNode )

    # Create sample chart node
    chartNode = slicer.vtkMRMLChartNode()
    slicer.mrmlScene.AddNode(chartNode)
    chartNode.SetName(self.sampleChartName)
    chartShItemID = shNode.GetItemByDataNode(chartNode)
    self.assertNotEqual( chartShItemID, self.invalidItemID )
    self.assertEqual( shNode.GetItemOwnerPluginName(chartShItemID), 'Charts' )

    # Add chart under study
    chartShItemID2 = shNode.CreateItem(self.studyItemID, chartNode)
    self.assertEqual( chartShItemID, chartShItemID2 )
    self.assertEqual( shNode.GetItemParent(chartShItemID), self.studyItemID )
    self.assertEqual( shNode.GetItemOwnerPluginName(chartShItemID), 'Charts' )

  # ------------------------------------------------------------------------------
  def section_CloneNode(self):
    self.delayDisplay("Clone node",self.delayMs)

    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    self.assertIsNotNone( shNode )

    markupsNode = slicer.util.getNode(self.sampleMarkupName)
    markupsShItemID = shNode.GetItemByDataNode(markupsNode)

    self.assertNotEqual( markupsShItemID, self.invalidItemID )
    self.assertIsNotNone( shNode.GetItemDataNode(markupsShItemID) )

    # Add storage node for markups node to test cloning those
    markupsStorageNode = slicer.vtkMRMLMarkupsFiducialStorageNode()
    slicer.mrmlScene.AddNode(markupsStorageNode)
    markupsNode.SetAndObserveStorageNodeID(markupsStorageNode.GetID())

    # Get clone node plugin
    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    self.assertIsNotNone( pluginHandler )

    cloneNodePlugin = pluginHandler.pluginByName('CloneNode')
    self.assertIsNotNone( cloneNodePlugin )

    # Set markup node as current (i.e. selected in the tree) for clone
    pluginHandler.setCurrentItem(markupsShItemID)

    # Get clone node context menu action and trigger
    cloneNodePlugin.itemContextMenuActions()[0].activate(qt.QAction.Trigger)

    self.assertEqual( slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLMarkupsFiducialNode'), 2 )
    self.assertEqual( slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLMarkupsDisplayNode'), 2 )
    self.assertEqual( slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLMarkupsFiducialStorageNode'), 2 )

    clonedMarkupsName = self.sampleMarkupName + self.cloneNodeNamePostfix
    clonedMarkupsNode = slicer.util.getNode(clonedMarkupsName)
    self.assertIsNotNone( clonedMarkupsNode )
    clonedMarkupsShItemID = shNode.GetItemChildWithName(self.studyItemID, clonedMarkupsName)
    self.assertNotEqual( clonedMarkupsShItemID, self.invalidItemID )
    self.assertIsNotNone( clonedMarkupsNode.GetDisplayNode() )
    self.assertIsNotNone( clonedMarkupsNode.GetStorageNode() )

    inSameStudy = slicer.vtkSlicerSubjectHierarchyModuleLogic.AreItemsInSameBranch(
      shNode, markupsShItemID, clonedMarkupsShItemID, slicer.vtkMRMLSubjectHierarchyConstants.GetDICOMLevelStudy())
    self.assertTrue( inSameStudy )
