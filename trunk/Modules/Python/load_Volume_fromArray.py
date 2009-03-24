XML = """<?xml version="1.0" encoding="utf-8"?>
<executable>
  <category>Python Modules</category>
  <title>Python Load Volume from NUMPY File</title>
  <description>Python module</description>
  <version>0.1.0.$Revision: 1892 $(alpha)</version>
  <documentation-url></documentation-url>
  <contributor>Julien von Siebenthal</contributor>
  
   <parameters>
    <label>Numpy File</label>
    <description>Parameters for Numpy files</description>
    <file>
      <name>inFilename</name>
      <longflag>inFilename</longflag>
      <description>File containing numpy parameter</description>
      <label>Numpy Params file</label>
    </file>

    <file>
      <name>dataFilename</name>
      <longflag>dataFilename</longflag>
      <description>File containing numpy data</description>
      <label>Numpy Data file</label>
    </file>

    <file>
      <name>ijkFilename</name>
      <longflag>ijkFilename</longflag>
      <description>File containing numpy trafo</description>
      <label>Numpy Trafo file</label>
    </file>


  </parameters>

</executable>
"""

import os, time, numpy

vtk_types = { 2:numpy.int8, 3:numpy.uint8, 4:numpy.int16,  5:numpy.uint16,  6:numpy.int32,  7:numpy.uint32,  10:numpy.float32,  11:numpy.float64 }
numpy_sizes = { numpy.int8:1, numpy.uint8:1, numpy.int16:2,  numpy.uint16:2,  numpy.int32:4,  numpy.uint32:4,  numpy.float32:4,  numpy.float64:8 }
numpy_nrrd_names = { 'int8':'char', 'uint8':'unsigned char', 'int16':'short',  'uint16':'ushort',  'int32':'int',  'uint32':'uint',  'float32':'float',  'float64':'double' }
numpy_vtk_types = { 'int8':'2', 'uint8':'3', 'int16':'4',  'uint16':'5',  'int32':'6',  'uint32':'7',  'float32':'10',  'float64':'11' }

def Execute (inFilename="", dataFilename="", ijkFilename=""):
  Slicer = __import__ ( "Slicer" )
  slicer = Slicer.slicer
  scene = slicer.MRMLScene

  if inFilename == "" or inFilename.split('.')[1]!='in':
     return

  if dataFilename == "" or dataFilename.split('.')[1]!='data':
     return

  if ijkFilename == "" or ijkFilename.split('.')[1]!='ijk':
     return


  # take dimensions of the image
  print 'Load dims from : ', inFilename
  dims = numpy.fromfile(inFilename, 'uint16')

  print 'Dims of data : ', dims 

  ijk = numpy.fromfile(ijkFilename, 'float')
  ijk = ijk.reshape(4, 4)

  #if len(dims) == 5:
  #  dtype = vtk_types [ int(dims[4]) ]
  #  data = numpy.fromfile(dataFilename, dtype) 
  #  data = data.reshape(dims[0], dims[1], dims[2], dims[3])

  if len(dims) == 6:
    dtype = vtk_types [ int(dims[5]) ]
    data = numpy.fromfile(dataFilename, dtype) 
    data = data.reshape(dims[2], dims[1], dims[0], 9)


    shape = data.shape
    dtype = data.dtype

    print 'Data shape : ', shape
    print 'Data type : ', dtype

    r1 = slicer.vtkMRMLDiffusionTensorVolumeNode()
    r11 = slicer.vtkMRMLDiffusionTensorVolumeDisplayNode()
    r11.ScalarVisibilityOff()
    #r11.SetScalarRange(data.min(), data.max())
    scene.AddNode(r11)
    #scene.AddNodeNoNotify(r11)

    r1.AddAndObserveDisplayNodeID(r11.GetName())
    #r1.AddAndObserveDisplayNodeID(r11.GetID())
    r11.AddSliceGlyphDisplayNodes(r1)

    tensorImage = slicer.vtkImageData()
    tensorImage.SetDimensions(shape[0], shape[1], shape[2])
    #tensorImage.SetNumberOfScalarComponents(1)
    #tensorImage.SetScalarTypeToFloat()
    #tensorImage.AllocateScalars()
    tensorImage.GetPointData().SetScalars(None)
    tensorImage.Update()
    
    trafo = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    ijk = numpy.dot(trafo, ijk)

    mat = slicer.vtkMatrix4x4()
    for i in range(4):
     for j in range(4):
        mat.SetElement(i,j, ijk[i,j])


    r1.SetAndObserveImageData(tensorImage)
    r1.SetOrigin(ijk[0][3], ijk[1][3], ijk[2][3])
    r1.SetSpacing(ijk[0][0], ijk[1][1], ijk[2][2])
    r1.SetIJKToRASMatrix(mat)
    r1.SetMeasurementFrameMatrix( mat )
    r1.GetImageData().SetScalarTypeToFloat()

    scene.AddNode(r1)

    tensorArray = slicer.vtkFloatArray()
    tensorArray.SetNumberOfComponents(9)
    tensorArray.SetNumberOfTuples(shape[0]*shape[1]*shape[2])
  
    #for i in range(shape[0]*shape[1]*shape[2]):
    #   tensorArray.SetComponent(i, 0, dataL[i, 1])
    #   tensorArray.SetComponent(i, 1, dataL[i, 2])
    #   tensorArray.SetComponent(i, 2, dataL[i, 3])
    #   tensorArray.SetComponent(i, 3, dataL[i, 2])
    #   tensorArray.SetComponent(i, 4, dataL[i, 4])
    #   tensorArray.SetComponent(i, 5, dataL[i, 5])
    #   tensorArray.SetComponent(i, 6, dataL[i, 3])
    #   tensorArray.SetComponent(i, 7, dataL[i, 5])
    #   tensorArray.SetComponent(i, 8, dataL[i, 6])

    r1.GetImageData().GetPointData().SetTensors(tensorArray)
    r1.GetImageData().GetPointData().GetTensors().SetName('NRRDImage')

    tmp = r1.GetImageData().GetPointData().GetTensors().ToArray()
    dataL = numpy.reshape(data, (shape[0]*shape[1]*shape[2], 9))
    tmp[:] = dataL[:]


    r1.GetDisplayNode().SetDefaultColorMap()
    r1.Modified()


  elif len(dims) == 4:
    dtype = vtk_types [ int(dims[3]) ]
    data = numpy.fromfile(dataFilename, dtype) 
    data = data.reshape(dims[2], dims[1], dims[0])

    shape = data.shape
    dtype = data.dtype

    print 'Data shape : ', shape
    print 'Data type : ', dtype

    r1 = slicer.vtkMRMLScalarVolumeNode()
    r11 = slicer.vtkMRMLScalarVolumeDisplayNode()
    r11.ScalarVisibilityOn()
    #r11.SetScalarRange(data.min(), data.max())
    scene.AddNode(r11)

    r1.AddAndObserveDisplayNodeID(r11.GetName())

    imgD = slicer.vtkImageData()
    imgD.SetDimensions(shape[0], shape[1], shape[2])
    imgD.SetNumberOfScalarComponents(1)
    imgD.SetScalarTypeToFloat()
    imgD.AllocateScalars()

    trafo = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    ijk = numpy.dot(trafo, ijk)
    
    mat = slicer.vtkMatrix4x4()
    for i in range(4):
     for j in range(4):
        mat.SetElement(i,j, ijk[i,j])


    r1.SetAndObserveImageData(imgD)
    r1.SetOrigin(ijk[0][3], ijk[1][3], ijk[2][3])
    r1.SetSpacing(ijk[0][0], ijk[1][1], ijk[2][2])
    r1.SetIJKToRASMatrix(mat)

    scene.AddNode(r1)

    tmp = r1.GetImageData().GetPointData().GetScalars().ToArray()
    dataL = numpy.reshape(data, (shape[0]*shape[1]*shape[2], 1))
    tmp[:] = dataL[:]

    r1.GetDisplayNode().SetDefaultColorMap()
    r1.Modified()

  else:
    return  
  
  # should return if dims is different

   

  return

