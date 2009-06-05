set filename "C:/Data/EndoNav/savedleftrenal_VideoFlag.vtk"
set outfile "C:/Data/EndoNav/series/trackImg"

vtkStructuredPointsReader Reader

Reader SetFileName $filename
Reader Update
[Reader GetOutput] UpdateData

set vol [Reader GetOutput]
set field [$vol GetFieldData]
set reg [$field GetArray "Registration"]
set calib [$field GetArray "Calibration"]
set sensorData [$field GetArray "SensorPositions"]

vtkImageLuminance ImageLuminance
ImageLuminance SetInput [Reader GetOutput]
ImageLuminance Update

set img [ImageLuminance GetOutput]
$img UpdateData
set dims [$img GetDimensions] 
set nslices [lindex $dims 2]
set sp [$img GetSpacing]

vtkImageReslice Reslice
Reslice SetResliceAxesDirectionCosines 1 0 0 0 1 0 0 0 1 
Reslice SetOutputSpacing [lindex $sp 0] [lindex $sp 1] 1.0 
Reslice SetOutputDimensionality 2 
Reslice SetInput $img
Reslice SetOutputExtent 0 [expr [lindex $dims 0]-1] 0 [expr [lindex $dims 0]-1] 0 1 

vtkStructuredPointsWriter writer
writer SetFileTypeToBinary
vtkDoubleArray sensor

for {set i 0} {$i < $nslices} {incr i} {
    Reslice SetResliceAxesOrigin 0 0 $i
    Reslice Update
    set slc [Reslice GetOutput]
    $slc UpdateData

    set outfield [$slc GetFieldData]
#    $outfield AddArray $reg
#    $outfield AddArray $calib
#    $sensorData GetData $i $i 0 5 sensor
#    $outfield AddArray sensor

    $slc UpdateData

    writer SetFileName ${outfile}${i}.vtk
    writer SetInput $slc
    writer Write
}
