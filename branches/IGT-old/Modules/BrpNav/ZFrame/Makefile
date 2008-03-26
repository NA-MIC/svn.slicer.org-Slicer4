#NTDIR   = /home/junichi/igtdev/NaviTrack/branches/nt-brp
NTDIR   = /projects/igtdev/tokuda/NaviTrack/branches/nt-brp
CXX     = g++ -g
#ITKDIR  = /home/junichi/igtdev/InsightToolkit-3.2.0
ITKDIR  = /projects/igtdev/tokuda/Insight
INCDIR  = -I${NTDIR}/include -I${ITKDIR} -I${ITKDIR}/Code/Common -I${ITKDIR}/Code/Numerics -I${ITKDIR}/Code/IO -I${ITKDIR}/Utilities/MetaIO -I${ITKDIR}/Utilities/NrrdIO -I${ITKDIR}/Utilities/DICOMParser -I${ITKDIR}/Utilities/expat -I${ITKDIR}/Utilities/nifti/niftilib -I${ITKDIR}/Utilities/nifti/znzlib -I${ITKDIR}/Utilities/itkExtHdrs -I${ITKDIR}/Utilities -I${ITKDIR}/Utilities/vxl/vcl -I${ITKDIR}/Utilities/vxl/core -I${ITKDIR}/Utilities/gdcm -I${ITKDIR}/Utilities/gdcm/src -I/projects/igtdev/tokuda/Insight/Utilities/vxl/v3p/netlib

LIBDIR  = -L. -L${NTDIR} -L${ITKDIR}/bin
LIBS   = -lNaviTrack -lITKCommon -lITKIO -lITKNrrdIO -litkgdcm -litkjpeg12 -litkjpeg16 -litkopenjpeg -litkpng -litktiff -litkjpeg8 -lITKSpatialObject -lITKCommon -lITKNumerics -litkvnl_inst -litkvnl_algo -litkv3p_netlib -litkvnl -litkvcl -lm -lpthread -lITKMetaIO -litksys -lITKDICOMParser -lITKEXPAT -lITKniftiio -lITKznz -litkzlib -lITKStatistics -lITKAlgorithms

PROGRAMS= zframe

### Program section
all: ${PROGRAMS}

zframe: zframe.o pgmimage.o
	${CXX} -o $@ $? ${CXXFLAGS} ${LIBDIR} ${LIBS}

zframe.o: zframe.cxx
	${CXX} -Wall ${INCDIR}  -c $< 

pgmimage.o: pgmimage.cxx
	${CXX} -Wall ${INCDIR} -c $<

clean:
	rm -f *.o ${PROGRAMS}

